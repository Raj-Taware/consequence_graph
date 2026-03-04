"""
Query engine.
Given a target node, computes:
  - blast radius (upstream + downstream subgraph)
  - severity score per affected node
  - auto-depth calculation
  - inferred change type
  - summary hint for LLM context injection
"""
import math
from typing import Optional

from .graph import KnowledgeGraph, NodeType, EdgeType, LIGHTNING_HOOKS


# ── Severity scoring ──────────────────────────────────────────────────────────

HIGH_RISK_EDGE_TYPES = {EdgeType.PRODUCES_TENSOR.value, EdgeType.OVERRIDES_HOOK.value, EdgeType.INHERITS.value}
MEDIUM_RISK_EDGE_TYPES = {EdgeType.READS_CONFIG.value, EdgeType.WRITES_CONFIG.value, EdgeType.CONSUMES_FORMAT.value}


def _node_severity_score(graph: KnowledgeGraph, node_id: str, affected_entries: list[dict]) -> int:
    """Compute a severity score for a TARGET node based on its graph properties."""
    score = 0
    node_data = graph.get_node(node_id) or {}

    # Is a Lightning hook?
    if node_data.get("is_lightning_hook") or node_data.get("name") in LIGHTNING_HOOKS:
        score += 3

    # Has tensor contracts?
    if node_data.get("tensor_shapes"):
        score += 3

    # Has config keys?
    if node_data.get("config_keys"):
        score += 2

    # High in-degree (many things depend on it)
    if graph.in_degree(node_id) > 5:
        score += 2

    # High-severity incoming edges in blast radius
    high_risk_entries = [e for e in affected_entries if e.get("edge_type") in HIGH_RISK_EDGE_TYPES]
    score += min(len(high_risk_entries), 4)

    # Is it a leaf node (no dependents)?
    if graph.out_degree(node_id) == 0:
        score -= 2

    return max(0, score)


def _auto_depth(score: int) -> int:
    return min(4, max(1, math.ceil(score / 3)))


def _infer_change_type(node_data: dict) -> str:
    if not node_data:
        return "unknown"
    if node_data.get("tensor_shapes"):
        return "inferred:tensor_shape"
    if node_data.get("is_lightning_hook"):
        return "inferred:hook_contract"
    if node_data.get("config_keys"):
        return "inferred:config_dependency"
    if node_data.get("node_type") == NodeType.CLASS.value:
        return "inferred:class_interface"
    return "inferred:general"


def _severity_label(score: int) -> str:
    if score >= 7:
        return "critical"
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _entry_severity(entry: dict) -> str:
    w = entry.get("severity_weight", 0)
    depth = entry.get("depth", 1)
    adjusted = w - (depth - 1)  # severity decays with depth
    if adjusted >= 3:
        return "high"
    if adjusted >= 2:
        return "medium"
    return "low"


def _dedup_entries(entries: list[dict]) -> list[dict]:
    """Keep only highest-severity entry per node."""
    seen: dict[str, dict] = {}
    for e in entries:
        nid = e["node"]
        if nid not in seen or e.get("severity_weight", 0) > seen[nid].get("severity_weight", 0):
            seen[nid] = e
    return sorted(seen.values(), key=lambda x: -x.get("severity_weight", 0))


# ── QueryEngine ───────────────────────────────────────────────────────────────

class QueryEngine:
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def impact(self, target: str, depth: Optional[int] = None) -> dict:
        """
        Core query: return full impact report for target node.
        depth=None → auto-compute from severity score.
        """
        # 1. Resolve node — explicit mode so dunders and ambiguous names work for user queries
        candidates = self.graph.resolve_node_id(target, explicit=True)
        if not candidates:
            return {
                "error": f"Node '{target}' not found in graph.",
                "hint": "Run `codegraph index <path>` first, or check spelling.",
                "suggestions": [],
            }

        if len(candidates) > 1:
            # Return disambiguation list
            return {
                "ambiguous": True,
                "query": target,
                "matches": candidates,
                "hint": "Use the full node ID from 'matches' to disambiguate.",
            }

        node_id = candidates[0]
        node_data = self.graph.get_node(node_id) or {}

        # 2. First pass with depth=2 to compute severity
        upstream_raw = self.graph.upstream(node_id, depth=2)
        downstream_raw = self.graph.downstream(node_id, depth=2)
        all_raw = upstream_raw + downstream_raw

        score = _node_severity_score(self.graph, node_id, all_raw)
        resolved_depth = depth if depth is not None else _auto_depth(score)

        # 3. Full traversal with resolved depth
        if resolved_depth != 2:
            upstream_raw = self.graph.upstream(node_id, depth=resolved_depth)
            downstream_raw = self.graph.downstream(node_id, depth=resolved_depth)

        # 4. Enrich each entry
        upstream = self._enrich_entries(upstream_raw)
        downstream = self._enrich_entries(downstream_raw)

        # 5. Dedup
        upstream = _dedup_entries(upstream)
        downstream = _dedup_entries(downstream)

        # 6. Build summary hint
        hint = self._build_hint(node_id, node_data, upstream, downstream, score)

        # 7. Critical path
        critical_path = self._critical_path(node_id, upstream, downstream)

        return {
            "target": node_id,
            "target_meta": {
                "type": node_data.get("node_type"),
                "file": node_data.get("file_path"),
                "line": node_data.get("line_no"),
                "signature": node_data.get("signature"),
                "docstring": node_data.get("docstring"),
                "tensor_shapes": node_data.get("tensor_shapes", {}),
                "config_keys": node_data.get("config_keys", []),
                "is_lightning_hook": node_data.get("is_lightning_hook", False),
                "centrality": self.graph.centrality_score(node_id),
            },
            "change_type": _infer_change_type(node_data),
            "severity": _severity_label(score),
            "severity_score": score,
            "depth_used": resolved_depth,
            "depth_mode": "auto" if depth is None else "manual",
            "blast_radius": {
                "upstream": upstream,
                "downstream": downstream,
                "upstream_count": len(upstream),
                "downstream_count": len(downstream),
            },
            "critical_path": critical_path,
            "llm_context_hint": hint,
        }

    def _enrich_entries(self, entries: list[dict]) -> list[dict]:
        enriched = []
        for e in entries:
            node_data = self.graph.get_node(e["node"]) or {}
            e["severity"] = _entry_severity(e)
            e["name"] = node_data.get("name", e["node"].split(".")[-1])
            e["file"] = node_data.get("file_path", "")
            e["line"] = node_data.get("line_no", 0)
            e["docstring"] = node_data.get("docstring", "")
            e["tensor_shapes"] = node_data.get("tensor_shapes", {})
            e["is_lightning_hook"] = node_data.get("is_lightning_hook", False)
            enriched.append(e)
        return enriched

    def _build_hint(
        self,
        node_id: str,
        node_data: dict,
        upstream: list[dict],
        downstream: list[dict],
        score: int,
    ) -> str:
        name = node_data.get("name", node_id.split(".")[-1])
        lines = [f"Changing `{name}` has a {_severity_label(score)} impact (score: {score})."]

        if upstream:
            up_names = [e["name"] for e in upstream[:3]]
            lines.append(f"It depends on: {', '.join(up_names)}{'...' if len(upstream) > 3 else ''}.")

        if downstream:
            down_names = [e["name"] for e in downstream[:3]]
            lines.append(f"It is depended upon by: {', '.join(down_names)}{'...' if len(downstream) > 3 else ''}.")

        high_risk = [e for e in upstream + downstream if e.get("severity") == "high"]
        if high_risk:
            hr_names = list({e["name"] for e in high_risk[:3]})
            lines.append(f"High-risk propagation through: {', '.join(hr_names)}.")

        tensor_entries = [e for e in upstream + downstream if e.get("tensor_shapes")]
        if tensor_entries:
            lines.append(
                f"Tensor shape contracts are in play — "
                f"{len(tensor_entries)} node(s) have shape annotations that may break."
            )

        hook_entries = [e for e in upstream + downstream if e.get("is_lightning_hook")]
        if hook_entries:
            lines.append(
                f"Lightning hooks are affected: "
                f"{', '.join(e['name'] for e in hook_entries[:3])}."
            )

        return " ".join(lines)

    def _critical_path(
        self,
        node_id: str,
        upstream: list[dict],
        downstream: list[dict],
    ) -> list[str]:
        """Return the highest-severity chain: top upstream → target → top downstream."""
        path = []
        if upstream:
            top_up = max(upstream, key=lambda e: e.get("severity_weight", 0))
            path.append(top_up["node"])
        path.append(node_id)
        if downstream:
            top_down = max(downstream, key=lambda e: e.get("severity_weight", 0))
            path.append(top_down["node"])
        return path
