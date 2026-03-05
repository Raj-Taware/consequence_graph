"""
consequencegraph visual server.
Serves the knowledge graph as a REST API + interactive D3.js frontend.

Run:
    python server.py --path ./neural_lam --preset neural_lam
    then open http://localhost:7842
"""
import json
import os
import re
import sys
import argparse
import subprocess
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.graph import KnowledgeGraph
from core.indexer import Indexer
from core.enricher import Enricher
from core.query import QueryEngine
from output.llm_context import format_impact_as_context

try:
    from fastapi import FastAPI, HTTPException, Request, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────

PRODUCTION = os.environ.get("CONSEQUENCEGRAPH_ENV") == "production"
RATE_LIMIT_WINDOW = 60        # seconds
RATE_LIMIT_MAX    = 60        # requests per window per IP
RATE_BUCKETS_MAX  = 10_000   # cap tracked IPs to prevent memory leak (H2)
CORS_ORIGIN = os.environ.get("CORS_ALLOW_ORIGIN", "*")  # H3: restrict in prod via render.yaml


# ── Rate limiter ──────────────────────────────────────────────────────────────

_rate_buckets: dict = defaultdict(list)

def _get_real_ip(request) -> str:
    """Real client IP behind proxy/LB (H2: X-Forwarded-For)."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    bucket = _rate_buckets[ip]
    _rate_buckets[ip] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_buckets[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_buckets[ip].append(now)
    if len(_rate_buckets) > RATE_BUCKETS_MAX:  # H2: evict oldest
        oldest = next(iter(_rate_buckets))
        del _rate_buckets[oldest]
    return True


# ── Graph state ───────────────────────────────────────────────────────────────

_graph: KnowledgeGraph = None
_engine: QueryEngine = None
_index_path: str = None


def get_graph() -> KnowledgeGraph:
    return _graph


def get_engine() -> QueryEngine:
    return _engine


def build_or_load_graph(path: str, preset: str = None, force_reindex: bool = False):
    global _graph, _engine, _index_path
    _index_path = os.path.abspath(path)
    # C4: prefer <path>/cache.json (pre-built on Render); fall back to dev-mode path
    _path_cache = os.path.join(_index_path, "cache.json")
    cache = _path_cache if os.path.isfile(_path_cache) else os.path.join(os.getcwd(), ".consequencegraph", "cache.json")

    _graph = KnowledgeGraph()
    _graph.CACHE_FILE = cache

    if not force_reindex and _graph.load(cache):
        print(f"[consequencegraph server] Loaded from cache: {_graph.node_count()} nodes")
    else:
        print(f"[consequencegraph server] Indexing: {path}")
        Indexer(_graph, path).index_path(path)
        Enricher(_graph).run()
        if preset == "neural_lam":
            from presets.neural_lam import apply
            apply(_graph)
        _graph.save(cache)
        print(f"[consequencegraph server] Done: {_graph.node_count()} nodes, {_graph.edge_count()} edges")

    _engine = QueryEngine(_graph)


# ── API ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="consequencegraph", description="Consequence-aware code knowledge graph")

app.add_middleware(CORSMiddleware, allow_origins=[CORS_ORIGIN], allow_methods=["GET", "POST"])  # H3


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if PRODUCTION and request.url.path.startswith("/api/"):
        ip = _get_real_ip(request)  # H2: real IP behind Render LB
        if not check_rate_limit(ip):
            return JSONResponse({"error": "Rate limit exceeded. Try again in a minute."}, status_code=429)
    return await call_next(request)


@app.get("/api/graph")
def api_graph():
    """Return full graph as D3-compatible node-link JSON."""
    g = get_graph()
    nodes = []
    for nid, data in g.g.nodes(data=True):
        nodes.append({
            "id": nid,
            "name": data.get("name", nid.split(".")[-1]),
            "type": data.get("node_type", "unknown"),
            "module": data.get("module", ""),
            "file": data.get("file_path", ""),
            "line": data.get("line_no", 0),
            "docstring": data.get("docstring", ""),
            "is_lightning_hook": data.get("is_lightning_hook", False),
            "tensor_shapes": data.get("tensor_shapes", {}),
            "config_keys": data.get("config_keys", []),
            "in_degree": g.in_degree(nid),
            "out_degree": g.out_degree(nid),
        })

    edges = []
    for u, v, data in g.g.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "type": data.get("edge_type", "unknown"),
            "severity_weight": data.get("severity_weight", 0),
            "reason": data.get("reason", ""),
        })

    return {"nodes": nodes, "edges": edges, "stats": g.stats()}


@app.get("/api/impact/{node_id:path}")
def api_impact(node_id: str, depth: int = Query(None, ge=1, le=6)):  # L2: cap depth
    """Return impact report for a node."""
    engine = get_engine()
    result = engine.impact(node_id, depth=depth)
    return result


@app.get("/api/impact_text/{node_id:path}")
def api_impact_text(node_id: str, depth: int = Query(None, ge=1, le=6)):  # L2
    """Return human-readable impact report."""
    engine = get_engine()
    result = engine.impact(node_id, depth=depth)
    return {"text": format_impact_as_context(result)}


@app.get("/api/search")
def api_search(q: str = Query(..., min_length=1, max_length=200), limit: int = Query(20, ge=1, le=100)):  # H4
    """Search nodes by name."""
    g = get_graph()
    results = []
    q_lower = q.lower()
    for nid, data in g.g.nodes(data=True):
        if q_lower in nid.lower():
            results.append({
                "id": nid,
                "name": data.get("name", ""),
                "type": data.get("node_type", ""),
                "file": data.get("file_path", ""),
            })
        if len(results) >= limit:
            break
    return {"results": results}


@app.get("/api/stats")
def api_stats():
    return get_graph().stats()


@app.post("/api/reindex")
def api_reindex():
    if PRODUCTION:
        raise HTTPException(status_code=403, detail="Reindex disabled in production demo.")
    build_or_load_graph(_index_path, force_reindex=True)
    return {"status": "ok", "nodes": get_graph().node_count(), "edges": get_graph().edge_count()}


@app.post("/api/consequence")
async def api_consequence(request: Request):
    """
    Unified consequence analysis. Accepts a free-form query + optional anchor node.
    Classifies intent, routes to the right analysis strategy, returns adaptive output:
      - specific_change   → step-by-step plan in dependency order
      - decision          → A vs B comparison with change counts
      - removal           → ordered cleanup plan
      - exploration       → structural paragraph + intersection breakdown
    """
    body = await request.json()
    query_text = body.get("query", "").strip()
    anchor_id  = body.get("anchor_node_id")       # set when user clicked a node first

    if not query_text:
        raise HTTPException(status_code=400, detail="query field required")
    if len(query_text) > 1000:  # H4: cap to prevent O(n×|q|) regex DoS
        raise HTTPException(status_code=400, detail="query too long (max 1000 chars)")
    # C2: validate anchor_id before trusting it
    if anchor_id is not None and (not isinstance(anchor_id, str) or len(anchor_id) > 500):
        anchor_id = None

    graph  = get_graph()
    engine = get_engine()

    # 1. Extract mentioned nodes — anchor always included and weighted higher
    mentioned = _cq_extract_nodes(query_text, graph)
    if anchor_id and graph.has_node(anchor_id) and anchor_id not in mentioned:  # C2
        mentioned.insert(0, anchor_id)
    if not mentioned:
        return {
            "error": "No known nodes found in query.",
            "hint": "Mention function or class names from the codebase, e.g. 'WeatherDataset.__getitem__' or 'training_step'.",
        }

    # 2. Classify intent
    intent     = _cq_classify_intent(query_text)
    change_type = _cq_classify_change_type(query_text)

    # 3. Compute blast radius per mentioned node
    impacts = {}
    for nid in mentioned:
        r = engine.impact(nid, depth=3)
        if "error" not in r and "ambiguous" not in r:
            impacts[nid] = r

    if not impacts:
        return {"error": "Could not compute impact for any mentioned nodes.", "mentioned": mentioned}

    # 4. Score all nodes by structural relevance
    intersection_scores: dict = {}
    edge_types_by_node:  dict = {}
    reasons_by_node:     dict = {}
    depth_by_node:       dict = {}

    for origin_id, impact in impacts.items():
        for direction in ("downstream", "upstream"):
            for entry in impact.get("blast_radius", {}).get(direction, []):
                nid = entry["node"]
                if any(nid.startswith(p) for p in ("hook::", "config::", "tensor_contract::")):
                    continue
                if nid in mentioned:
                    continue
                intersection_scores[nid] = intersection_scores.get(nid, 0) + 1
                edge_types_by_node.setdefault(nid, set()).add(entry.get("edge_type", ""))
                depth_by_node[nid] = min(depth_by_node.get(nid, 99), entry.get("depth", 99))
                reasons_by_node.setdefault(nid, []).append({
                    "origin":    origin_id,
                    "edge_type": entry.get("edge_type", ""),
                    "reason":    entry.get("reason", ""),
                    "depth":     entry.get("depth", 1),
                    "direction": direction,
                })

    # 5. Tier classification
    HIGH_RISK = {"produces_tensor_for", "overrides_hook", "inherits"}
    MED_RISK  = {"consumes_format", "reads_config", "calls"}

    # Bug 6: edge types that indicate real breakage vs. mere proximity
    PROXIMITY_ONLY = {"imports", "defined_in", "named_reference"}

    will_break  = []
    likely_need = []
    be_aware    = []

    for nid, score in sorted(intersection_scores.items(), key=lambda x: -x[1]):
        node_data  = graph.get_node(nid) or {}
        file_path  = node_data.get("file_path", "")
        edges      = edge_types_by_node.get(nid, set())
        depth      = depth_by_node.get(nid, 3)
        reasons    = reasons_by_node.get(nid, [])

        # Bug 6: never put external library nodes in will_break or likely_need
        if _is_external_node(nid, file_path):
            continue

        # Bug 6: proximity-only edges (imports, defined_in) don't indicate breakage
        if edges and edges.issubset(PROXIMITY_ONLY):
            continue

        # Bug 6: pure upstream inheritance from a base class doesn't mean it breaks
        all_directions = {r["direction"] for r in reasons}
        upstream_only = all_directions == {"upstream"}
        if upstream_only and edges.issubset({"inherits", "imports", "defined_in"}):
            continue

        # Compute tier score
        tier_score = score * 2
        if edges & HIGH_RISK:      tier_score += 3
        if edges & MED_RISK:       tier_score += 1
        if depth == 1:             tier_score += 2
        if node_data.get("is_lightning_hook"):    tier_score += 2
        if node_data.get("tensor_shapes"):        tier_score += 1
        if nid == anchor_id:                      tier_score += 3

        consequence = _cq_consequence_sentence(
            intent, change_type, nid, node_data, edges, reasons, depth
        )

        entry = {
            "node":       nid,
            "name":       node_data.get("name", nid.split(".")[-1]),
            "type":       node_data.get("node_type", ""),
            "file":       file_path,
            "line":       node_data.get("line_no", 0),
            "is_hook":    node_data.get("is_lightning_hook", False),
            "shapes":     node_data.get("tensor_shapes", {}),
            "edge_types": list(edges),
            "depth":      depth,
            "tier_score": tier_score,
            "intersection_count": score,
            "via":        list({r["origin"].split(".")[-1] for r in reasons})[:3],
            "consequence": consequence,
        }

        if tier_score >= 6:
            will_break.append(entry)
        elif tier_score >= 3:
            likely_need.append(entry)
        else:
            be_aware.append(entry)

    # Hard caps per tier
    will_break  = sorted(will_break,  key=lambda x: -x["tier_score"])[:5]
    likely_need = sorted(likely_need, key=lambda x: -x["tier_score"])[:4]
    be_aware    = sorted(be_aware,    key=lambda x: -x["tier_score"])[:3]

    # 6. Build adaptive lead
    lead = _cq_build_lead(
        intent, change_type, query_text, mentioned, impacts,
        will_break, likely_need, graph
    )

    # 7. Highlighted node IDs for graph
    highlighted = set(mentioned) | {n["node"] for n in will_break + likely_need + be_aware}

    return {
        "intent":        intent,
        "change_type":   change_type,
        "query":         query_text,
        "anchor":        anchor_id if (anchor_id and graph.has_node(anchor_id)) else None,  # C2: only echo valid
        "mentioned":     mentioned,
        "mentioned_details": [
            {
                "node": nid,
                "name": (graph.get_node(nid) or {}).get("name", nid.split(".")[-1]),
                "file": (graph.get_node(nid) or {}).get("file_path", ""),
                "line": (graph.get_node(nid) or {}).get("line_no", 0),
                "severity": impacts.get(nid, {}).get("severity", ""),
            }
            for nid in mentioned
        ],
        "lead":          lead,
        "will_break":    will_break,
        "likely_need":   likely_need,
        "be_aware":      be_aware,
        "highlighted":   list(highlighted),
    }


# ── Consequence helpers ───────────────────────────────────────────────────────

# Bug 4: NLP stopwords — common English narrative words and domain terms that
# appear in natural-language queries but should never be matched as code nodes.
_NLP_STOPWORDS = {
    # English narrative / intent words
    "downstream", "upstream", "change", "changes", "modify", "modification",
    "update", "updates", "approach", "design", "idea", "concept", "feature",
    "implement", "implementation", "add", "remove", "refactor", "integrate",
    "handle", "handling", "compute", "calculation", "dynamic", "static",
    "should", "could", "would", "will", "think", "since", "also", "either",
    "rather", "already", "during", "every", "inside", "across", "between",
    "instead", "feels", "natural", "extension", "strict", "highly",
    # ML domain narrative words that happen to match node names
    "forcing", "batch", "loss", "mask", "weight", "buffer", "graph",
    "model", "layer", "module", "step", "forward", "backward", "train",
    "valid", "test", "data", "output", "input", "result", "value",
    "tensor", "shape", "size", "index", "node", "edge", "mesh",
    # Python keywords that appear as node fragments
    "init", "self", "true", "false", "none", "type", "class", "base",
}

# Bug 6: External library prefixes — nodes from these are never in the "will break" tier
_EXTERNAL_NODE_PREFIXES = {
    "torch", "nn", "numpy", "np", "pytorch_lightning", "pl",
    "scipy", "sklearn", "pandas", "matplotlib", "wandb",
    "omegaconf", "hydra", "einops", "jaxtyping",
}


def _is_external_node(node_id: str, file_path: str) -> bool:
    """Return True if this node belongs to an external library, not target codebase."""
    if file_path == "<external>":
        return True
    prefix = node_id.split(".")[0]
    if prefix in _EXTERNAL_NODE_PREFIXES:
        return True
    return False


def _cq_extract_nodes(query: str, graph: KnowledgeGraph) -> list:
    found, seen = [], set()
    candidates = []
    for nid, data in graph.g.nodes(data=True):
        # Bug 6: skip external nodes entirely from NLP extraction
        if _is_external_node(nid, data.get("file_path", "")):
            continue
        name = data.get("name", "")
        if name and len(name) > 3:
            candidates.append((nid, name))
        parts = nid.split(".")
        if len(parts) >= 2:
            candidates.append((nid, parts[-2]))
            candidates.append((nid, parts[-1]))
    deduped = list({(n, nm): None for n, nm in candidates}.keys())
    deduped.sort(key=lambda x: -len(x[1]))
    for nid, name in deduped:
        if nid in seen:
            continue
        # Bug 4: skip stopwords — don't match narrative English as code nodes
        if name.lower() in _NLP_STOPWORDS:
            continue
        if re.search(r'\b' + re.escape(name) + r'\b', query, re.IGNORECASE):
            found.append(nid)
            seen.add(nid)
    return found[:10]


def _cq_classify_intent(query: str) -> str:
    q = query.lower()
    # Decision: two approaches being weighed
    if any(p in q for p in ["either", "or", "vs", "versus", "approach", "should i", "which"]):
        return "decision"
    # Removal
    if any(w in q for w in ["remove", "delete", "deprecate", "drop"]):
        return "removal"
    # Specific targeted change
    if any(w in q for w in ["add", "rename", "change", "modify", "replace", "refactor",
                              "implement", "introduce"]):
        return "specific_change"
    return "exploration"


def _cq_classify_change_type(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["return", "tuple", "output", "yield", "add tensor", "new tensor"]):
        return "return_format_change"
    if any(w in q for w in ["rename", "move"]):
        return "rename"
    if any(w in q for w in ["param", "argument", "signature", "kwarg"]):
        return "signature_change"
    if any(w in q for w in ["dim", "shape", "hidden", "channels", "d_h", "d_model"]):
        return "dimension_change"
    if any(w in q for w in ["buffer", "tensor", "weight", "mask", "register"]):
        return "add_tensor_component"
    if any(w in q for w in ["config", "hparam", "yaml", "setting"]):
        return "config_change"
    if any(w in q for w in ["remove", "delete", "deprecate"]):
        return "removal"
    return "modification"


def _cq_consequence_sentence(
    intent: str, change_type: str, node_id: str,
    node_data: dict, edges: set, reasons: list, depth: int
) -> str:
    name    = node_data.get("name", node_id.split(".")[-1])
    is_hook = node_data.get("is_lightning_hook", False)
    shapes  = node_data.get("tensor_shapes", {})
    n_origins = len({r["origin"] for r in reasons})
    file    = node_data.get("file_path", "")
    fname   = file.split("\\")[-1].split("/")[-1] if file else ""
    loc     = f" ({fname}:{node_data.get('line_no','')})" if fname else ""

    # Shape hint for the sentence
    shape_hint = ""
    if shapes:
        k, v = next(iter(shapes.items()))
        sv = v.get("shape", v) if isinstance(v, dict) else v
        shape_hint = f" — current contract: `{k}: {sv}`"

    if change_type == "return_format_change":
        if "consumes_format" in edges:
            return f"`{name}`{loc} unpacks this return value directly. Adding a tensor shifts the tuple — destructuring breaks here."
        if "produces_tensor_for" in edges:
            return f"`{name}`{loc} has an explicit tensor shape contract{shape_hint}. The contract must be updated to reflect the new output."
        if is_hook:
            return f"Lightning hook `{name}`{loc} receives this as its batch input — a format change causes a runtime mismatch at the framework boundary."

    if change_type == "add_tensor_component":
        if "__init__" in node_id:
            return f"`{name}`{loc} is where `register_buffer(...)` must be called — buffers registered here persist across devices, checkpoints, and `.to()` calls."
        if "consumes_format" in edges or "produces_tensor_for" in edges:
            return f"`{name}`{loc} sits in the tensor data flow{shape_hint}. The new weight tensor must be threaded through here explicitly."
        if is_hook:
            return f"Lightning hook `{name}`{loc} — this is where the buffer gets applied to the loss. It must be in scope here."

    if change_type == "signature_change":
        if "calls" in edges:
            return f"`{name}`{loc} calls this directly at depth {depth} — all argument lists at this call site need updating."
        if "inherits" in edges:
            return f"`{name}`{loc} inherits this method — the override signature must remain compatible or the MRO breaks."
        if "overrides_hook" in edges:
            return f"`{name}`{loc} implements a Lightning hook — the framework enforces the signature contract, changing it breaks training."

    if change_type == "rename":
        if "named_reference" in edges:
            return f"`{name}`{loc} references this name in a docstring or string literal — won't break at runtime but becomes stale documentation."
        if "calls" in edges:
            return f"`{name}`{loc} calls this by name — call sites break immediately on rename."
        if "imports" in edges:
            return f"`{name}`{loc} imports this symbol — the import path breaks on rename."

    if change_type == "removal":
        if "inherits" in edges:
            return f"`{name}`{loc} inherits from this class — removing it collapses the entire class hierarchy at this point."
        if "calls" in edges:
            return f"`{name}`{loc} calls this directly — removal raises `AttributeError` here at runtime."
        return f"`{name}`{loc} depends on this existing — removal causes `NameError` or `ImportError` at this file."

    if change_type == "dimension_change":
        if shapes:
            return f"`{name}`{loc} has shape contract{shape_hint} — the dimension change propagates into this contract."
        if "produces_tensor_for" in edges or "consumes_format" in edges:
            return f"`{name}`{loc} is in the tensor flow path — a dimension change cascades here via shape-dependent operations."

    if change_type == "config_change":
        if "reads_config" in edges:
            return f"`{name}`{loc} reads this config key — a rename or removal breaks the lookup silently (returns None, not an exception)."

    # Generic — but still specific about intersection
    if n_origins > 1:
        origins_str = ", ".join(list({r["origin"].split(".")[-1] for r in reasons})[:3])
        return f"`{name}`{loc} appears in the blast radius of {n_origins} affected nodes ({origins_str}) — structurally central to this change."
    edge_desc = next(iter(edges), "dependency")
    origin = reasons[0]["origin"].split(".")[-1] if reasons else "a mentioned node"
    return f"`{name}`{loc} is connected via `{edge_desc}` from `{origin}` at depth {depth}."


def _cq_build_lead(
    intent: str, change_type: str, query: str,
    mentioned: list, impacts: dict,
    will_break: list, likely_need: list,
    graph: KnowledgeGraph
) -> dict:
    """Build the adaptive lead section depending on query intent."""
    total_structural = len(will_break) + len(likely_need)

    if intent == "decision":
        # Bug 5: only compare function/class nodes — never modules, tensor contracts,
        # or metadata nodes. Comparing a module namespace to a function is invalid.
        EXECUTABLE_TYPES = {"function", "class"}
        options = []
        for nid in mentioned:
            node_data_m = graph.get_node(nid) or {}
            if node_data_m.get("node_type") not in EXECUTABLE_TYPES:
                continue
            if _is_external_node(nid, node_data_m.get("file_path", "")):
                continue
            impact = impacts.get(nid, {})
            br = impact.get("blast_radius", {})
            downstream = [
                e for e in br.get("downstream", [])
                if not e["node"].startswith(("hook::", "config::", "tensor_contract::"))
                and not _is_external_node(e["node"], e.get("file_path", ""))
            ]
            options.append({
                "node":  nid,
                "name":  node_data_m.get("name", nid.split(".")[-1]),
                "downstream_count": len(downstream),
                "severity": impact.get("severity", "low"),
            })
        options.sort(key=lambda x: x["downstream_count"])

        if len(options) >= 2:
            a, b = options[0], options[-1]
            diff = b["downstream_count"] - a["downstream_count"]
            recommendation = (
                f"`{a['name']}` is the structurally cheaper option — "
                f"{a['downstream_count']} downstream nodes affected vs "
                f"{b['downstream_count']} for `{b['name']}` "
                f"({diff} fewer cascading changes)."
            )
        else:
            recommendation = f"Analysis touches {len(mentioned)} candidate locations. See tier breakdown for structural cost."

        return {
            "format":         "decision",
            "recommendation": recommendation,
            "options":        options,
            "total_changes":  total_structural,
        }

    if intent == "removal":
        # Build a dependency-ordered cleanup plan
        # Sort will_break by depth — shallowest first (fix dependents before removing)
        ordered = sorted(will_break + likely_need, key=lambda x: x["depth"])
        steps = []
        for i, node in enumerate(ordered[:6], 1):
            steps.append({
                "step": i,
                "node": node["node"],
                "name": node["name"],
                "action": node["consequence"],
            })

        anchor_names = [(graph.get_node(n) or {}).get("name", n.split(".")[-1]) for n in mentioned[:2]]
        return {
            "format":  "removal_plan",
            "summary": (
                f"Removing `{'` / `'.join(anchor_names)}` requires cleaning up "
                f"{total_structural} dependent locations first. "
                f"Work from the outermost callers inward — removing the source before "
                f"clearing its consumers causes cascading runtime errors."
            ),
            "ordered_steps": steps,
        }

    if intent == "specific_change":
        # Step-by-step plan ordered by depth (shallowest = closest to source = change first)
        all_nodes = will_break + likely_need
        # Sort: direct dependents (depth 1) last — they break because of their dependency,
        # so fix the source first, then update consumers in order
        ordered = sorted(all_nodes, key=lambda x: x["depth"])
        steps = []
        anchor_name = (graph.get_node(mentioned[0]) or {}).get("name", mentioned[0].split(".")[-1]) if mentioned else "target"
        steps.append({
            "step": 1,
            "node": mentioned[0] if mentioned else "",
            "name": anchor_name,
            "action": f"Make the change here — this is the source node your query describes.",
        })
        for i, node in enumerate(ordered[:5], 2):
            steps.append({
                "step": i,
                "node": node["node"],
                "name": node["name"],
                "action": node["consequence"],
            })

        change_label = {
            "return_format_change": "return format change",
            "signature_change": "signature change",
            "rename": "rename",
            "dimension_change": "dimension change",
            "add_tensor_component": "new tensor component",
            "removal": "removal",
        }.get(change_type, "change")

        return {
            "format":  "step_by_step",
            "summary": (
                f"This {change_label} has {len(will_break)} nodes that will break "
                f"and {len(likely_need)} that likely need updating. "
                f"Follow the plan below in order — each step depends on the previous."
            ),
            "steps": steps,
        }

    # exploration / general — structural paragraph
    mentioned_names = [(graph.get_node(n) or {}).get("name", n.split(".")[-1]) for n in mentioned[:4]]
    top_node = will_break[0] if will_break else (likely_need[0] if likely_need else None)
    structural_insight = ""
    if top_node:
        structural_insight = (
            f" The most structurally central node not in your query is "
            f"`{top_node['name']}` — it sits in {top_node['intersection_count']} "
            f"blast radii and will need attention."
        )

    return {
        "format":  "exploration",
        "summary": (
            f"Your query touches {len(mentioned)} known nodes "
            f"({', '.join(f'`{n}`' for n in mentioned_names)})."
            f"{structural_insight} "
            f"{total_structural} additional nodes are structurally relevant to this idea."
        ),
    }


@app.get("/api/node_context/{node_id:path}")
def api_node_context(node_id: str):
    """Return pre-population context for a node — shown in textarea when user clicks."""
    graph  = get_graph()
    engine = get_engine()
    data   = graph.get_node(node_id)
    if not data:
        return {"error": "Node not found"}

    name     = data.get("name", node_id.split(".")[-1])
    ntype    = data.get("node_type", "")
    file     = data.get("file_path", "")
    fname    = file.split("\\")[-1].split("/")[-1] if file else ""
    line     = data.get("line_no", 0)
    sig      = data.get("signature", "")
    doc      = data.get("docstring", "")
    shapes   = data.get("tensor_shapes", {})
    is_hook  = data.get("is_lightning_hook", False)

    impact   = engine.impact(node_id, depth=2)
    sev      = impact.get("severity", "")
    downstream = impact.get("blast_radius", {}).get("downstream_count", 0)
    upstream   = impact.get("blast_radius", {}).get("upstream_count", 0)

    # Build context string
    lines = [f"Node: {name}  ({ntype} · {fname}:{line})"]
    if sig:
        lines.append(f"Signature: {name}{sig}")
    if is_hook:
        lines.append("Role: PyTorch Lightning lifecycle hook")
    if sev:
        lines.append(f"Impact: {sev.upper()} — {downstream} downstream, {upstream} upstream")
    if shapes:
        shape_parts = [f"{k}: {v.get('shape', v) if isinstance(v, dict) else v}" for k, v in list(shapes.items())[:3]]
        lines.append("Tensor contracts: " + ", ".join(shape_parts))
    if doc:
        lines.append(f'Docstring: "{doc[:120]}{"..." if len(doc) > 120 else ""}"')

    lines.append("")
    lines.append("Describe your idea or change:")

    return {
        "node_id":  node_id,
        "name":     name,
        "context_text": "\n".join(lines),
        "severity": sev,
        "downstream_count": downstream,
    }


@app.get("/api/diff")
def api_diff(base: str = Query("main", max_length=100)):  # C1: validate before git
    """Return impact for all functions changed vs git base branch."""
    if not re.match(r'^[a-zA-Z0-9/_.\.\-]{1,100}$', base) or base.startswith('-'):
        raise HTTPException(status_code=400, detail="Invalid base ref.")
    try:
        result = subprocess.run(
            ["git", "diff", base, "--name-only"],
            capture_output=True, text=True, cwd=_index_path
        )
        changed_files = [f.strip() for f in result.stdout.splitlines() if f.endswith(".py")]
    except Exception:
        raise HTTPException(status_code=500, detail="Git operation failed.")  # M2

    if not changed_files:
        return {"changed_files": [], "impacts": []}

    # Get changed functions from diff
    try:
        diff_result = subprocess.run(
            ["git", "diff", base, "--unified=0"] + changed_files,
            capture_output=True, text=True, cwd=_index_path
        )
        changed_funcs = _parse_diff_for_functions(diff_result.stdout)
    except Exception:
        changed_funcs = []

    engine = get_engine()
    impacts = []
    for func_name in changed_funcs[:20]:  # cap at 20
        r = engine.impact(func_name)
        if "error" not in r and "ambiguous" not in r:
            impacts.append({
                "function": func_name,
                "target": r.get("target"),
                "severity": r.get("severity"),
                "severity_score": r.get("severity_score"),
                "downstream_count": r.get("blast_radius", {}).get("downstream_count", 0),
                "critical_path": r.get("critical_path", []),
            })

    impacts.sort(key=lambda x: -x.get("severity_score", 0))
    return {"changed_files": changed_files, "impacts": impacts}


def _parse_diff_for_functions(diff_text: str) -> list[str]:
    """Extract function names from git diff output."""
    funcs = []
    for line in diff_text.splitlines():
        # git diff context lines like: @@ -10,5 +10,7 @@ def my_function(...):
        m = re.search(r'@@ .+ @@ (?:def|class) (\w+)', line)
        if m:
            funcs.append(m.group(1))
    return list(dict.fromkeys(funcs))  # deduplicate preserving order



# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def frontend():
    return HTMLResponse(content=_FRONTEND_HTML)


# ── HTML/JS frontend ──────────────────────────────────────────────────────────

_FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>consequencegraph — Consequence-Aware Code Knowledge Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #e6edf3; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  #topbar { display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: #161b22; border-bottom: 1px solid #30363d; flex-shrink: 0; }
  #topbar h1 { font-size: 14px; font-weight: 600; color: #58a6ff; letter-spacing: 0.05em; }
  #search-box { flex: 1; max-width: 320px; padding: 6px 10px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #e6edf3; font-size: 12px; outline: none; }
  #search-box:focus { border-color: #58a6ff; }
  #search-results { position: absolute; top: 44px; left: 180px; width: 320px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; z-index: 100; max-height: 300px; overflow-y: auto; display: none; }
  .search-result { padding: 8px 12px; cursor: pointer; font-size: 12px; border-bottom: 1px solid #21262d; }
  .search-result:hover { background: #1f2937; }
  .search-result .node-type { font-size: 10px; color: #8b949e; margin-left: 6px; }

  #btn-diff { padding: 6px 12px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; color: #e6edf3; font-size: 11px; cursor: pointer; }
  #btn-diff:hover { border-color: #58a6ff; color: #58a6ff; }
  #btn-reindex { padding: 6px 12px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; color: #e6edf3; font-size: 11px; cursor: pointer; }
  #stats-bar { font-size: 11px; color: #8b949e; margin-left: auto; }

  #main { display: flex; flex: 1; overflow: hidden; }
  #graph-container { flex: 1; position: relative; overflow: hidden; }
  svg { width: 100%; height: 100%; }

  #sidebar { width: 360px; background: #161b22; border-left: 1px solid #30363d; overflow-y: auto; flex-shrink: 0; display: flex; flex-direction: column; }
  #sidebar-header { padding: 14px 16px; border-bottom: 1px solid #30363d; }
  #sidebar-header h2 { font-size: 13px; color: #58a6ff; }
  #sidebar-content { padding: 14px 16px; flex: 1; font-size: 12px; line-height: 1.6; }
  #sidebar-content .placeholder { color: #8b949e; font-style: italic; }

  .impact-section { margin-top: 14px; }
  .impact-section h3 { font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  .impact-node { padding: 7px 10px; margin-bottom: 5px; border-radius: 5px; border-left: 3px solid transparent; cursor: pointer; }
  .impact-node:hover { background: #21262d; }
  .impact-node.sev-high { border-color: #f85149; }
  .impact-node.sev-medium { border-color: #e3b341; }
  .impact-node.sev-low { border-color: #3fb950; }
  .impact-node .node-name { font-weight: 600; font-size: 12px; }
  .impact-node .node-edge { font-size: 10px; color: #8b949e; margin-top: 2px; }
  .impact-node .node-reason { font-size: 10px; color: #6e7681; margin-top: 3px; line-height: 1.4; }

  .severity-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase; margin-left: 8px; }
  .sev-critical { background: #490202; color: #f85149; }
  .sev-high { background: #2d1f03; color: #e3b341; }
  .sev-medium { background: #0d2211; color: #3fb950; }
  .sev-low { background: #131d2e; color: #58a6ff; }

  .meta-row { display: flex; gap: 6px; margin-bottom: 4px; flex-wrap: wrap; }
  .meta-pill { padding: 2px 7px; background: #21262d; border-radius: 10px; font-size: 10px; color: #8b949e; }
  .meta-pill.hook { background: #1a1f2e; color: #79c0ff; }
  .meta-pill.tensor { background: #1f1a2e; color: #d2a8ff; }

  .shapes-block { background: #0d1117; border-radius: 4px; padding: 8px; font-size: 11px; margin-top: 6px; overflow-x: auto; }
  .shapes-block .shape-row { display: flex; gap: 8px; margin-bottom: 3px; }
  .shapes-block .shape-param { color: #79c0ff; }
  .shapes-block .shape-val { color: #d2a8ff; }
  .shapes-block .shape-conf { color: #6e7681; font-size: 10px; }

  #legend { position: absolute; bottom: 16px; left: 16px; background: rgba(22,27,34,0.92); border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; font-size: 11px; }
  #legend h4 { color: #8b949e; font-size: 10px; text-transform: uppercase; margin-bottom: 6px; }
  .legend-row { display: flex; align-items: center; gap: 7px; margin-bottom: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

  #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: #8b949e; font-size: 13px; }

  .node { cursor: pointer; }
  .node circle { stroke-width: 1.5px; transition: r 0.15s; }
  .node:hover circle { stroke-width: 2.5px; }
  .node.selected circle { stroke-width: 3px; }
  .node.dimmed { opacity: 0.12; }
  .node.highlighted circle { stroke-width: 3px; }

  .link { stroke-opacity: 0.35; fill: none; }
  .link.highlighted { stroke-opacity: 0.9; stroke-width: 2px; }
  .link.dimmed { stroke-opacity: 0.04; }

  .node-label { font-size: 9px; fill: #8b949e; pointer-events: none; }
  .node-label.visible { fill: #c9d1d9; }

  /* ── View Consequence panel ─────────────────────────────── */
  #consequence-panel { padding: 12px 16px; border-top: 1px solid #30363d; flex-shrink: 0; }
  #consequence-panel .panel-label { font-size: 10px; color: #8b949e; display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.06em; }
  #consequence-panel .panel-label span { color:#3fb950; font-size:9px; font-style:italic; text-transform:none; letter-spacing:0; }
  #consequence-query { width:100%; padding:7px 10px; background:#0d1117; border:1px solid #30363d; border-radius:6px; color:#e6edf3; font-size:11px; resize:none; outline:none; font-family:inherit; line-height:1.5; }
  #consequence-query:focus { border-color:#3fb950; }
  #btn-consequence { width:100%; margin-top:6px; padding:8px; background:#238636; border:none; border-radius:6px; color:white; font-size:11px; cursor:pointer; font-family:inherit; font-weight:600; letter-spacing:0.03em; }
  #btn-consequence:hover { background:#2ea043; }
  #btn-consequence:disabled { background:#1a3020; color:#3d6b45; cursor:not-allowed; }

  /* ── Consequence result tiers ───────────────────────────── */
  .tier-section { margin-top:14px; }
  .tier-header { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
  .tier-badge { font-size:9px; font-weight:700; padding:2px 7px; border-radius:10px; text-transform:uppercase; letter-spacing:0.08em; }
  .tier-will  { background:#3b0f0f; color:#f85149; }
  .tier-likely{ background:#2d1f03; color:#e3b341; }
  .tier-aware { background:#131d2e; color:#8b949e; }
  .tier-count { font-size:10px; color:#6e7681; }

  .cq-node { padding:9px 11px; margin-bottom:6px; border-radius:6px; background:#0d1117; border:1px solid #21262d; cursor:pointer; transition: border-color 0.15s; }
  .cq-node:hover { border-color:#30363d; }
  .cq-node.tier-1 { border-left:3px solid #f85149; }
  .cq-node.tier-2 { border-left:3px solid #e3b341; }
  .cq-node.tier-3 { border-left:3px solid #30363d; }
  .cq-node .cq-name { font-size:12px; font-weight:600; color:#e6edf3; display:flex; align-items:center; gap:6px; }
  .cq-node .cq-meta { font-size:10px; color:#6e7681; margin-top:2px; }
  .cq-node .cq-consequence { font-size:11px; color:#c9d1d9; margin-top:6px; line-height:1.5; padding-top:6px; border-top:1px solid #21262d; }
  .cq-edge-badge { font-size:9px; padding:1px 5px; background:#21262d; border-radius:3px; color:#8b949e; font-weight:400; }

  /* ── Lead card ──────────────────────────────────────────── */
  .lead-card { background:#0d1117; border:1px solid #30363d; border-radius:7px; padding:12px 14px; margin-bottom:14px; }
  .lead-card .lead-type { font-size:9px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; }
  .lead-card .lead-summary { font-size:12px; color:#c9d1d9; line-height:1.65; }
  .lead-card .lead-rec { font-size:12px; color:#3fb950; line-height:1.6; margin-top:6px; font-weight:600; }

  .step-list { margin-top:8px; }
  .step-row { display:flex; gap:10px; padding:7px 0; border-bottom:1px solid #21262d; cursor:pointer; }
  .step-row:last-child { border-bottom:none; }
  .step-num { font-size:11px; color:#58a6ff; font-weight:700; min-width:18px; }
  .step-name { font-size:11px; color:#e6edf3; font-weight:600; }
  .step-action { font-size:10px; color:#8b949e; margin-top:2px; line-height:1.4; }

  .option-row { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #21262d; }
  .option-row:last-child { border-bottom:none; }
  .option-name { font-size:11px; color:#e6edf3; }
  .option-cost { font-size:10px; color:#8b949e; }
  .option-cost.cheaper { color:#3fb950; }
</style>
</head>
<body>

<div id="topbar">
  <h1>⬡ consequencegraph</h1>
  <input id="search-box" type="text" placeholder="Search nodes... (function, class, module)" autocomplete="off">
  <div id="search-results"></div>
  <button id="btn-diff" onclick="loadDiff()">⎇ Diff vs main</button>
  <button id="btn-reindex" onclick="reindex()">↺ Reindex</button>
  <span id="stats-bar">loading...</span>
</div>

<div id="main">
  <div id="graph-container">
    <div id="loading">Loading graph...</div>
    <svg id="graph-svg"></svg>
    <div id="legend">
      <h4>Node types</h4>
      <div class="legend-row"><div class="legend-dot" style="background:#58a6ff"></div>function</div>
      <div class="legend-row"><div class="legend-dot" style="background:#d2a8ff"></div>class</div>
      <div class="legend-row"><div class="legend-dot" style="background:#f0883e"></div>lightning hook</div>
      <div class="legend-row"><div class="legend-dot" style="background:#f85149"></div>tensor contract</div>
      <div class="legend-row"><div class="legend-dot" style="background:#e3b341"></div>config key</div>
      <div class="legend-row"><div class="legend-dot" style="background:#3fb950"></div>module</div>
    </ul>
    </div>
  </div>

  <div id="sidebar">
    <div id="sidebar-header">
      <h2 id="sidebar-title">Click a node to explore</h2>
    </div>
    <div id="sidebar-content">
      <p class="placeholder">Select any node in the graph to see its impact analysis, dependencies, and downstream consequences.</p>
    </div>
    <div id="consequence-panel">
      <div class="panel-label">
        ⚡ View Consequence
        <span id="cq-hint">click a node to pre-fill context</span>
      </div>
      <textarea id="consequence-query" rows="3"
        placeholder="Describe a change, design question, or idea — mention function and class names directly.&#10;e.g. 'Add a spatial weight tensor to WeatherDataset.__getitem__ return tuple'"></textarea>
      <button id="btn-consequence" onclick="runConsequence()">Analyse consequences →</button>
    </div>
  </div>
</div>

<script>
const NODE_COLORS = {
  function: '#58a6ff',
  class: '#d2a8ff',
  module: '#3fb950',
  config_key: '#e3b341',
  tensor_contract: '#f85149',
  lightning_hook: '#f0883e',
  data_format: '#79c0ff',
  unknown: '#8b949e',
};

const EDGE_COLORS = {
  calls: '#58a6ff',
  inherits: '#d2a8ff',
  produces_tensor_for: '#f85149',
  overrides_hook: '#f0883e',
  consumes_format: '#e3b341',
  reads_config: '#3fb950',
  writes_config: '#56d364',
  named_reference: '#8b949e',
  imports: '#21262d',
  defined_in: '#21262d',
  instantiates: '#79c0ff',
};

// H1: HTML-escape — applied to all server-sourced data before innerHTML
function _h(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

let allNodes = [], allEdges = [], simulation, svg, g, nodeEl, linkEl, labelEl;
let selectedNodeId = null;
let currentZoom = 1;
let zoomBehavior = null;
let W = 0, H = 0;

async function loadGraph() {
  try {
    const resp = await fetch('/api/graph');
    if (!resp.ok) {
      const err = await resp.text();
      document.getElementById('loading').textContent = `Server error: ${resp.status} — ${err.slice(0,200)}`;
      return;
    }
    const data = await resp.json();
    allNodes = data.nodes.filter(n => !n.id.startsWith('hook::'));
    allEdges = data.edges.filter(e =>
      !e.type.includes('defined_in') && !e.type.includes('imports')
    );
    document.getElementById('stats-bar').textContent =
      `${data.stats.total_nodes} nodes · ${data.stats.total_edges} edges`;
    document.getElementById('loading').style.display = 'none';
    renderGraph();
  } catch(e) {
    document.getElementById('loading').textContent = `Failed to load graph: ${e.message}. Check server terminal for errors.`;
  }
}

function updateLabelVisibility() {
  if (!labelEl) return;
  // Show more labels as user zooms in — Google Maps style
  labelEl.style('display', d => {
    if (currentZoom > 2.5) return 'block';           // show everything
    if (currentZoom > 1.5) return d.in_degree > 2 ? 'block' : 'none';
    if (currentZoom > 0.8) return d.in_degree > 4 ? 'block' : 'none';
    return d.in_degree > 8 ? 'block' : 'none';       // only hubs at far out
  });
  // Scale label font inversely with zoom so they stay readable
  labelEl.style('font-size', `${Math.max(8, Math.min(13, 10 / currentZoom))}px`);
}

function flyToNode(nodeData) {
  if (!nodeData || !zoomBehavior) return;
  // Guard: node must have settled coordinates
  const x = nodeData.x, y = nodeData.y;
  if (x == null || y == null || isNaN(x) || isNaN(y)) return;

  const targetScale = Math.max(currentZoom, 2.0);
  const tx = W / 2 - x * targetScale;
  const ty = H / 2 - y * targetScale;

  svg.transition()
    .duration(750)
    .ease(d3.easeCubicInOut)
    .call(
      zoomBehavior.transform,
      d3.zoomIdentity.translate(tx, ty).scale(targetScale)
    );
}

function renderGraph() {
  const container = document.getElementById('graph-container');
  W = container.clientWidth; H = container.clientHeight;
  d3.select('#graph-svg').selectAll('*').remove();

  svg = d3.select('#graph-svg').attr('width', W).attr('height', H);

  zoomBehavior = d3.zoom().scaleExtent([0.05, 6])
    .on('zoom', (e) => {
      g.attr('transform', e.transform);
      currentZoom = e.transform.k;
      updateLabelVisibility();
      // Reheat simulation when zooming in so clumped nodes spread apart
      if (e.transform.k > 1.5 && simulation && simulation.alpha() < 0.05) {
        simulation
          .force('charge', d3.forceManyBody().strength(
            d => (-200 - d.in_degree * 20) * Math.min(e.transform.k, 3)
          ))
          .force('collision', d3.forceCollide(
            d => Math.max(12, 6 + d.in_degree * 1.5) * Math.min(e.transform.k, 2)
          ))
          .alpha(0.25)
          .restart();
      }
    });
  svg.call(zoomBehavior);

  g = svg.append('g');

  // Build lookup
  const nodeById = Object.fromEntries(allNodes.map(n => [n.id, n]));

  // Filter edges to only those with valid endpoints
  const validEdges = allEdges.filter(e => nodeById[e.source] && nodeById[e.target]);

  // Arrow markers
  const defs = svg.append('defs');
  Object.entries(EDGE_COLORS).forEach(([type, color]) => {
    defs.append('marker')
      .attr('id', `arrow-${type}`)
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', 18).attr('refY', 0)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L8,0L0,4')
      .attr('fill', color)
      .attr('opacity', 0.6);
  });

  // Links
  linkEl = g.append('g').selectAll('line')
    .data(validEdges).enter().append('line')
    .attr('class', 'link')
    .attr('stroke', d => EDGE_COLORS[d.type] || '#444')
    .attr('stroke-width', d => Math.max(0.5, d.severity_weight * 0.5))
    .attr('marker-end', d => `url(#arrow-${d.type})`);

  // Nodes
  let dragMoved = false;

  const nodeGroup = g.append('g').selectAll('.node')
    .data(allNodes).enter().append('g')
    .attr('class', 'node')
    .attr('id', d => `node-${d.id.replace(/[^a-zA-Z0-9]/g, '_')}`)
    .call(d3.drag()
      .on('start', (e, d) => {
        dragMoved = false;
        if (!e.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', (e, d) => {
        dragMoved = true;
        d.fx = e.x; d.fy = e.y;
      })
      .on('end', (e, d) => {
        if (!e.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
        // Only fire select if this was a click, not a drag
        if (!dragMoved) selectNode(d);
      })
    );

  nodeEl = nodeGroup;

  nodeGroup.append('circle')
    .attr('r', d => Math.max(4, Math.min(14, 3 + d.in_degree * 0.6)))
    .attr('fill', d => NODE_COLORS[d.type] || '#8b949e')
    .attr('stroke', d => d.is_lightning_hook ? '#f0883e' : (NODE_COLORS[d.type] || '#8b949e'))
    .attr('stroke-opacity', 0.8);

  labelEl = nodeGroup.append('text')
    .attr('class', 'node-label')
    .attr('dx', d => Math.max(4, Math.min(14, 3 + d.in_degree * 0.6)) + 3)
    .attr('dy', '0.35em')
    .text(d => d.name);

  // Initial label visibility based on zoom level
  updateLabelVisibility();

  // Simulation — stronger repulsion for dense graphs
  simulation = d3.forceSimulation(allNodes)
    .force('link', d3.forceLink(validEdges).id(d => d.id).distance(d => {
      // High-weight edges (tensor contracts, hooks) get more distance
      return 50 + (d.severity_weight || 0) * 15;
    }).strength(0.2))
    .force('charge', d3.forceManyBody().strength(d => {
      // Hubs repel harder — prevents the dense core clump
      return -200 - d.in_degree * 20;
    }))
    .force('center', d3.forceCenter(W / 2, H / 2).strength(0.05))
    .force('collision', d3.forceCollide(d => Math.max(12, 6 + d.in_degree * 1.5)))
    .on('tick', ticked);
}

function ticked() {
  linkEl
    .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  nodeEl.attr('transform', d => `translate(${d.x},${d.y})`);
}

async function selectNode(d) {
  selectedNodeId = d.id;
  document.getElementById('sidebar-title').textContent = d.name;

  // Wait one animation frame so simulation coordinates have settled
  requestAnimationFrame(() => flyToNode(d));

  // Pre-fill the consequence textarea with node context
  prefillConsequenceContext(d.id);

  const resp = await fetch(`/api/impact/${encodeURIComponent(d.id)}`);
  const impact = await resp.json();

  if (impact.error || impact.ambiguous) {
    document.getElementById('sidebar-content').innerHTML =
      `<p class="placeholder">${_h(impact.error || 'Ambiguous — multiple matches found.')}</p>`;
    return;
  }

  renderImpactSidebar(d, impact);
  highlightBlastRadius(impact);
}

function renderImpactSidebar(node, impact) {
  const br = impact.blast_radius || {};
  const meta = impact.target_meta || {};
  const sev = impact.severity || 'low';

  let html = `
    <div class="meta-row">
      <span class="meta-pill">${_h(meta.type || node.type)}</span>
      ${meta.is_lightning_hook ? '<span class="meta-pill hook">⚡ Lightning hook</span>' : ''}
      <span class="severity-badge sev-${_h(sev)}">${_h(sev)}</span>
    </div>`;

  if (meta.file) {
    const fname = meta.file.split(/[\/\\]/).pop();
    html += `<div style="color:#8b949e;font-size:10px;margin-top:4px">${_h(fname)}:${_h(meta.line || 0)}</div>`;
  }
  if (meta.signature) {
    html += `<div style="color:#79c0ff;font-size:11px;margin-top:6px;font-family:'SF Mono',monospace">${_h(node.name)}${_h(meta.signature)}</div>`;
  }
  if (meta.docstring) {
    html += `<div style="color:#6e7681;font-size:10px;margin-top:5px;line-height:1.5">${_h(meta.docstring)}</div>`;
  }

  // Tensor shapes
  if (meta.tensor_shapes && Object.keys(meta.tensor_shapes).length > 0) {
    html += `<div class="shapes-block"><div style="color:#8b949e;font-size:10px;margin-bottom:4px">Tensor shapes</div>`;
    for (const [param, info] of Object.entries(meta.tensor_shapes)) {
      const shapeVal = typeof info === 'object' ? info.shape : info;
      const conf = typeof info === 'object' ? Math.round((info.confidence || 0) * 100) : null;
      const src = typeof info === 'object' ? info.source : null;
      html += `<div class="shape-row">
        <span class="shape-param">${_h(param)}</span>
        <span class="shape-val">${_h(shapeVal)}</span>
        ${conf ? `<span class="shape-conf">${_h(conf)}% · ${_h(src)}</span>` : ''}
      </div>`;
    }
    html += `</div>`;
  }

  html += `<div style="color:#8b949e;font-size:10px;margin-top:10px">${_h(impact.llm_context_hint || '')}</div>`;

  // Critical path
  if (impact.critical_path && impact.critical_path.length > 1) {
    html += `<div class="impact-section"><h3>Critical path</h3>
      <div style="font-size:10px;color:#8b949e;word-break:break-all">
        ${impact.critical_path.map(n => `<span style="color:#79c0ff">${_h(n.split('.').pop())}</span>`).join(' → ')}
      </div></div>`;
  }

  // Upstream
  const upstream = (br.upstream || []).slice(0, 12);
  if (upstream.length > 0) {
    html += `<div class="impact-section"><h3>Upstream (${br.upstream_count}) — what this depends on</h3>`;
    upstream.forEach(e => { html += renderImpactNode(e); });
    html += `</div>`;
  }

  // Downstream
  const downstream = (br.downstream || []).slice(0, 15);
  if (downstream.length > 0) {
    html += `<div class="impact-section"><h3>Downstream (${br.downstream_count}) — what depends on this</h3>`;
    downstream.forEach(e => { html += renderImpactNode(e); });
    html += `</div>`;
  }

  document.getElementById('sidebar-content').innerHTML = html;
}

function renderImpactNode(e) {
  const sev = e.severity || 'low';
  const name = e.name || e.node.split('.').pop();
  return `<div class="impact-node sev-${_h(sev)}" onclick="selectNodeById('${_h(e.node)}')">
    <div class="node-name">${_h(name)}</div>
    <div class="node-edge">${_h(e.edge_type)} · depth ${_h(e.depth)}</div>
    ${e.reason ? `<div class="node-reason">${_h(e.reason.substring(0, 100))}</div>` : ''}
  </div>`;
}

// ── View Consequence ──────────────────────────────────────────────────────────

// When user clicks a node, pre-fill the textarea with structured context
async function prefillConsequenceContext(nodeId) {
  try {
    const resp = await fetch(`/api/node_context/${encodeURIComponent(nodeId)}`);
    const data = await resp.json();
    if (data.error) return;

    const ta = document.getElementById('consequence-query');
    ta.value = data.context_text;
    ta.dataset.anchorId = nodeId;

    const hint = document.getElementById('cq-hint');
    hint.textContent = `anchored to ${data.name} · ${data.severity || ''} severity`;
    hint.style.color = data.severity === 'critical' ? '#f85149'
                     : data.severity === 'high'     ? '#e3b341'
                     : '#3fb950';

    // Focus at end of textarea so user types after the context
    ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);
  } catch(e) {
    // Silently skip — textarea stays as-is
  }
}

async function runConsequence() {
  const ta   = document.getElementById('consequence-query');
  const query = ta.value.trim();
  if (!query) { alert('Describe a change, idea, or design question first.'); return; }

  const anchorId = ta.dataset.anchorId || null;
  const btn = document.getElementById('btn-consequence');
  btn.textContent = 'Analysing…';
  btn.disabled = true;

  try {
    const resp = await fetch('/api/consequence', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ query, anchor_node_id: anchorId }),
    });
    const data = await resp.json();
    if (data.error) { alert(data.error + (data.hint ? '\n\n' + data.hint : '')); return; }
    renderConsequenceSidebar(data);
    highlightConsequenceNodes(data);
  } catch(e) {
    alert('Request failed: ' + e.message);
  } finally {
    btn.textContent = 'Analyse consequences →';
    btn.disabled = false;
  }
}

function renderConsequenceSidebar(data) {
  const intentLabels = {
    specific_change: 'Specific change',
    decision:        'Design decision',
    removal:         'Removal',
    exploration:     'Exploration',
  };
  const changeLabels = {
    return_format_change:  '⟳ Return format',
    signature_change:      '⟨⟩ Signature',
    dimension_change:      '⊞ Dimension',
    rename:                '✎ Rename',
    removal:               '✕ Removal',
    add_tensor_component:  '⊕ New tensor',
    config_change:         '⚙ Config',
    modification:          '~ Modification',
  };

  let html = '';

  // ── Header badge ───────────────────────────────────────────
  const intentLabel  = intentLabels[data.intent]  || data.intent;
  const changeLabel  = changeLabels[data.change_type] || '';
  html += `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px">
    <span style="font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px;background:#131d2e;color:#79c0ff;text-transform:uppercase;letter-spacing:.08em">${_h(intentLabel)}</span>
    ${changeLabel ? `<span style="font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px;background:#1a1f2e;color:#8b949e;letter-spacing:.05em">${_h(changeLabel)}</span>` : ''}
  </div>`;

  // ── Lead card — adaptive per intent ────────────────────────
  const lead = data.lead || {};
  html += `<div class="lead-card">`;

  if (lead.format === 'decision') {
    html += `<div class="lead-type">Decision analysis</div>`;
    if (lead.recommendation) {
      html += `<div class="lead-rec">→ ${_h(lead.recommendation)}</div>`;
    }
    if (lead.options && lead.options.length) {
      const sorted = [...lead.options].sort((a,b) => a.downstream_count - b.downstream_count);
      html += `<div style="margin-top:10px">`;
      sorted.forEach((opt, i) => {
        const cheaper = i === 0;
        html += `<div class="option-row">
          <span class="option-name">${_h(opt.name)}</span>
          <span class="option-cost ${cheaper ? 'cheaper' : ''}">${_h(opt.downstream_count)} downstream${cheaper ? ' ✓ cheaper' : ''}</span>
        </div>`;
      });
      html += `</div>`;
    }

  } else if (lead.format === 'removal_plan') {
    html += `<div class="lead-type">Removal — cleanup order</div>
      <div class="lead-summary">${_h(lead.summary)}</div>`;
    if (lead.ordered_steps && lead.ordered_steps.length) {
      html += `<div class="step-list">`;
      lead.ordered_steps.forEach(s => {
        html += `<div class="step-row" onclick="selectNodeById('${_h(s.node)}')">
          <div class="step-num">${_h(s.step)}</div>
          <div>
            <div class="step-name">${_h(s.name)}</div>
            <div class="step-action">${_h(s.action)}</div>
          </div>
        </div>`;
      });
      html += `</div>`;
    }

  } else if (lead.format === 'step_by_step') {
    html += `<div class="lead-type">Implementation plan</div>
      <div class="lead-summary">${_h(lead.summary)}</div>`;
    if (lead.steps && lead.steps.length) {
      html += `<div class="step-list">`;
      lead.steps.forEach(s => {
        html += `<div class="step-row" ${s.node ? `onclick="selectNodeById('${_h(s.node)}')"` : ''}>
          <div class="step-num">${_h(s.step)}</div>
          <div>
            <div class="step-name">${_h(s.name)}</div>
            <div class="step-action">${_h(s.action)}</div>
          </div>
        </div>`;
      });
      html += `</div>`;
    }

  } else {
    // exploration
    html += `<div class="lead-type">Structural overview</div>
      <div class="lead-summary">${(lead.summary || '').replace(/`([^`]+)`/g, '<code style="background:#21262d;padding:1px 4px;border-radius:3px;color:#79c0ff">$1</code>')}</div>`;
  }
  html += `</div>`;

  // ── Nodes anchored in query ─────────────────────────────────
  if (data.mentioned_details && data.mentioned_details.length) {
    html += `<div style="font-size:10px;color:#6e7681;margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em">
      Nodes in your query (${data.mentioned_details.length})</div>`;
    html += `<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:14px">`;
    data.mentioned_details.forEach(n => {
      const sevColor = n.severity === 'critical' ? '#f85149' : n.severity === 'high' ? '#e3b341' : '#58a6ff';
      html += `<div onclick="selectNodeById('${_h(n.node)}')"
        style="font-size:10px;padding:3px 8px;background:#0d1117;border:1px solid #30363d;border-radius:4px;color:${sevColor};cursor:pointer;white-space:nowrap">
        ${_h(n.name)}</div>`;
    });
    html += `</div>`;
  }

  // ── Tier 1: will break ──────────────────────────────────────
  if (data.will_break && data.will_break.length) {
    html += `<div class="tier-section">
      <div class="tier-header">
        <span class="tier-badge tier-will">🔴 Will break</span>
        <span class="tier-count">${data.will_break.length} node${data.will_break.length > 1 ? 's' : ''} — must address</span>
      </div>`;
    data.will_break.forEach(n => { html += renderCqNode(n, 1); });
    html += `</div>`;
  }

  // ── Tier 2: likely need ─────────────────────────────────────
  if (data.likely_need && data.likely_need.length) {
    html += `<div class="tier-section">
      <div class="tier-header">
        <span class="tier-badge tier-likely">🟡 Likely need</span>
        <span class="tier-count">${data.likely_need.length} node${data.likely_need.length > 1 ? 's' : ''} — verify before shipping</span>
      </div>`;
    data.likely_need.forEach(n => { html += renderCqNode(n, 2); });
    html += `</div>`;
  }

  // ── Tier 3: be aware ────────────────────────────────────────
  if (data.be_aware && data.be_aware.length) {
    html += `<div class="tier-section">
      <div class="tier-header">
        <span class="tier-badge tier-aware">⚪ Be aware</span>
        <span class="tier-count">${data.be_aware.length} node${data.be_aware.length > 1 ? 's' : ''} — in blast radius, likely safe</span>
      </div>`;
    data.be_aware.forEach(n => { html += renderCqNode(n, 3); });
    html += `</div>`;
  }

  document.getElementById('sidebar-title').textContent = '⚡ Consequence analysis';
  document.getElementById('sidebar-content').innerHTML = html;
}

function renderCqNode(n, tier) {
  const fname  = (n.file || '').split(/[/\\]/).pop();
  const loc    = fname ? `${_h(fname)}${n.line ? ':' + _h(n.line) : ''}` : '';
  const edges  = (n.edge_types || []).slice(0, 2).map(_h).join(', ');
  const via    = (n.via || []).map(_h).join(', ');
  // Apply _h FIRST then substitute backtick→<code> so injected backtick content can't escape
  const consequence = _h(n.consequence || '').replace(
    /`([^`]+)`/g,
    '<code style="background:#21262d;padding:1px 4px;border-radius:3px;color:#79c0ff;font-size:10px">$1</code>'
  );
  const hookBadge = n.is_hook ? '<span class="cq-edge-badge" style="color:#f0883e">hook</span>' : '';
  const shapesBadge = n.shapes && Object.keys(n.shapes).length
    ? `<span class="cq-edge-badge">shapes</span>` : '';

  return `<div class="cq-node tier-${_h(tier)}" onclick="selectNodeById('${_h(n.node)}')">
    <div class="cq-name">
      ${_h(n.name)}
      ${hookBadge}${shapesBadge}
      ${edges ? `<span class="cq-edge-badge">${edges}</span>` : ''}
    </div>
    <div class="cq-meta">${loc}${via ? ` · via ${via}` : ''}${n.intersection_count > 1 ? ` · ${_h(n.intersection_count)} blast radii` : ''}</div>
    ${consequence ? `<div class="cq-consequence">${consequence}</div>` : ''}
  </div>`;
}

function highlightConsequenceNodes(data) {
  const highlighted = new Set(data.highlighted || []);
  const willBreakSet = new Set((data.will_break || []).map(n => n.node));
  const likelySet    = new Set((data.likely_need || []).map(n => n.node));

  if (!nodeEl || !linkEl) return;

  nodeEl.classed('dimmed',      d => !highlighted.has(d.id))
        .classed('highlighted', d => highlighted.has(d.id))
        .style('stroke', d => willBreakSet.has(d.id) ? '#f85149'
                            : likelySet.has(d.id)    ? '#e3b341'
                            : highlighted.has(d.id)  ? '#3fb950'
                            : null)
        .style('stroke-width', d => highlighted.has(d.id) ? '2px' : null);

  linkEl.classed('dimmed', d => {
    const s = typeof d.source === 'object' ? d.source.id : d.source;
    const t = typeof d.target === 'object' ? d.target.id : d.target;
    return !highlighted.has(s) && !highlighted.has(t);
  });

  labelEl.style('display', d => highlighted.has(d.id) ? 'block' : 'none');
}

function highlightBlastRadius(impact) {
  if (!nodeEl || !linkEl) return;
  const br = impact.blast_radius || {};
  const upIds = new Set((br.upstream || []).map(e => e.node));
  const downIds = new Set((br.downstream || []).map(e => e.node));
  const allHighlighted = new Set([impact.target, ...upIds, ...downIds]);

  nodeEl.classed('dimmed', d => !allHighlighted.has(d.id))
    .classed('highlighted', d => allHighlighted.has(d.id) && d.id !== impact.target)
    .classed('selected', d => d.id === impact.target);

  linkEl.classed('dimmed', d => {
    const srcId = typeof d.source === 'object' ? d.source.id : d.source;
    const tgtId = typeof d.target === 'object' ? d.target.id : d.target;
    return !allHighlighted.has(srcId) && !allHighlighted.has(tgtId);
  }).classed('highlighted', d => {
    const srcId = typeof d.source === 'object' ? d.source.id : d.source;
    const tgtId = typeof d.target === 'object' ? d.target.id : d.target;
    return allHighlighted.has(srcId) && allHighlighted.has(tgtId);
  });

  // Show labels on highlighted
  labelEl.style('display', d =>
    allHighlighted.has(d.id) ? 'block' : (d.in_degree > 3 ? 'block' : 'none')
  );
}

function selectNodeById(id) {
  const node = allNodes.find(n => n.id === id);
  if (node) selectNode(node);
}
const searchBox = document.getElementById('search-box');
const searchResults = document.getElementById('search-results');

searchBox.addEventListener('input', async () => {
  const q = searchBox.value.trim();
  if (q.length < 2) { searchResults.style.display = 'none'; return; }
  const resp = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=12`);
  const data = await resp.json();
  if (!data.results.length) { searchResults.style.display = 'none'; return; }
  searchResults.innerHTML = data.results.map(r => `
    <div class="search-result" onclick="selectNodeById('${_h(r.id)}'); searchResults.style.display='none'; searchBox.value='';">
      <span style="color:${NODE_COLORS[r.type]||'#8b949e'}">${_h(r.name)}</span>
      <span class="node-type">${_h(r.type)} · ${_h(r.id.split('.').slice(0,-1).join('.'))}</span>
    </div>`).join('');
  searchResults.style.display = 'block';
});

document.addEventListener('click', e => {
  if (!searchBox.contains(e.target)) searchResults.style.display = 'none';
});



// Diff
async function loadDiff() {
  const resp = await fetch('/api/diff?base=main');
  const data = await resp.json();
  if (!data.impacts.length) {
    alert('No changed Python functions found vs main branch.');
    return;
  }
  let html = `<div style="color:#8b949e;font-size:11px;margin-bottom:10px">
    Changed files: ${data.changed_files.length}</div>`;
  data.impacts.forEach(imp => {
    html += `<div class="impact-node sev-${_h(imp.severity || 'low')}" onclick="selectNodeById('${_h(imp.target)}')">
      <div class="node-name">${_h(imp.function)}</div>
      <div class="node-edge">${_h(imp.severity)} · ${_h(imp.downstream_count)} downstream</div>
    </div>`;
  });
  document.getElementById('sidebar-title').textContent = '⎇ Diff impact vs main';
  document.getElementById('sidebar-content').innerHTML = html;
}

// Reindex
async function reindex() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('loading').textContent = 'Re-indexing...';
  await fetch('/api/reindex', { method: 'POST' });
  location.reload();
}

// Init
loadGraph();
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="consequencegraph visual server")
    parser.add_argument("--path", default="./neural_lam", help="Path to index")
    parser.add_argument("--preset", default=None, choices=["neural_lam"])
    parser.add_argument("--port", type=int, default=7842)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    build_or_load_graph(args.path, preset=args.preset, force_reindex=args.reindex)
    print(f"[consequencegraph server] Open http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")  # M3: enable access logs


if __name__ == "__main__":
    main()
