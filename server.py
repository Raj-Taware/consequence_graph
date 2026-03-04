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
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────

PRODUCTION = os.environ.get("CONSEQUENCEGRAPH_ENV") == "production"
RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_MAX    = 60   # requests per window per IP


# ── Rate limiter ──────────────────────────────────────────────────────────────

_rate_buckets: dict = defaultdict(list)

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    bucket = _rate_buckets[ip]
    _rate_buckets[ip] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_buckets[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_buckets[ip].append(now)
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
    cache = os.path.join(os.getcwd(), ".consequencegraph", "cache.json")

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

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"])


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if PRODUCTION and request.url.path.startswith("/api/"):
        ip = request.client.host if request.client else "unknown"
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
def api_impact(node_id: str, depth: int = None):
    """Return impact report for a node."""
    engine = get_engine()
    result = engine.impact(node_id, depth=depth)
    return result


@app.get("/api/impact_text/{node_id:path}")
def api_impact_text(node_id: str, depth: int = None):
    """Return human-readable impact report."""
    engine = get_engine()
    result = engine.impact(node_id, depth=depth)
    return {"text": format_impact_as_context(result)}


@app.get("/api/search")
def api_search(q: str, limit: int = 20):
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


@app.get("/api/simulate")
def api_simulate(node_id: str, change: str, depth: int = None):
    """
    Given a node and a plain-English change description, classify the change
    type and return an annotated impact report explaining what specifically
    breaks at each downstream node.
    """
    engine = get_engine()
    graph = get_graph()

    # 1. Classify the change
    change_type, change_meta = _classify_change(change, node_id, graph)

    # 2. Get blast radius
    result = engine.impact(node_id, depth=depth)
    if "error" in result or "ambiguous" in result:
        return result

    # 3. Annotate each downstream node with what specifically breaks
    for entry in result.get("blast_radius", {}).get("downstream", []):
        node_data = graph.get_node(entry["node"]) or {}
        entry["consequence"] = _annotate_consequence(
            change_type, change_meta, entry, node_data
        )

    result["change_type"] = change_type
    result["change_description"] = change
    result["change_meta"] = change_meta
    return result


def _classify_change(description: str, node_id: str, graph: KnowledgeGraph) -> tuple[str, dict]:
    """
    Classify a plain-English change description into a structured change type.
    Returns (change_type, metadata_dict).
    """
    desc = description.lower().strip()

    # Signature / argument changes
    if any(w in desc for w in ["add param", "add argument", "new param", "new argument",
                                 "remove param", "remove argument", "rename param",
                                 "change signature", "add kwarg"]):
        return "signature_change", {"description": description}

    # Return value / output changes
    if any(w in desc for w in ["return", "output", "tuple", "add tensor", "new tensor",
                                 "remove tensor", "change return", "yield"]):
        # Try to extract shape hints from description
        import re
        shapes = re.findall(r'\[([^\]]+)\]|\(([^)]+)\)', description)
        return "return_format_change", {
            "description": description,
            "shapes_mentioned": [s[0] or s[1] for s in shapes],
        }

    # Rename
    if any(w in desc for w in ["rename", "move", "refactor name"]):
        return "rename", {"description": description}

    # Data format / schema changes
    if any(w in desc for w in ["format", "schema", "structure", "field", "key",
                                 "dict", "batch", "dataset", "datastore"]):
        return "data_format_change", {"description": description}

    # Dimension / shape changes
    if any(w in desc for w in ["dim", "shape", "size", "hidden", "channels",
                                 "feature", "embedding", "d_h", "d_model", "hidden_dim"]):
        import re
        dims = re.findall(r'\d+', description)
        return "dimension_change", {
            "description": description,
            "dimensions_mentioned": dims,
        }

    # Removal
    if any(w in desc for w in ["remove", "delete", "drop", "deprecate"]):
        return "removal", {"description": description}

    # Config change
    if any(w in desc for w in ["config", "hparam", "hyperparameter", "yaml", "setting"]):
        return "config_change", {"description": description}

    # Default — treat as generic modification
    return "modification", {"description": description}


def _annotate_consequence(
    change_type: str,
    change_meta: dict,
    blast_entry: dict,
    node_data: dict,
) -> str:
    """
    Given a change type and a downstream node, return a specific consequence string
    explaining what will break and why.
    """
    edge_type = blast_entry.get("edge_type", "")
    node_name = blast_entry.get("name") or blast_entry.get("node", "").split(".")[-1]
    is_hook = node_data.get("is_lightning_hook", False)
    shapes = node_data.get("tensor_shapes", {})

    if change_type == "return_format_change":
        if edge_type == "consumes_format":
            return f"`{node_name}` unpacks this return value — tuple destructuring will fail if arity or key names change."
        if edge_type == "produces_tensor_for":
            return f"Tensor contract at `{node_name}` will be violated — shape annotations need updating."
        if is_hook:
            return f"Lightning hook `{node_name}` receives the batch from this function — format mismatch will cause runtime error."

    elif change_type == "signature_change":
        if edge_type == "calls":
            return f"`{node_name}` calls this function directly — all call sites need updated argument lists."
        if edge_type == "overrides_hook":
            return f"Lightning requires a specific signature for this hook — changing it breaks the framework contract."
        if edge_type == "inherits":
            return f"Subclass `{node_name}` inherits this method — override signature must stay compatible."

    elif change_type == "dimension_change":
        if shapes:
            shape_str = ", ".join(f"{k}: {v.get('shape', v) if isinstance(v, dict) else v}" for k, v in list(shapes.items())[:2])
            return f"`{node_name}` has shape contract [{shape_str}] — dimension change will propagate here."
        if edge_type in ("produces_tensor_for", "consumes_format"):
            return f"`{node_name}` is in the tensor flow path — hidden dimension change cascades here."
        return f"`{node_name}` depends on this dimension — weight matrices or layer definitions may be mismatched."

    elif change_type == "rename":
        if edge_type == "named_reference":
            return f"`{node_name}` references this name in a docstring or comment — update needed."
        if edge_type == "calls":
            return f"`{node_name}` calls this by name — call sites will break."
        if edge_type == "imports":
            return f"`{node_name}` imports this symbol — import statement needs updating."

    elif change_type == "removal":
        if edge_type == "inherits":
            return f"`{node_name}` inherits from this — removal breaks the class hierarchy entirely."
        if edge_type == "calls":
            return f"`{node_name}` calls this function — will raise AttributeError or NameError at runtime."
        return f"`{node_name}` depends on this existing — removal causes NameError or ImportError."

    elif change_type == "config_change":
        if edge_type == "reads_config":
            return f"`{node_name}` reads this config key — key rename or removal breaks this lookup."

    elif change_type == "data_format_change":
        if edge_type == "consumes_format":
            return f"`{node_name}` consumes this data format — field rename or schema change breaks unpacking here."
        if is_hook:
            return f"Lightning hook `{node_name}` receives data from this pipeline — format change causes runtime failure."

            return f"Lightning hook `{node_name}` receives data from this pipeline — format change causes runtime failure."

    # Generic fallback
    return f"`{node_name}` is in the blast radius via `{edge_type}` — verify compatibility after this change."


@app.post("/api/reason")
async def api_reason(request: Request):
    """
    Multi-node intersection analysis for architectural/design queries.
    Extracts all mentioned nodes, computes combined blast radii,
    returns only structurally relevant intersection nodes ranked by relevance.
    """
    body = await request.json()
    query_text = body.get("query", "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="query field required")

    graph = get_graph()
    engine = get_engine()

    mentioned_nodes = _extract_mentioned_nodes(query_text, graph)
    if not mentioned_nodes:
        return {
            "error": "No known nodes found in query.",
            "hint": "Try mentioning function or class names from the codebase directly.",
            "query": query_text,
        }

    query_intent = _classify_query_intent(query_text)

    all_radii: dict = {}
    node_impacts: dict = {}
    for node_id in mentioned_nodes:
        result = engine.impact(node_id, depth=3)
        if "error" in result or "ambiguous" in result:
            continue
        node_impacts[node_id] = result
        downstream_ids = {e["node"] for e in result.get("blast_radius", {}).get("downstream", [])}
        upstream_ids = {e["node"] for e in result.get("blast_radius", {}).get("upstream", [])}
        all_radii[node_id] = downstream_ids | upstream_ids

    if not all_radii:
        return {"error": "Could not compute impact for any mentioned nodes.", "mentioned": mentioned_nodes}

    intersection_scores: dict = {}
    edge_types_by_node: dict = {}
    reasons_by_node: dict = {}

    for origin_id, radius in all_radii.items():
        impact = node_impacts.get(origin_id, {})
        all_entries = (
            impact.get("blast_radius", {}).get("downstream", []) +
            impact.get("blast_radius", {}).get("upstream", [])
        )
        for entry in all_entries:
            nid = entry["node"]
            if any(nid.startswith(p) for p in ("hook::", "config::", "tensor_contract::")):
                continue
            intersection_scores[nid] = intersection_scores.get(nid, 0) + 1
            edge_types_by_node.setdefault(nid, set()).add(entry.get("edge_type", ""))
            reasons_by_node.setdefault(nid, []).append({
                "origin": origin_id,
                "edge_type": entry.get("edge_type", ""),
                "reason": entry.get("reason", ""),
                "depth": entry.get("depth", 1),
            })

    relevant_edge_types = _relevant_edge_types_for_intent(query_intent)
    ranked = []
    for nid, score in sorted(intersection_scores.items(), key=lambda x: -x[1]):
        if nid in mentioned_nodes:
            continue
        node_data = graph.get_node(nid) or {}
        edges = edge_types_by_node.get(nid, set())
        relevance = score
        if edges & relevant_edge_types:
            relevance += 2
        if node_data.get("is_lightning_hook"):
            relevance += 1
        if node_data.get("tensor_shapes"):
            relevance += 1

        annotation = _reason_consequence(
            query_intent, query_text, nid, node_data, reasons_by_node.get(nid, [])
        )
        ranked.append({
            "node": nid,
            "name": node_data.get("name", nid.split(".")[-1]),
            "type": node_data.get("node_type", ""),
            "file": node_data.get("file_path", ""),
            "line": node_data.get("line_no", 0),
            "is_lightning_hook": node_data.get("is_lightning_hook", False),
            "tensor_shapes": node_data.get("tensor_shapes", {}),
            "intersection_count": score,
            "relevance": relevance,
            "edge_types": list(edges),
            "origins": list({r["origin"].split(".")[-1] for r in reasons_by_node.get(nid, [])}),
            "annotation": annotation,
        })

    ranked.sort(key=lambda x: -x["relevance"])
    must_change = [n for n in ranked if n["relevance"] >= 3][:8]
    likely_change = [n for n in ranked if 1 < n["relevance"] < 3][:5]

    return {
        "query": query_text,
        "query_intent": query_intent,
        "mentioned_nodes": mentioned_nodes,
        "mentioned_node_details": [
            {
                "node": nid,
                "name": (graph.get_node(nid) or {}).get("name", nid.split(".")[-1]),
                "file": (graph.get_node(nid) or {}).get("file_path", ""),
                "line": (graph.get_node(nid) or {}).get("line_no", 0),
            }
            for nid in mentioned_nodes
        ],
        "must_change": must_change,
        "likely_change": likely_change,
        "summary": _build_reason_summary(query_intent, mentioned_nodes, must_change, graph),
    }


def _extract_mentioned_nodes(query: str, graph: KnowledgeGraph) -> list:
    found = []
    seen = set()
    candidates = []
    for nid, data in graph.g.nodes(data=True):
        name = data.get("name", "")
        if name and len(name) > 3:
            candidates.append((nid, name))
        parts = nid.split(".")
        if len(parts) >= 2:
            candidates.append((nid, parts[-2]))
            candidates.append((nid, parts[-1]))
    seen_keys = set()
    unique = []
    for nid, name in candidates:
        k = (nid, name)
        if k not in seen_keys:
            seen_keys.add(k)
            unique.append((nid, name))
    unique.sort(key=lambda x: -len(x[1]))
    for nid, name in unique:
        if nid in seen:
            continue
        if re.search(r'\b' + re.escape(name) + r'\b', query, re.IGNORECASE):
            found.append(nid)
            seen.add(nid)
    return found[:10]


def _classify_query_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["add", "implement", "introduce", "create", "new"]):
        if any(w in q for w in ["buffer", "tensor", "weight", "mask"]):
            return "add_tensor_component"
        return "add_feature"
    if any(w in q for w in ["move", "refactor", "migrate", "integrate"]):
        return "refactor_location"
    if any(w in q for w in ["remove", "delete", "deprecate"]):
        return "removal"
    if any(w in q for w in ["where", "should", "better", "approach", "design", "architecture"]):
        return "design_question"
    return "general_query"


def _relevant_edge_types_for_intent(intent: str) -> set:
    base = {"calls", "inherits", "instantiates"}
    if intent in ("add_tensor_component", "add_feature"):
        return base | {"produces_tensor_for", "consumes_format", "reads_config"}
    if intent == "refactor_location":
        return base | {"consumes_format", "imports", "reads_config"}
    return base | {"produces_tensor_for", "consumes_format", "reads_config", "overrides_hook"}


def _reason_consequence(intent: str, query: str, node_id: str, node_data: dict, reasons: list) -> str:
    name = node_data.get("name", node_id.split(".")[-1])
    is_hook = node_data.get("is_lightning_hook", False)
    shapes = node_data.get("tensor_shapes", {})
    origins = [r["origin"].split(".")[-1] for r in reasons]
    edge_types = {r["edge_type"] for r in reasons}
    intersection_count = len({r["origin"] for r in reasons})

    if intent == "add_tensor_component":
        if "__init__" in node_id:
            return f"`{name}` — `register_buffer(...)` must be called here to persist the new tensor across devices and checkpoints."
        if "consumes_format" in edge_types or "produces_tensor_for" in edge_types:
            return f"`{name}` is in the tensor data flow — the new weight tensor must be threaded through here."
        if is_hook:
            return f"Lightning hook `{name}` — the buffer must be accessible at this scope for the weight to apply."
        if shapes:
            k, v = next(iter(shapes.items()))
            sv = v.get("shape", v) if isinstance(v, dict) else v
            return f"`{name}` has shape contract {k}: {sv} — new tensor must be dimensionally compatible."
        if intersection_count > 1:
            return f"`{name}` sits at the intersection of {intersection_count} affected paths ({', '.join(set(origins[:3]))}). Structural crossroads."

    if intent == "design_question":
        if intersection_count > 1:
            return f"`{name}` is structurally coupled to {intersection_count} of your candidate locations — affected regardless of design choice."
        if "consumes_format" in edge_types:
            return f"`{name}` consumes the data format that will change — must update under either approach."

    if intersection_count > 1:
        return f"`{name}` appears in {intersection_count} blast radii — central to this change."
    edge_desc = next(iter(edge_types), "dependency")
    return f"`{name}` connected via `{edge_desc}` from {origins[0] if origins else 'a mentioned node'}."


def _build_reason_summary(intent: str, mentioned_nodes: list, must_change: list, graph: KnowledgeGraph) -> str:
    n_mentioned = len(mentioned_nodes)
    n_must = len(must_change)
    names = [(graph.get_node(n) or {}).get("name", n.split(".")[-1]) for n in mentioned_nodes[:4]]
    if intent == "add_tensor_component":
        return (
            f"Query touches {n_mentioned} nodes ({', '.join(names)}). "
            f"Adding a tensor component requires changes at {n_must} structural points not explicitly mentioned. "
            f"Critical path: Datastore init → ARModel.__init__ (register_buffer) → training_step (apply weight). "
            f"Nodes marked 'must change' sit in the data flow between your proposed source and consumer."
        )
    return (
        f"Query references {n_mentioned} nodes. "
        f"{n_must} additional nodes are structurally relevant. "
        f"Ranked by how many mentioned nodes they intersect."
    )


@app.get("/api/diff")
def api_diff(base: str = "main"):
    """Return impact for all functions changed vs git base branch."""
    try:
        result = subprocess.run(
            ["git", "diff", base, "--name-only"],
            capture_output=True, text=True, cwd=_index_path
        )
        changed_files = [f.strip() for f in result.stdout.splitlines() if f.endswith(".py")]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Git error: {e}")

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

  #change-input-panel { padding: 12px 16px; border-top: 1px solid #30363d; }
  #change-input-panel label { font-size: 10px; color: #8b949e; display: block; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.06em; }
  #change-desc { width: 100%; padding: 7px 10px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #e6edf3; font-size: 11px; resize: none; outline: none; font-family: inherit; }
  #change-desc:focus { border-color: #58a6ff; }
  #btn-simulate { width: 100%; margin-top: 6px; padding: 7px; background: #1f6feb; border: none; border-radius: 6px; color: white; font-size: 11px; cursor: pointer; font-family: inherit; }
  #btn-simulate:hover { background: #388bfd; }
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
    <div id="change-input-panel">
      <div style="display:flex;gap:0;margin-bottom:8px">
        <button id="tab-simulate" onclick="switchTab('simulate')"
          style="flex:1;padding:5px;background:#1f6feb;border:none;border-radius:4px 0 0 4px;color:white;font-size:10px;cursor:pointer;font-family:inherit">
          Simulate change
        </button>
        <button id="tab-reason" onclick="switchTab('reason')"
          style="flex:1;padding:5px;background:#21262d;border:1px solid #30363d;border-left:none;border-radius:0 4px 4px 0;color:#8b949e;font-size:10px;cursor:pointer;font-family:inherit">
          Reason about design
        </button>
      </div>

      <div id="panel-simulate">
        <textarea id="change-desc" rows="2" placeholder="e.g. Add a new tensor to the return tuple" style="width:100%;padding:7px 10px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;font-size:11px;resize:none;outline:none;font-family:inherit"></textarea>
        <button id="btn-simulate" onclick="simulateChange()" style="width:100%;margin-top:6px;padding:7px;background:#1f6feb;border:none;border-radius:6px;color:white;font-size:11px;cursor:pointer;font-family:inherit">Predict consequences →</button>
      </div>

      <div id="panel-reason" style="display:none">
        <textarea id="reason-desc" rows="4" placeholder="Describe your architectural question or proposed change in plain English. Mention class/function names directly." style="width:100%;padding:7px 10px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;font-size:11px;resize:none;outline:none;font-family:inherit"></textarea>
        <button id="btn-reason" onclick="runReason()" style="width:100%;margin-top:6px;padding:7px;background:#238636;border:none;border-radius:6px;color:white;font-size:11px;cursor:pointer;font-family:inherit">Analyse design impact →</button>
      </div>
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

  const resp = await fetch(`/api/impact/${encodeURIComponent(d.id)}`);
  const impact = await resp.json();

  if (impact.error || impact.ambiguous) {
    document.getElementById('sidebar-content').innerHTML =
      `<p class="placeholder">${impact.error || 'Ambiguous — multiple matches found.'}</p>`;
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
      <span class="meta-pill">${meta.type || node.type}</span>
      ${meta.is_lightning_hook ? '<span class="meta-pill hook">⚡ Lightning hook</span>' : ''}
      <span class="severity-badge sev-${sev}">${sev}</span>
    </div>`;

  if (meta.file) {
    const fname = meta.file.split(/[\/\\]/).pop();
    html += `<div style="color:#8b949e;font-size:10px;margin-top:4px">${fname}:${meta.line || 0}</div>`;
  }
  if (meta.signature) {
    html += `<div style="color:#79c0ff;font-size:11px;margin-top:6px;font-family:'SF Mono',monospace">${node.name}${meta.signature}</div>`;
  }
  if (meta.docstring) {
    html += `<div style="color:#6e7681;font-size:10px;margin-top:5px;line-height:1.5">${meta.docstring}</div>`;
  }

  // Tensor shapes
  if (meta.tensor_shapes && Object.keys(meta.tensor_shapes).length > 0) {
    html += `<div class="shapes-block"><div style="color:#8b949e;font-size:10px;margin-bottom:4px">Tensor shapes</div>`;
    for (const [param, info] of Object.entries(meta.tensor_shapes)) {
      const shapeVal = typeof info === 'object' ? info.shape : info;
      const conf = typeof info === 'object' ? Math.round((info.confidence || 0) * 100) : null;
      const src = typeof info === 'object' ? info.source : null;
      html += `<div class="shape-row">
        <span class="shape-param">${param}</span>
        <span class="shape-val">${shapeVal}</span>
        ${conf ? `<span class="shape-conf">${conf}% · ${src}</span>` : ''}
      </div>`;
    }
    html += `</div>`;
  }

  html += `<div style="color:#8b949e;font-size:10px;margin-top:10px">${impact.llm_context_hint || ''}</div>`;

  // Critical path
  if (impact.critical_path && impact.critical_path.length > 1) {
    html += `<div class="impact-section"><h3>Critical path</h3>
      <div style="font-size:10px;color:#8b949e;word-break:break-all">
        ${impact.critical_path.map(n => `<span style="color:#79c0ff">${n.split('.').pop()}</span>`).join(' → ')}
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
  return `<div class="impact-node sev-${sev}" onclick="selectNodeById('${e.node}')">
    <div class="node-name">${name}</div>
    <div class="node-edge">${e.edge_type} · depth ${e.depth}</div>
    ${e.reason ? `<div class="node-reason">${e.reason.substring(0, 100)}</div>` : ''}
  </div>`;
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
    <div class="search-result" onclick="selectNodeById('${r.id}'); searchResults.style.display='none'; searchBox.value='';">
      <span style="color:${NODE_COLORS[r.type]||'#8b949e'}">${r.name}</span>
      <span class="node-type">${r.type} · ${r.id.split('.').slice(0,-1).join('.')}</span>
    </div>`).join('');
  searchResults.style.display = 'block';
});

document.addEventListener('click', e => {
  if (!searchBox.contains(e.target)) searchResults.style.display = 'none';
});

function switchTab(tab) {
  const isSimulate = tab === 'simulate';
  document.getElementById('panel-simulate').style.display = isSimulate ? 'block' : 'none';
  document.getElementById('panel-reason').style.display = isSimulate ? 'none' : 'block';
  document.getElementById('tab-simulate').style.background = isSimulate ? '#1f6feb' : '#21262d';
  document.getElementById('tab-simulate').style.color = isSimulate ? 'white' : '#8b949e';
  document.getElementById('tab-reason').style.background = isSimulate ? '#21262d' : '#238636';
  document.getElementById('tab-reason').style.color = isSimulate ? '#8b949e' : 'white';
}

async function runReason() {
  const query = document.getElementById('reason-desc').value.trim();
  if (!query) { alert('Describe your design question or proposed change.'); return; }

  const btn = document.getElementById('btn-reason');
  btn.textContent = 'Analysing...';
  btn.disabled = true;

  try {
    const resp = await fetch('/api/reason', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query})
    });
    const data = await resp.json();
    if (data.error) { alert(data.error + '\n' + (data.hint || '')); return; }
    renderReasonSidebar(data);
  } finally {
    btn.textContent = 'Analyse design impact →';
    btn.disabled = false;
  }
}

function renderReasonSidebar(data) {
  const intentLabels = {
    add_tensor_component: '⊞ Add tensor component',
    add_feature: '+ Add feature',
    refactor_location: '⟳ Refactor location',
    removal: '✕ Removal',
    design_question: '? Design question',
    general_query: '~ General query',
  };

  let html = `
    <div style="background:#1a1f2e;border-radius:6px;padding:10px 12px;margin-bottom:12px">
      <div style="font-size:10px;color:#8b949e;margin-bottom:3px">Intent classified as</div>
      <div style="color:#79c0ff;font-size:12px;font-weight:600">${intentLabels[data.query_intent] || data.query_intent}</div>
    </div>`;

  // Summary
  if (data.summary) {
    html += `<div style="color:#8b949e;font-size:11px;line-height:1.6;margin-bottom:12px;padding:8px 10px;background:#0d1117;border-radius:5px">${data.summary}</div>`;
  }

  // Mentioned nodes
  if (data.mentioned_node_details?.length) {
    html += `<div class="impact-section"><h3>Nodes found in query (${data.mentioned_node_details.length})</h3>`;
    data.mentioned_node_details.forEach(n => {
      const fname = (n.file || '').split(/[\/\\]/).pop();
      html += `<div class="impact-node sev-medium" onclick="selectNodeById('${n.node}')">
        <div class="node-name">${n.name}</div>
        <div class="node-edge">${fname}${n.line ? ':' + n.line : ''}</div>
      </div>`;
    });
    html += `</div>`;
  }

  // Must change
  if (data.must_change?.length) {
    html += `<div class="impact-section"><h3>🔴 Must address (${data.must_change.length})</h3>`;
    data.must_change.forEach(n => {
      html += renderReasonNode(n);
    });
    html += `</div>`;
  }

  // Likely change
  if (data.likely_change?.length) {
    html += `<div class="impact-section"><h3>🟡 Likely affected (${data.likely_change.length})</h3>`;
    data.likely_change.forEach(n => {
      html += renderReasonNode(n);
    });
    html += `</div>`;
  }

  // Highlight all relevant nodes in graph
  const allRelevant = new Set([
    ...(data.mentioned_nodes || []),
    ...(data.must_change || []).map(n => n.node),
    ...(data.likely_change || []).map(n => n.node),
  ]);
  if (nodeEl && linkEl) {
    nodeEl.classed('dimmed', d => !allRelevant.has(d.id))
      .classed('highlighted', d => allRelevant.has(d.id));
    linkEl.classed('dimmed', d => {
      const s = typeof d.source === 'object' ? d.source.id : d.source;
      const t = typeof d.target === 'object' ? d.target.id : d.target;
      return !allRelevant.has(s) && !allRelevant.has(t);
    });
    labelEl.style('display', d => allRelevant.has(d.id) ? 'block' : 'none');
  }

  document.getElementById('sidebar-title').textContent = 'Design impact analysis';
  document.getElementById('sidebar-content').innerHTML = html;
}

function renderReasonNode(n) {
  const fname = (n.file || '').split(/[\/\\]/).pop();
  const origins = (n.origins || []).slice(0, 3).join(', ');
  return `<div class="impact-node sev-high" onclick="selectNodeById('${n.node}')">
    <div class="node-name">${n.name}
      <span style="font-size:9px;color:#6e7681;font-weight:400;margin-left:6px">in ${n.intersection_count} blast radii</span>
    </div>
    <div class="node-edge">${fname}${n.line ? ':' + n.line : ''} · via ${origins}</div>
    ${n.annotation ? `<div class="node-reason" style="color:#c9d1d9;margin-top:3px">${n.annotation}</div>` : ''}
  </div>`;
}

// Simulate change
async function simulateChange() {
  const desc = document.getElementById('change-desc').value.trim();
  if (!selectedNodeId) {
    alert('Select a node first, then describe your change.');
    return;
  }
  if (!desc) {
    alert('Describe the change you want to make.');
    return;
  }

  const btn = document.getElementById('btn-simulate');
  btn.textContent = 'Analysing...';
  btn.disabled = true;

  try {
    const resp = await fetch(
      `/api/simulate?node_id=${encodeURIComponent(selectedNodeId)}&change=${encodeURIComponent(desc)}`
    );
    const impact = await resp.json();
    if (impact.error) {
      alert(impact.error);
      return;
    }

    const node = allNodes.find(n => n.id === selectedNodeId) || {};
    renderSimulateSidebar(node, impact);
    highlightBlastRadius(impact);
  } finally {
    btn.textContent = 'Predict consequences →';
    btn.disabled = false;
  }
}

function renderSimulateSidebar(node, impact) {
  const br = impact.blast_radius || {};
  const changeType = impact.change_type || 'modification';
  const sev = impact.severity || 'low';

  const changeTypeLabels = {
    return_format_change: '⟳ Return format change',
    signature_change: '⟨⟩ Signature change',
    dimension_change: '⊞ Dimension change',
    rename: '✎ Rename / move',
    removal: '✕ Removal',
    config_change: '⚙ Config change',
    data_format_change: '⊟ Data format change',
    modification: '~ General modification',
  };

  let html = `
    <div style="background:#1a1f2e;border-radius:6px;padding:10px 12px;margin-bottom:12px">
      <div style="font-size:10px;color:#8b949e;margin-bottom:4px">Change detected as</div>
      <div style="color:#79c0ff;font-size:12px;font-weight:600">${changeTypeLabels[changeType] || changeType}</div>
      <div style="color:#6e7681;font-size:10px;margin-top:4px;font-style:italic">"${impact.change_description}"</div>
    </div>
    <div class="meta-row">
      <span class="severity-badge sev-${sev}">${sev} impact</span>
      <span class="meta-pill">${br.downstream_count || 0} affected</span>
    </div>`;

  // Critical path
  if (impact.critical_path && impact.critical_path.length > 1) {
    html += `<div class="impact-section"><h3>Propagation path</h3>
      <div style="font-size:10px;color:#8b949e;word-break:break-all;line-height:1.8">
        ${impact.critical_path.map((n, i) => {
          const name = n.split('.').pop();
          const color = i === 0 ? '#f85149' : i === impact.critical_path.length - 1 ? '#e3b341' : '#79c0ff';
          return `<span style="color:${color}">${name}</span>`;
        }).join(' → ')}
      </div></div>`;
  }

  // Downstream with consequence annotations
  const downstream = (br.downstream || []).filter(e =>
    !e.node.startsWith('hook::') && !e.node.startsWith('config::')
  ).slice(0, 20);

  if (downstream.length > 0) {
    html += `<div class="impact-section"><h3>What breaks (${downstream.length} shown)</h3>`;
    downstream.forEach(e => {
      const sev = e.severity || 'low';
      const name = e.name || e.node.split('.').pop();
      const consequence = e.consequence || e.reason || '';
      html += `<div class="impact-node sev-${sev}" onclick="selectNodeById('${e.node}')">
        <div class="node-name">${name}</div>
        <div class="node-edge">${e.edge_type} · depth ${e.depth}</div>
        ${consequence ? `<div class="node-reason" style="color:#c9d1d9;margin-top:3px">${consequence}</div>` : ''}
      </div>`;
    });
    html += `</div>`;
  }

  document.getElementById('sidebar-title').textContent = `Simulate: ${node.name || selectedNodeId.split('.').pop()}`;
  document.getElementById('sidebar-content').innerHTML = html;
}

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
    html += `<div class="impact-node sev-${imp.severity || 'low'}" onclick="selectNodeById('${imp.target}')">
      <div class="node-name">${imp.function}</div>
      <div class="node-edge">${imp.severity} · ${imp.downstream_count} downstream</div>
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
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
