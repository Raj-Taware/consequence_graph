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
import ast
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
        elif preset == "wmg":
            from presets.wmg import apply
            apply(_graph)
        elif preset == "wmg_neural_lam":
            # Both repos indexed together — apply neural-lam domain knowledge
            # then wmg format contracts + cross-repo edges on top.
            from presets.neural_lam import apply as apply_nl
            from presets.wmg import apply as apply_wmg
            apply_nl(_graph)
            apply_wmg(_graph)
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
            # wmg preset fields
            "is_cross_repo": data.get("is_cross_repo", False),
            "mismatch_risk": data.get("mismatch_risk", ""),
            "is_lossiness_boundary": data.get("is_lossiness_boundary", False),
            "is_critical_preset": data.get("is_critical_preset", False),
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
        # Try to suggest real node names by extracting code-identifier-shaped tokens.
        # Only use words that look like identifiers: contain underscores, are camelCase,
        # or are long enough to be specific (≥8 chars). Skip plain English words.
        _ENGLISH_STOPWORDS = {
            "what", "which", "where", "when", "does", "will", "break", "change",
            "function", "functions", "class", "classes", "method", "methods",
            "read", "write", "return", "call", "calls", "use", "uses", "used",
            "with", "from", "into", "this", "that", "their", "there", "these",
            "those", "have", "should", "would", "could", "affect", "affects",
            "impact", "impacts", "downstream", "upstream", "depend", "depends",
            "feature", "features", "encoding", "positional", "spatial", "global",
            "local", "static", "input", "output", "tensor", "model", "data",
            "node", "nodes", "graph", "layer", "layers", "module", "modules",
            "train", "training", "forward", "backward", "loss", "batch",
        }

        def _looks_like_identifier(word: str) -> bool:
            """True if word looks like a code symbol rather than plain English."""
            if "_" in word:          return True   # snake_case: lat_lon_static
            if word[0].isupper() and any(c.islower() for c in word[1:]) \
               and any(c.isupper() for c in word[1:]):
                return True                         # CamelCase: WeatherDataset
            if len(word) >= 10:      return True    # long enough to be specific
            return False

        raw_words = [w.strip('.,?!()[]') for w in query_text.split()]
        candidate_words = [
            w for w in raw_words
            if len(w) >= 4
            and w.lower() not in _ENGLISH_STOPWORDS
            and _looks_like_identifier(w)
        ]

        suggestions = []
        seen_names: set = set()
        for word in candidate_words[:6]:
            word_lower = word.lower()
            for nid, data in graph.g.nodes(data=True):
                name = data.get("name", "")
                if name in seen_names:
                    continue
                if any(nid.startswith(p) for p in ("config::", "hook::", "tensor_contract::")):
                    continue
                # Require the match to be on a meaningful boundary
                name_lower = name.lower()
                nid_lower  = nid.lower()
                match = (
                    word_lower == name_lower                           # exact name match
                    or word_lower in name_lower.split("_")             # snake part match
                    or f"_{word_lower}" in nid_lower                   # suffix in ID
                    or nid_lower.endswith(f".{word_lower}")            # module.name
                    or (len(word_lower) >= 6 and word_lower in name_lower)  # substring, long only
                )
                if match:
                    suggestions.append({"id": nid, "name": name, "file": data.get("file_path","")})
                    seen_names.add(name)
                if len(suggestions) >= 6:
                    break
            if len(suggestions) >= 6:
                break

        no_identifiers = len(candidate_words) == 0
        hint = (
            "This query uses descriptive language but no code identifiers. "
            "Try searching for the function or class name directly — e.g. use the search box above to find it, then click the node to pre-fill the query."
            if no_identifiers else
            "Mention function or class names from the codebase directly — e.g. 'WeatherDataset.__getitem__' or 'training_step'."
        )
        return {
            "error": "No known nodes found in query.",
            "hint": hint,
            "suggestions": suggestions[:6],
        }

    # 2. Classify intent
    intent     = _cq_classify_intent(query_text)
    change_type = _cq_classify_change_type(query_text)

    # 3. Compute blast radius per mentioned node
    # engine.impact() excludes config::, hook::, tensor_contract:: nodes from resolution.
    # For those, fall back to direct graph traversal so comparison queries on config nodes work.
    _SYNTHETIC_PREFIXES = ("config::", "hook::", "tensor_contract::", "data_format::")
    impacts = {}
    for nid in mentioned:
        r = engine.impact(nid, depth=3)
        if "error" not in r and "ambiguous" not in r:
            impacts[nid] = r
        elif graph.has_node(nid) and any(nid.startswith(p) for p in _SYNTHETIC_PREFIXES):
            # Direct traversal fallback for synthetic nodes.
            # Config nodes are LEAF nodes — edges go function→config::key, never the reverse.
            # So "downstream" for a config key = all functions that read/write it = upstream().
            # We swap: use upstream for the "what does this affect" count on config nodes.
            node_data = graph.get_node(nid) or {}
            readers_raw  = graph.upstream(nid, depth=3)   # functions that use this config key
            writers_raw  = graph.downstream(nid, depth=3) # typically empty for config nodes
            all_raw = readers_raw + writers_raw
            score = len(all_raw)
            impacts[nid] = {
                "target": nid,
                "target_meta": {
                    "type": node_data.get("node_type"),
                    "file": node_data.get("file_path", ""),
                    "line": node_data.get("line_no", 0),
                    "is_lightning_hook": False,
                    "tensor_shapes": {},
                    "config_keys": [],
                },
                "severity": "medium" if score > 5 else "low",
                "severity_score": score,
                "blast_radius": {
                    "upstream":          readers_raw,
                    "downstream":        readers_raw,  # expose readers as "downstream" so comparison logic finds them
                    "upstream_count":    len(readers_raw),
                    "downstream_count":  len(readers_raw),
                },
                "critical_path": [],
                "llm_context_hint": "",
            }

    if not impacts:
        return {"error": "Could not compute impact for any mentioned nodes.", "mentioned": mentioned}

    # 3b. Post-process: config nodes are leaf nodes (edges go function→config::key,
    # never config→function). engine.impact() resolves them via exact match but returns
    # downstream_count=0 because they have no successors. For these nodes, the upstream
    # (functions that read/write the key) IS the blast radius — swap it in.
    _SYNTHETIC_PREFIXES = ("config::", "hook::", "tensor_contract::", "data_format::")
    for nid, impact in impacts.items():
        if any(nid.startswith(p) for p in _SYNTHETIC_PREFIXES):
            br = impact.get("blast_radius", {})
            if br.get("downstream_count", 0) == 0 and br.get("upstream_count", 0) > 0:
                readers = br.get("upstream", [])
                impact["blast_radius"]["downstream"]       = readers
                impact["blast_radius"]["downstream_count"] = len(readers)

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

    # 8. Tuple arity mismatch analysis — for return format / tensor addition changes
    arity_warnings = []
    if change_type in ("return_format_change", "add_tensor_component") and mentioned:
        primary = mentioned[0]
        new_arity = _infer_arity_delta(query_text, primary, graph)
        if new_arity is not None:
            mismatches = _check_arity_mismatches(primary, graph, new_arity)
            for m in mismatches:
                highlighted.add(m["node"])
            arity_warnings = mismatches

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
        "arity_warnings": arity_warnings,
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


# ── Tuple arity analysis ──────────────────────────────────────────────────────

def _parse_return_arity(file_path: str, func_name: str) -> int | None:
    """
    Re-parse the source file and find the dominant return tuple arity for func_name.
    Returns the most common tuple length across all return statements, or None if
    the function doesn't return a tuple or the file can't be parsed.
    """
    if not file_path or file_path.startswith("<") or not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return None

    arities = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    if isinstance(stmt.value, ast.Tuple):
                        arities.append(len(stmt.value.elts))
    if not arities:
        return None
    # Return the most common arity (handles functions with multiple return paths)
    from collections import Counter
    return Counter(arities).most_common(1)[0][0]


def _parse_unpack_arity(file_path: str, func_name: str) -> int | None:
    """
    Re-parse the source file and find tuple unpacking patterns in func_name that
    receive from a variable named 'batch', or the first argument after 'self'.
    Returns the unpack arity, or None if no unpacking found.

    Detects patterns like:
      a, b, c, d = batch
      (a, b, c) = batch
      init_states, target_states, forcing, static = batch
    """
    if not file_path or file_path.startswith("<") or not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return None

    # Known batch-carrying variable names in PyTorch Lightning
    _BATCH_VARS = {"batch", "batch_data", "data", "sample", "item"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            # Also collect parameter names (first non-self arg is often the batch)
            params = [a.arg for a in node.args.args if a.arg != "self"]
            batch_names = _BATCH_VARS | (set(params[:1]) if params else set())

            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    # LHS is a tuple unpack
                    if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Tuple):
                        # RHS is one of the batch variable names
                        rhs = stmt.value
                        rhs_name = None
                        if isinstance(rhs, ast.Name):
                            rhs_name = rhs.id
                        elif isinstance(rhs, ast.Subscript) and isinstance(rhs.value, ast.Name):
                            rhs_name = rhs.value.id
                        if rhs_name and rhs_name in batch_names:
                            return len(stmt.targets[0].elts)
    return None


def _check_arity_mismatches(
    source_node_id: str,
    graph: KnowledgeGraph,
    simulated_new_arity: int | None,
) -> list[dict]:
    """
    For a given source node (e.g. __getitem__), find all downstream consumes_format
    nodes, compute their unpack arity, and report mismatches against the simulated
    new return arity.

    Returns a list of mismatch dicts:
      { node, name, file, line, expected_arity, actual_arity, variables, message }
    """
    if simulated_new_arity is None:
        return []

    mismatches = []
    # Walk downstream edges for consumes_format
    for _, dst, edge_data in graph.g.out_edges(source_node_id, data=True):
        if edge_data.get("edge_type") != "consumes_format":
            continue
        dst_data = graph.get_node(dst) or {}
        dst_file  = dst_data.get("file_path", "")
        dst_name  = dst_data.get("name", dst.split(".")[-1])
        dst_line  = dst_data.get("line_no", 0)

        unpack_arity = _parse_unpack_arity(dst_file, dst_name)
        if unpack_arity is None:
            continue
        if unpack_arity == simulated_new_arity:
            continue  # no mismatch

        # Mismatch — build a precise message
        if simulated_new_arity > unpack_arity:
            msg = (
                f"`{dst_name}` unpacks {unpack_arity} values from batch "
                f"but the modified source now returns {simulated_new_arity} — "
                f"`ValueError: too many values to unpack` at runtime."
            )
        else:
            msg = (
                f"`{dst_name}` expects {unpack_arity} values from batch "
                f"but the modified source now returns {simulated_new_arity} — "
                f"`ValueError: not enough values to unpack` at runtime."
            )

        mismatches.append({
            "node":            dst,
            "name":            dst_name,
            "file":            dst_file,
            "line":            dst_line,
            "consumer_arity":  unpack_arity,
            "source_arity":    simulated_new_arity,
            "message":         msg,
        })

    return mismatches


def _infer_arity_delta(query: str, source_node_id: str, graph: KnowledgeGraph) -> int | None:
    """
    Infer what the new return arity would be after the described change.
    Returns the new arity if we can determine it, else None.

    Cases:
      - "add a tensor"  / "add ... to return" → current + 1
      - "remove tensor" / "remove ... from return" → current - 1
      - explicit number mentioned → use that
    """
    node_data = graph.get_node(source_node_id) or {}
    file_path = node_data.get("file_path", "")
    func_name = node_data.get("name", source_node_id.split(".")[-1])
    current_arity = _parse_return_arity(file_path, func_name)

    if current_arity is None:
        return None

    q = query.lower()

    # Explicit "N tensors" / "N elements" — rare but handle it
    explicit = re.search(r'\b(\d)\s*(?:tensor|item|element|value|field)', q)
    if explicit:
        return int(explicit.group(1))

    if any(w in q for w in ["add", "append", "include", "extra", "additional", "new tensor",
                              "new field", "introduce"]):
        return current_arity + 1

    if any(w in q for w in ["remove", "drop", "delete", "strip", "exclude"]):
        return max(1, current_arity - 1)

    return None



def _cq_extract_nodes(query: str, graph: KnowledgeGraph) -> list[str]:
    """
    Extract codebase node IDs mentioned in free-form query text.

    Bug 1 fix: polymorphic methods (same name, multiple class implementations)
    are grouped. Only the most structurally central representative is returned
    per name, annotated with a sibling count so the UI can note "and N others".
    This prevents get_xy appearing 3 times in the mentioned list.
    """
    found, seen = [], set()

    # Build candidates, skipping externals and stopwords
    candidates = []
    for nid, data in graph.g.nodes(data=True):
        if _is_external_node(nid, data.get("file_path", "")):
            continue
        name = data.get("name", "")
        if name and len(name) > 2 and name.lower() not in _NLP_STOPWORDS:
            candidates.append((nid, name, data))
        parts = nid.split(".")
        for part in parts[-2:]:
            if part and len(part) > 2 and part.lower() not in _NLP_STOPWORDS:
                candidates.append((nid, part, data))

    # Deduplicate (nid, name) pairs
    deduped_map: dict = {}
    for nid, name, data in candidates:
        key = (nid, name.lower())
        if key not in deduped_map:
            deduped_map[key] = (nid, name, data)
    deduped = list(deduped_map.values())
    # Longer names first — more specific matches win
    deduped.sort(key=lambda x: -len(x[1]))

    # Group matches by name to detect polymorphic implementations
    # name_lower → list of (nid, data)
    name_matches: dict[str, list] = {}
    for nid, name, data in deduped:
        nl = name.lower()
        if re.search(r'\b' + re.escape(name) + r'\b', query, re.IGNORECASE):
            name_matches.setdefault(nl, []).append((nid, data))

    # For each matched name, pick the most structurally central node as representative
    for nl, group in name_matches.items():
        # Sort by graph degree descending — most connected = most representative
        group.sort(key=lambda x: -(graph.in_degree(x[0]) + graph.out_degree(x[0])))
        best_nid, best_data = group[0]
        if best_nid in seen:
            continue
        found.append(best_nid)
        seen.add(best_nid)

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
    ntype   = node_data.get("node_type", "")
    n_origins = len({r["origin"] for r in reasons})
    file    = node_data.get("file_path", "")
    fname   = file.split("\\")[-1].split("/")[-1] if file else ""
    loc     = f" ({fname}:{node_data.get('line_no','')})" if fname else ""
    origin  = reasons[0]["origin"].split(".")[-1] if reasons else "a mentioned node"

    # Shape hint
    shape_hint = ""
    if shapes:
        k, v = next(iter(shapes.items()))
        sv = v.get("shape", v) if isinstance(v, dict) else v
        shape_hint = f" — current contract: `{k}: {sv}`"

    # ── Decision / comparison intent ─────────────────────────────────────────
    # For comparison queries, describe *why this node matters* for the decision.
    if intent == "decision":
        if is_hook:
            return f"`{name}`{loc} is a Lightning hook shared by both paths — whichever option you choose, this hook must be updated to match the new contract."
        if "tests" in node_id or name.startswith("test_"):
            origins = list({r["origin"].split(".")[-1] for r in reasons})[:2]
            return f"`{name}`{loc} exercises {'both' if n_origins > 1 else 'this'} path{'s' if n_origins > 1 else ''} — it will need updating regardless of which option you choose."
        if "consumes_format" in edges or "produces_tensor_for" in edges:
            return f"`{name}`{loc} sits in the tensor data flow from `{origin}` — the format contract here changes under either option."
        if n_origins > 1:
            origins_str = ", ".join(list({r["origin"].split(".")[-1] for r in reasons})[:2])
            return f"`{name}`{loc} is affected by both `{origins_str}` — this is shared coupling that both options carry."
        return f"`{name}`{loc} is downstream of `{origin}` at depth {depth} — affected by this path."

    # ── Specific change types ─────────────────────────────────────────────────
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

    # ── Generic fallback — role-specific, never generic ──────────────────────
    # Use node role + edge type to produce a specific sentence rather than the
    # "appears in blast radius" canned phrase.
    if is_hook:
        return f"Lightning hook `{name}`{loc} — framework contract at this boundary will be affected."
    if name.startswith("test_") or "tests" in node_id:
        return f"`{name}`{loc} tests the behaviour of `{origin}` — will need updating to match the new contract."
    if ntype == "class":
        return f"`{name}`{loc} is a class connected via `{'|'.join(list(edges)[:2]) or 'dependency'}` from `{origin}` — check constructor and method signatures."
    if "reads_config" in edges:
        return f"`{name}`{loc} reads config from `{origin}` — verify the key still exists and has the expected shape."
    if "calls" in edges:
        depth_word = "directly" if depth == 1 else f"at depth {depth}"
        return f"`{name}`{loc} calls into `{origin}` {depth_word} — update this call site."
    if "inherits" in edges:
        return f"`{name}`{loc} inherits from `{origin}` — verify compatibility with the new interface."
    if "consumes_format" in edges:
        return f"`{name}`{loc} consumes output from `{origin}` — the format contract here changes."
    if "produces_tensor_for" in edges:
        return f"`{name}`{loc} produces tensors consumed by `{origin}` — shape contracts may need updating."
    if n_origins > 1:
        origins_str = ", ".join(list({r["origin"].split(".")[-1] for r in reasons})[:2])
        return f"`{name}`{loc} is in the shared blast radius of `{origins_str}` — verify it handles the new interface."
    edge_desc = next(iter(edges), "dependency")
    return f"`{name}`{loc} is connected to `{origin}` via `{edge_desc}` at depth {depth}."


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

        # Fallback: if EXECUTABLE_TYPES filter left nothing (e.g. both nodes are
        # config or data_format type — still valid to compare peers of the same type),
        # build options from all mentioned nodes that have impact data.
        # NOTE: do NOT filter out config:: nodes from downstream here — when comparing
        # config nodes against each other, their config-typed connections ARE the signal.
        if not options:
            mentioned_prefixes = tuple(
                nid.split("::")[0] + "::" for nid in mentioned if "::" in nid
            )
            for nid in mentioned:
                if nid not in impacts:
                    continue
                node_data_m = graph.get_node(nid) or {}
                if _is_external_node(nid, node_data_m.get("file_path", "")):
                    continue
                br = impacts[nid].get("blast_radius", {})
                downstream = [
                    e for e in br.get("downstream", [])
                    if not e["node"].startswith(("hook::", "tensor_contract::"))
                    and not _is_external_node(e["node"], e.get("file_path", ""))
                ]
                options.append({
                    "node":  nid,
                    "name":  node_data_m.get("name", nid.split(".")[-1]),
                    "downstream_count": len(downstream),
                    "severity": impacts[nid].get("severity", "low"),
                })
            options.sort(key=lambda x: x["downstream_count"])

        # Bug 2 fix: deduplicate by name — polymorphic implementations of the same
        # method are NOT valid alternatives to compare (self-referential comparison).
        # Keep the one with the highest downstream count as the representative.
        seen_names: dict = {}
        for opt in options:
            nm = opt["name"]
            if nm not in seen_names or opt["downstream_count"] > seen_names[nm]["downstream_count"]:
                seen_names[nm] = opt
        options = list(seen_names.values())

        # Bug 3 fix: drop container nodes that are parents of other options.
        # Comparing ARModel vs ARModel.unroll_prediction is a category error —
        # the class contains the method; they're not peer alternatives.
        option_ids = {o["node"] for o in options}
        options = [
            o for o in options
            if not any(
                other != o["node"] and other.startswith(o["node"] + ".")
                for other in option_ids
            )
        ]
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




@app.get("/api/diff")
def api_diff(base: str = Query("main", max_length=100)):  # C1: validate before git
    """Return impact for all functions changed vs git base branch."""
    if not re.match(r'^[a-zA-Z0-9/_.\-]{1,100}$', base) or base.startswith('-'):
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
  #btn-help {
    padding: 6px 10px; background: none; border: 1px solid #30363d;
    border-radius: 6px; color: #6e7681; font-size: 11px; cursor: pointer;
    transition: color 0.15s, border-color 0.15s; font-family: inherit;
  }
  #btn-help:hover { color: #e6edf3; border-color: #8b949e; }

  /* ── Help / glossary modal ──────────────────────────────── */
  #help-overlay {
    position: fixed; inset: 0; z-index: 1001;
    background: rgba(1,4,9,0.88); backdrop-filter: blur(6px);
    display: flex; align-items: center; justify-content: center;
  }
  #help-overlay.hidden { display: none; }
  .help-box {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 32px 36px; max-width: 600px; width: 92%; max-height: 85vh;
    overflow-y: auto; box-shadow: 0 24px 80px rgba(0,0,0,0.6);
  }
  .help-box h2 { font-size: 16px; color: #e6edf3; margin-bottom: 4px; }
  .help-box .help-sub { font-size: 11px; color: #6e7681; margin-bottom: 24px; }
  .help-section { margin-bottom: 24px; }
  .help-section-title {
    font-size: 9px; color: #6e7681; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 10px;
    padding-bottom: 6px; border-bottom: 1px solid #21262d;
  }
  .help-term {
    display: grid; grid-template-columns: 140px 1fr;
    gap: 8px 16px; padding: 7px 0; border-bottom: 1px solid #161b22;
    align-items: start;
  }
  .help-term:last-child { border-bottom: none; }
  .help-term-name {
    font-size: 11px; font-weight: 600; color: #79c0ff;
    font-family: 'SF Mono', monospace;
  }
  .help-term-def { font-size: 11px; color: #c9d1d9; line-height: 1.6; }
  .help-term-def code {
    background: #21262d; padding: 1px 4px; border-radius: 3px;
    color: #79c0ff; font-size: 10px;
  }
  .help-kbd {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 4px 0;
  }
  .help-kbd kbd {
    background: #21262d; border: 1px solid #30363d; border-radius: 4px;
    padding: 2px 7px; font-size: 10px; color: #c9d1d9;
    font-family: 'SF Mono', monospace; min-width: 24px; text-align: center;
  }
  .help-kbd span { font-size: 11px; color: #8b949e; }
  .help-close {
    margin-top: 20px; padding: 8px 20px; background: #21262d;
    border: 1px solid #30363d; border-radius: 6px; color: #e6edf3;
    font-size: 12px; cursor: pointer; font-family: inherit; float: right;
  }
  .help-close:hover { background: #30363d; }

  /* ── Copy as Markdown button ────────────────────────────── */
  .copy-md-btn {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 10px; color: #6e7681; background: none;
    border: 1px solid #21262d; border-radius: 5px;
    padding: 4px 9px; cursor: pointer; font-family: inherit;
    transition: color 0.12s, border-color 0.12s; margin-bottom: 12px;
  }
  .copy-md-btn:hover { color: #c9d1d9; border-color: #30363d; }
  .copy-md-btn.copied { color: #3fb950; border-color: #3fb950; }

  /* ── Query history hint ─────────────────────────────────── */
  .history-hint {
    font-size: 9px; color: #6e7681; margin-bottom: 4px;
    display: none;
  }
  .history-hint.visible { display: block; }
  #btn-view-toggle {
    padding: 6px 12px; background: #0d2211; border: 1px solid #3fb950;
    border-radius: 6px; color: #3fb950; font-size: 11px; cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }
  #btn-view-toggle:hover { background: #132d1a; }
  #btn-view-toggle.full-mode {
    background: #21262d; border-color: #30363d; color: #8b949e;
  }

  /* Focus-mode visibility — nodes/edges outside the ego network fade out */
  .node.focus-hidden { opacity: 0.06; pointer-events: none; }
  .node.focus-visible { opacity: 1; }
  .link.focus-hidden { opacity: 0.04; }
  .link.focus-visible { opacity: 0.8; }

  /* Focus-mode context ring on ego node */
  .node.ego-node circle { filter: drop-shadow(0 0 5px rgba(63,185,80,0.5)); }
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
  .node circle { stroke-width: 1.5px; transition: r 0.15s, filter 0.15s; }
  .node:hover circle { stroke-width: 2.5px; filter: brightness(1.35) drop-shadow(0 0 5px currentColor); }
  .node.selected circle { stroke-width: 3px; filter: brightness(1.5) drop-shadow(0 0 8px currentColor); }
  .node.dimmed { opacity: 0.14; }
  .node.highlighted circle { stroke-width: 3px; filter: brightness(1.3); }

  .link { stroke-opacity: 0.55; fill: none; }
  .link.highlighted { stroke-opacity: 1.0; stroke-width: 2.5px; }
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

  /* ── Consequence result tiers — redesigned ─────────────── */
  .tier-section { margin-top: 16px; }
  .tier-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 7px; padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
  }
  .tier-label { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; }
  .tier-label.t1 { color: #f85149; }
  .tier-label.t2 { color: #e3b341; }
  .tier-label.t3 { color: #6e7681; }
  .tier-action { font-size:10px; color:#6e7681; margin-left:auto; }
  .tier-action.t1 { color:#f85149; opacity:0.7; }

  /* Cards */
  .cq-node {
    margin-bottom: 5px; border-radius: 7px;
    background: #0d1117; border: 1px solid #21262d;
    cursor: pointer; transition: border-color 0.12s; overflow: hidden;
  }
  .cq-node:hover { border-color: #30363d; }
  .cq-node.tier-1 { border-left: 3px solid #f85149; }
  .cq-node.tier-2 { border-left: 3px solid #e3b341; }
  .cq-node.tier-3 { border-left: 3px solid #30363d; opacity: 0.82; }

  /* Row 1 — name + dot + pill */
  .cq-header { display:flex; align-items:center; gap:8px; padding:9px 11px 0 11px; }
  .cq-sev-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; margin-top:1px; }
  .cq-name {
    font-size:12px; font-weight:600; color:#e6edf3;
    flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }
  .cq-tier-pill {
    font-size:8px; font-weight:700; padding:1px 6px; border-radius:8px;
    text-transform:uppercase; letter-spacing:0.08em; flex-shrink:0;
  }
  .cq-pill-1 { background:#3b0f0f; color:#f85149; }
  .cq-pill-2 { background:#2d1f03; color:#e3b341; }
  .cq-pill-3 { background:#1a1f2e; color:#6e7681; }

  /* Row 2 — consequence sentence, always visible, primary */
  .cq-consequence {
    font-size:11px; color:#c9d1d9; line-height:1.65;
    padding: 5px 11px 0 26px;
  }
  .cq-no-consequence { font-size:11px; color:#6e7681; font-style:italic; padding:4px 11px 0 26px; }

  /* Expand toggle */
  .cq-expand-btn {
    font-size:9px; color:#6e7681; background:none; border:none;
    cursor:pointer; padding:5px 11px 7px 26px; display:block;
    width:100%; text-align:left; font-family:inherit; transition:color 0.12s;
  }
  .cq-expand-btn:hover { color:#8b949e; }
  .cq-node.expanded .cq-expand-btn { color:#58a6ff; }

  /* Row 3 — metadata, collapsed by default */
  .cq-detail {
    display:none; padding:7px 11px 9px 26px;
    border-top:1px solid #161b22; margin-top:2px;
  }
  .cq-node.expanded .cq-detail { display:block; }
  .cq-meta-row { display:flex; flex-wrap:wrap; gap:5px; align-items:center; }
  .cq-meta-item { font-size:10px; color:#6e7681; }
  .cq-meta-item.file { color:#8b949e; font-family:'SF Mono',monospace; font-size:10px; }
  .cq-badge {
    font-size:9px; padding:1px 5px; background:#161b22; border:1px solid #21262d;
    border-radius:3px; color:#8b949e; position:relative; cursor:default;
  }
  .cq-badge.hook-badge { color:#f0883e; border-color:#2a1a0e; }
  .cq-badge.shape-badge { color:#d2a8ff; border-color:#1f1a2e; }
  .cq-badge .badge-tip {
    display:none; position:absolute; bottom:calc(100% + 6px); left:0;
    width:200px; padding:7px 10px; background:#1c2128; border:1px solid #30363d;
    border-radius:6px; font-size:10px; color:#c9d1d9; line-height:1.5;
    z-index:200; pointer-events:none; white-space:normal;
  }
  .cq-badge:hover .badge-tip { display:block; }

  /* legacy badge — kept for other code paths */
  .cq-edge-badge { font-size:9px; padding:1px 5px; background:#21262d; border-radius:3px; color:#8b949e; font-weight:400; position:relative; }
  .cq-edge-badge .edge-tooltip {
    display:none; position:absolute; bottom:calc(100% + 6px); left:50%;
    transform:translateX(-50%); width:180px; padding:7px 10px;
    background:#1c2128; border:1px solid #30363d; border-radius:6px;
    font-size:10px; color:#c9d1d9; line-height:1.5; z-index:200;
    pointer-events:none; white-space:normal;
  }
  .cq-edge-badge .edge-tooltip::after {
    content:''; position:absolute; top:100%; left:50%; transform:translateX(-50%);
    border:5px solid transparent; border-top-color:#30363d;
  }
  .cq-edge-badge:hover .edge-tooltip { display:block; }

  /* ── Lead card ──────────────────────────────────────────── */
  .lead-card { background:#0d1117; border:1px solid #30363d; border-radius:7px; padding:12px 14px; margin-bottom:14px; }
  .lead-card .lead-type { font-size:9px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; }
  .lead-card .lead-summary { font-size:12px; color:#c9d1d9; line-height:1.65; }
  .lead-card .lead-rec { font-size:12px; color:#3fb950; line-height:1.6; margin-top:6px; font-weight:600; }

  /* ── Comparison verdict (decision intent) ───────────────── */
  .verdict-card {
    background: #0d1117; border: 1px solid #30363d; border-radius:8px;
    padding: 16px; margin-bottom: 12px;
  }
  .verdict-winner {
    font-size: 13px; font-weight: 700; color: #3fb950;
    margin-bottom: 12px; line-height: 1.5;
  }
  .verdict-winner code {
    background: #0d2211; padding: 1px 6px; border-radius: 4px;
    color: #3fb950; font-size: 12px;
  }
  .verdict-bars { display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px; }
  .verdict-bar-row { display: flex; align-items: center; gap: 8px; }
  .verdict-bar-label {
    font-size: 11px; font-weight: 600; color: #e6edf3;
    width: 110px; flex-shrink: 0; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .verdict-bar-track {
    flex: 1; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden;
  }
  .verdict-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
  }
  .verdict-bar-fill.winner { background: #3fb950; }
  .verdict-bar-fill.loser  { background: #f85149; }
  .verdict-bar-count {
    font-size: 10px; color: #8b949e; width: 36px;
    text-align: right; flex-shrink: 0;
  }
  .verdict-bar-count.winner { color: #3fb950; font-weight: 700; }
  .verdict-bar-count.loser  { color: #f85149; }

  .verdict-delta {
    font-size: 10px; color: #6e7681; line-height: 1.6;
    padding-top: 10px; border-top: 1px solid #21262d;
  }
  .verdict-delta strong { color: #8b949e; }

  /* Breakdown toggle for decision intent */
  /* ── Inline error card ───────────────────────────────────── */
  .cq-error-card {
    background: #1a0a0a; border: 1px solid #6e1414; border-radius: 8px;
    padding: 14px 16px; margin-bottom: 12px;
  }
  .cq-error-title { font-size: 12px; color: #f85149; font-weight: 600; margin-bottom: 6px; }
  .cq-error-hint  { font-size: 11px; color: #8b949e; line-height: 1.6; margin-bottom: 10px; }
  .cq-error-suggestions { font-size: 10px; color: #6e7681; margin-bottom: 6px; }
  .cq-suggestion-btn {
    display: block; width: 100%; text-align: left;
    background: #21262d; border: 1px solid #30363d; border-radius: 5px;
    padding: 6px 10px; margin-bottom: 5px; cursor: pointer;
    font-size: 11px; color: #79c0ff; font-family: 'SF Mono', monospace;
    transition: background 0.12s; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
  }
  .cq-suggestion-btn:hover { background: #30363d; }
  .cq-suggestion-file { color: #6e7681; font-family: inherit; font-size: 10px; }

  /* Back to explorer button */
  .back-btn {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 10px; color: #6e7681; background: none; border: none;
    cursor: pointer; padding: 0 0 12px 0; font-family: inherit;
    transition: color 0.12s; letter-spacing: 0.02em;
  }
  .back-btn:hover { color: #8b949e; }

  .breakdown-toggle {
    font-size: 10px; color: #6e7681; background: none; border: none;
    cursor: pointer; padding: 6px 0 2px 0; display: block;
    font-family: inherit; transition: color 0.12s; text-align: left;
  }
  .breakdown-toggle:hover { color: #8b949e; }
  .breakdown-toggle.open { color: #58a6ff; }
  .breakdown-section { display: none; }
  .breakdown-section.open { display: block; }

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

  /* ── Arity warning panel ────────────────────────────────── */
  .arity-panel { margin-top:14px; }
  .arity-banner { display:flex; align-items:center; gap:8px; padding:9px 12px; background:#1a0a0a; border:1px solid #f8514966; border-radius:6px; margin-bottom:8px; }
  .arity-icon { font-size:14px; }
  .arity-label { font-size:11px; color:#f85149; font-weight:700; }
  .arity-sublabel { font-size:10px; color:#8b949e; margin-top:1px; }
  .arity-card { padding:9px 12px; background:#0d1117; border:1px solid #f8514933; border-left:3px solid #f85149; border-radius:6px; margin-bottom:6px; cursor:pointer; }
  .arity-card .arity-node { font-size:12px; font-weight:600; color:#f85149; }
  .arity-card .arity-loc  { font-size:10px; color:#6e7681; margin-top:1px; }
  .arity-card .arity-msg  { font-size:11px; color:#c9d1d9; margin-top:6px; line-height:1.5; }

  /* ── Welcome overlay ────────────────────────────────────── */
  /* ── 3-step onboarding tour ─────────────────────────────── */
  #welcome-overlay {
    position: fixed; inset: 0; z-index: 1000;
    background: rgba(1,4,9,0.92); backdrop-filter: blur(6px);
    display: flex; align-items: center; justify-content: center;
    animation: fadeIn 0.3s ease;
  }
  #welcome-overlay.hidden { display: none; }
  @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }

  .tour-box {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 36px 40px; max-width: 540px; width: 90%;
    box-shadow: 0 24px 80px rgba(0,0,0,0.6);
  }

  /* Step indicator */
  .tour-steps {
    display: flex; align-items: center; gap: 8px; margin-bottom: 28px;
  }
  .tour-step-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #21262d;
    transition: background 0.2s;
  }
  .tour-step-dot.active { background: #3fb950; }
  .tour-step-dot.done   { background: #238636; }
  .tour-step-label {
    font-size: 9px; color: #6e7681; text-transform: uppercase;
    letter-spacing: 0.1em; margin-left: 4px;
  }

  /* Step content */
  .tour-slide { display: none; }
  .tour-slide.active { display: block; }
  .tour-slide-icon { font-size: 32px; margin-bottom: 12px; }
  .tour-slide h2 { font-size: 17px; color: #e6edf3; margin-bottom: 10px; font-weight: 700; }
  .tour-slide p {
    font-size: 12px; color: #8b949e; line-height: 1.8; margin-bottom: 20px;
  }
  .tour-slide p strong { color: #c9d1d9; }
  .tour-slide p code {
    background: #21262d; padding: 1px 5px; border-radius: 3px;
    color: #79c0ff; font-size: 11px;
  }

  /* Visual demo area */
  .tour-demo {
    background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
    padding: 14px 16px; margin-bottom: 20px; font-size: 11px;
  }
  .tour-demo-row {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0; border-bottom: 1px solid #161b22; color: #c9d1d9;
  }
  .tour-demo-row:last-child { border-bottom: none; }
  .tour-demo-dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
  }
  .tour-demo-label { flex: 1; font-family: 'SF Mono', monospace; font-size: 10px; }
  .tour-demo-badge {
    font-size: 9px; padding: 2px 7px; border-radius: 10px; font-weight: 600;
  }
  .sev-high   { background: #3d1a1a; color: #f85149; }
  .sev-medium { background: #2d2208; color: #e3b341; }
  .sev-low    { background: #1a1f2e; color: #58a6ff; }

  /* Query demo */
  .tour-query-demo {
    background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 10px; font-size: 11px; color: #c9d1d9;
    font-style: italic;
  }
  .tour-query-result {
    background: #0d1a12; border: 1px solid #1f6329; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 20px; font-size: 11px; color: #3fb950;
  }

  /* Example queries on step 3 */
  .wc-examples { display: flex; flex-direction: column; gap: 7px; margin-bottom: 20px; }
  .wc-example {
    padding: 9px 13px; background: #0d1117; border: 1px solid #21262d;
    border-radius: 7px; cursor: pointer; transition: border-color 0.15s, background 0.15s;
    display: flex; align-items: flex-start; gap: 10px;
  }
  .wc-example:hover { border-color: #3fb950; background: #0d1a12; }
  .wc-ex-icon { font-size: 13px; flex-shrink: 0; margin-top: 1px; }
  .wc-ex-text { font-size: 11px; color: #c9d1d9; line-height: 1.5; }
  .wc-ex-text em { color: #6e7681; font-style: normal; font-size: 10px; display: block; margin-top: 2px; }

  /* Nav buttons */
  .tour-nav {
    display: flex; align-items: center; justify-content: space-between;
  }
  .tour-btn-back {
    padding: 8px 16px; background: none; border: 1px solid #30363d;
    border-radius: 7px; color: #6e7681; font-size: 12px; cursor: pointer;
    font-family: inherit; transition: color 0.12s;
  }
  .tour-btn-back:hover { color: #c9d1d9; }
  .tour-btn-next {
    padding: 9px 22px; background: #238636; border: none; border-radius: 7px;
    color: white; font-size: 12px; cursor: pointer; font-family: inherit;
    font-weight: 600; transition: background 0.15s;
  }
  .tour-btn-next:hover { background: #2ea043; }
  .tour-skip {
    font-size: 10px; color: #6e7681; background: none; border: none;
    cursor: pointer; font-family: inherit; padding: 0;
  }
  .tour-skip:hover { color: #8b949e; }

  /* ── Hot nodes (empty sidebar state) ───────────────────── */
  .hot-nodes-section { padding: 4px 0; }
  .hot-nodes-label {
    font-size: 9px; color: #6e7681; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 10px;
  }
  .hot-node-card {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 11px; margin-bottom: 6px; background: #0d1117;
    border: 1px solid #21262d; border-radius: 7px; cursor: pointer;
    transition: border-color 0.15s;
  }
  .hot-node-card:hover { border-color: #30363d; }
  .hot-node-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .hot-node-info { flex: 1; min-width: 0; }
  .hot-node-name { font-size: 12px; color: #e6edf3; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .hot-node-meta { font-size: 10px; color: #6e7681; margin-top: 1px; }
  .hot-node-sev { font-size: 9px; font-weight: 700; padding: 1px 6px; border-radius: 8px; flex-shrink: 0; text-transform: uppercase; }
  .hot-sev-critical { background: #490202; color: #f85149; }
  .hot-sev-high     { background: #2d1f03; color: #e3b341; }
  .hot-sev-medium   { background: #0d2211; color: #3fb950; }
  .hot-sev-low      { background: #131d2e; color: #58a6ff; }

  .sidebar-intro {
    font-size: 11px; color: #6e7681; line-height: 1.65;
    padding: 12px 14px; background: #0d1117; border-radius: 7px;
    border: 1px solid #21262d; margin-bottom: 16px;
  }
  .sidebar-intro strong { color: #8b949e; }

  /* ── Query example chips ────────────────────────────────── */
  .query-chips { display: flex; flex-direction: column; gap: 5px; margin-bottom: 8px; }
  .query-chip {
    padding: 5px 9px; background: #0d1117; border: 1px solid #21262d;
    border-radius: 5px; font-size: 10px; color: #8b949e; cursor: pointer;
    transition: border-color 0.12s, color 0.12s; line-height: 1.4;
    text-align: left;
  }
  .query-chip:hover { border-color: #3fb950; color: #c9d1d9; }
  .query-chip-label { font-size: 9px; color: #6e7681; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 5px; }

  /* ── Edge type tooltips ─────────────────────────────────── */
  .cq-edge-badge { position: relative; }
  .cq-edge-badge .edge-tooltip {
    display: none; position: absolute; bottom: calc(100% + 6px); left: 50%;
    transform: translateX(-50%); width: 180px; padding: 7px 10px;
    background: #1c2128; border: 1px solid #30363d; border-radius: 6px;
    font-size: 10px; color: #c9d1d9; line-height: 1.5; z-index: 200;
    pointer-events: none; white-space: normal;
  }
  .cq-edge-badge .edge-tooltip::after {
    content: ''; position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    border: 5px solid transparent; border-top-color: #30363d;
  }
  .cq-edge-badge:hover .edge-tooltip { display: block; }

  /* ── Improved legend ────────────────────────────────────── */
  #legend { max-width: 180px; }
  .legend-row .legend-desc { font-size: 9px; color: #6e7681; margin-left: 4px; }

  /* ── Onboarding hint pulses ─────────────────────────────── */
  @keyframes pulse-border {
    0%, 100% { border-color: #30363d; }
    50% { border-color: #3fb950; box-shadow: 0 0 0 2px rgba(63,185,80,0.15); }
  }
  .pulse-hint { animation: pulse-border 2s ease-in-out 3; }
</style>
</head>
<body>

<!-- ── Welcome overlay ────────────────────────────────────── -->
<div id="welcome-overlay">
  <div class="tour-box">

    <!-- Step indicator -->
    <div class="tour-steps">
      <div class="tour-step-dot active" id="tdot-0"></div>
      <div class="tour-step-dot" id="tdot-1"></div>
      <div class="tour-step-dot" id="tdot-2"></div>
      <span class="tour-step-label" id="tour-step-label">Step 1 of 3</span>
      <button class="tour-skip" onclick="dismissWelcome()" style="margin-left:auto">Skip tour</button>
    </div>

    <!-- Step 1: What are nodes? -->
    <div class="tour-slide active" id="tour-slide-0">
      <div class="tour-slide-icon">⬡</div>
      <h2>Every function is a node. Every coupling is an edge.</h2>
      <p>
        This graph maps <strong>two real repos</strong> — weather-model-graphs (wmg) and neural-lam —
        as a single dependency graph. Nodes are functions, classes, Lightning hooks, and config keys.
        Edges are the couplings between them: calls, tensor format contracts, disk-format dependencies.
      </p>
      <div class="tour-demo">
        <div class="tour-demo-row">
          <div class="tour-demo-dot" style="background:#f85149"></div>
          <div class="tour-demo-label">to_pyg</div>
          <div class="tour-demo-badge sev-high">high impact</div>
        </div>
        <div class="tour-demo-row">
          <div class="tour-demo-dot" style="background:#e3b341"></div>
          <div class="tour-demo-label">create_all_graph_components</div>
          <div class="tour-demo-badge sev-medium">medium</div>
        </div>
        <div class="tour-demo-row">
          <div class="tour-demo-dot" style="background:#58a6ff"></div>
          <div class="tour-demo-label">WeatherDataset.__getitem__</div>
          <div class="tour-demo-badge sev-low">low</div>
        </div>
      </div>
      <div class="tour-nav">
        <span></span>
        <button class="tour-btn-next" onclick="tourNext()">Next →</button>
      </div>
    </div>

    <!-- Step 2: Click a node -->
    <div class="tour-slide" id="tour-slide-1">
      <div class="tour-slide-icon">🖱</div>
      <h2>Click any node to see its impact scope.</h2>
      <p>
        Clicking a node shows everything that <strong>depends on it</strong> — ranked by how badly it
        would break. The graph zooms to show only the relevant neighbourhood.
        Cross-repo edges are shown in a distinct colour — these are the couplings your IDE can't see.
      </p>
      <div class="tour-demo">
        <div class="tour-demo-row">
          <div class="tour-demo-dot" style="background:#f85149"></div>
          <div class="tour-demo-label">training_step <span style="color:#6e7681">← neural-lam</span></div>
          <div class="tour-demo-badge sev-high">will break</div>
        </div>
        <div class="tour-demo-row">
          <div class="tour-demo-dot" style="background:#e3b341"></div>
          <div class="tour-demo-label">load_graph_from_disk <span style="color:#6e7681">← neural-lam</span></div>
          <div class="tour-demo-badge sev-medium">likely needs update</div>
        </div>
        <div class="tour-demo-row">
          <div class="tour-demo-dot" style="background:#3fb950"></div>
          <div class="tour-demo-label">test_create_graph_generic <span style="color:#6e7681">← wmg</span></div>
          <div class="tour-demo-badge" style="background:#1a2d1a;color:#3fb950">in scope</div>
        </div>
      </div>
      <div class="tour-nav">
        <button class="tour-btn-back" onclick="tourBack()">← Back</button>
        <button class="tour-btn-next" onclick="tourNext()">Next →</button>
      </div>
    </div>

    <!-- Step 3: Ask a question -->
    <div class="tour-slide" id="tour-slide-2">
      <div class="tour-slide-icon">⚡</div>
      <h2>Ask what breaks before you change anything.</h2>
      <p>
        Type a change description in the panel at the bottom — mention the <strong>function or class name</strong>
        directly. The tool classifies your intent, finds affected nodes, and ranks them by severity.
        Try one of these to start:
      </p>
      <div class="wc-examples">
        <div class="wc-example" onclick="dismissAndQuery('What breaks in neural-lam if I change the output format of to_pyg?')">
          <div class="wc-ex-icon">🔴</div>
          <div class="wc-ex-text">What breaks if I change <code>to_pyg()</code>'s output format?
            <em>Cross-repo impact — wmg serializer → neural-lam loader</em></div>
        </div>
        <div class="wc-example" onclick="dismissAndQuery('What Lightning hooks does WeatherDataset.__getitem__ affect if I add a tensor to its return tuple?')">
          <div class="wc-ex-icon">⚡</div>
          <div class="wc-ex-text">Add a tensor to <code>WeatherDataset.__getitem__</code>'s return tuple
            <em>Tensor format change → training cascade</em></div>
        </div>
        <div class="wc-example" onclick="dismissAndQuery('Should I modify create_all_graph_components or to_pyg — which has fewer downstream consequences?')">
          <div class="wc-ex-icon">⚖️</div>
          <div class="wc-ex-text">Compare modifying <code>create_all_graph_components</code> vs <code>to_pyg</code>
            <em>Design decision — structural cost comparison</em></div>
        </div>
      </div>
      <div class="tour-nav">
        <button class="tour-btn-back" onclick="tourBack()">← Back</button>
        <button class="tour-btn-next" onclick="dismissWelcome()">Explore the graph →</button>
      </div>
    </div>

  </div>
</div>

<!-- ── Help / glossary modal ──────────────────────────────── -->
<div id="help-overlay" class="hidden">
  <div class="help-box">
    <h2>consequencegraph — glossary & shortcuts</h2>
    <p class="help-sub">A cross-repo static analysis tool that maps ML framework contracts as graph nodes.</p>

    <div class="help-section">
      <div class="help-section-title">Core concepts</div>
      <div class="help-term">
        <div class="help-term-name">Impact scope</div>
        <div class="help-term-def">All nodes that would be affected if a given function or class changes. Includes direct callers, format consumers, and transitive dependents up to the configured depth.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">Tensor contract</div>
        <div class="help-term-def">An implicit agreement between two functions about the shape, dtype, or ordering of a tensor. Unlike a Python type annotation, contracts aren't enforced by the language — they break silently at runtime.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">Format contract</div>
        <div class="help-term-def">A cross-repo coupling that exists on disk rather than in Python call graphs — e.g. the filename <code>mesh_features.pt</code> that wmg writes and neural-lam reads. No IDE can see this link; consequencegraph makes it explicit.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">Lightning hook</div>
        <div class="help-term-def">A method like <code>training_step</code> or <code>validation_step</code> whose signature is enforced by the PyTorch Lightning framework at runtime. Changing these signatures breaks training even if Python doesn't complain.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">Cross-repo node</div>
        <div class="help-term-def">A dependency that spans two repositories — in this graph, between weather-model-graphs (wmg) and neural-lam. These are the hardest couplings to spot manually.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">Config key</div>
        <div class="help-term-def">A named value (like <code>g2m</code> or <code>m2m</code>) read or written by multiple functions. Config key nodes let you trace how a hyperparameter or graph name propagates through the codebase.</div>
      </div>
    </div>

    <div class="help-section">
      <div class="help-section-title">Result tiers</div>
      <div class="help-term">
        <div class="help-term-name">Will break</div>
        <div class="help-term-def">Nodes with a direct structural dependency on what you're changing — a call, format consumption, or hook contract. These will fail at runtime without intervention.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">Likely needs update</div>
        <div class="help-term-def">Nodes connected via a medium-risk edge (config read, indirect call). They probably need updating but may work depending on the nature of your change.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">In scope</div>
        <div class="help-term-def">Nodes in the broader dependency neighbourhood. Unlikely to break but worth reviewing — especially tests and documentation.</div>
      </div>
    </div>

    <div class="help-section">
      <div class="help-section-title">Edge types</div>
      <div class="help-term">
        <div class="help-term-name">produces_tensor_for</div>
        <div class="help-term-def">This function outputs a tensor that feeds directly into the target. Shape or format changes here cascade downstream.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">consumes_format</div>
        <div class="help-term-def">This node expects data in a specific format from the source. If the source format changes, this node breaks.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">overrides_hook</div>
        <div class="help-term-def">PyTorch Lightning hook override — the framework enforces the signature contract. Changing it breaks the training loop.</div>
      </div>
      <div class="help-term">
        <div class="help-term-name">reads_config</div>
        <div class="help-term-def">This node reads a config key. Renaming or removing the key causes a silent <code>None</code> return rather than an exception.</div>
      </div>
    </div>

    <div class="help-section">
      <div class="help-section-title">Keyboard shortcuts</div>
      <div class="help-kbd"><kbd>/</kbd> <span>Focus the search box</span></div>
      <div class="help-kbd"><kbd>Esc</kbd> <span>Back to explorer / close this panel</span></div>
      <div class="help-kbd"><kbd>Enter</kbd> <span>Run consequence analysis (when query box is focused)</span></div>
      <div class="help-kbd"><kbd>↑</kbd> <span>Previous query (when query box is focused)</span></div>
      <div class="help-kbd"><kbd>↓</kbd> <span>Next query (when query box is focused)</span></div>
      <div class="help-kbd"><kbd>?</kbd> <span>Open this help panel</span></div>
    </div>

    <button class="help-close" onclick="toggleHelp()">Close</button>
    <div style="clear:both"></div>
  </div>
</div>

<div id="topbar">
  <h1>consequencegraph</h1>
  <input id="search-box" type="text" placeholder="Search nodes... (function, class, module)" autocomplete="off">
  <div id="search-results"></div>
  <button id="btn-diff" onclick="loadDiff()">Diff vs main</button>
  <button id="btn-reindex" onclick="reindex()">Reindex</button>
  <button id="btn-view-toggle" onclick="toggleViewMode()" title="Switch between focused and full graph view">Focus view</button>
  <button id="btn-help" onclick="toggleHelp()" title="Glossary & keyboard shortcuts">? Help</button>
  <span id="stats-bar">loading...</span>
</div>

<div id="main">
  <div id="graph-container">
    <div id="loading">Loading graph...</div>
    <svg id="graph-svg"></svg>
    <div id="legend">
      <h4>Node types</h4>
      <div class="legend-row"><div class="legend-dot" style="background:#4dabf7"></div>function</div>
      <div class="legend-row"><div class="legend-dot" style="background:#cc5de8"></div>class</div>
      <div class="legend-row"><div class="legend-dot" style="background:#ff922b"></div>lightning hook</div>
      <div class="legend-row"><div class="legend-dot" style="background:#ff6b6b"></div>tensor contract</div>
      <div class="legend-row"><div class="legend-dot" style="background:#fcc419"></div>config key</div>
      <div class="legend-row"><div class="legend-dot" style="background:#51cf66"></div>module</div>
      <div class="legend-row"><div class="legend-dot" style="background:#74c0fc"></div>format contract</div>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid #21262d">
        <div style="font-size:9px;color:#6e7681;margin-bottom:4px;text-transform:uppercase;letter-spacing:.08em">Edge types</div>
        <div class="legend-row" style="margin-bottom:3px"><div style="width:18px;height:2px;background:#ff6b6b;flex-shrink:0"></div><span style="font-size:10px;color:#8b949e;margin-left:6px">produces tensor</span></div>
        <div class="legend-row" style="margin-bottom:3px"><div style="width:18px;height:2px;background:#fcc419;flex-shrink:0"></div><span style="font-size:10px;color:#8b949e;margin-left:6px">consumes format</span></div>
        <div class="legend-row" style="margin-bottom:3px"><div style="width:18px;height:2px;background:#4dabf7;flex-shrink:0"></div><span style="font-size:10px;color:#8b949e;margin-left:6px">calls</span></div>
        <div class="legend-row"><div style="width:18px;height:2px;background:#cc5de8;flex-shrink:0"></div><span style="font-size:10px;color:#8b949e;margin-left:6px">inherits</span></div>
      </div>
    </div>
  </div>

  <div id="sidebar">
    <div id="sidebar-header">
      <h2 id="sidebar-title">Start exploring</h2>
    </div>
    <div id="sidebar-content">
      <div class="sidebar-intro">
        <strong>Two ways to explore:</strong><br>
        Click any node in the graph to see what it affects — or describe a change in the panel below and get a ranked consequence breakdown.
      </div>
      <div class="hot-nodes-section">
        <div class="hot-nodes-label">Most structurally central nodes</div>
        <div id="hot-nodes-list"><div style="color:#6e7681;font-size:11px">Loading...</div></div>
      </div>
    </div>
    <div id="consequence-panel">
      <div class="panel-label">
        View Consequence
        <span id="cq-hint">click a node to pre-fill context</span>
      </div>
      <div class="query-chip-label">Example queries</div>
      <div class="query-chips">
        <button class="query-chip" onclick="fillQuery('What breaks in neural-lam if I change to_pyg output format?')">What breaks if I change <code>to_pyg()</code>?</button>
        <button class="query-chip" onclick="fillQuery('Add a spatial weight tensor to WeatherDataset.__getitem__ return tuple')">Add tensor to <code>__getitem__</code> return tuple</button>
        <button class="query-chip" onclick="fillQuery('Should I modify g2m or m2m — which has fewer downstream consequences?')">Compare <code>g2m</code> vs <code>m2m</code> impact</button>
      </div>
      <textarea id="consequence-query" rows="3"
        placeholder="Describe a change, design question, or idea — mention function and class names directly.&#10;e.g. 'Add a spatial weight tensor to WeatherDataset.__getitem__ return tuple'"></textarea>
      <div class="history-hint" id="history-hint">↑ ↓ to browse previous queries</div>
      <button id="btn-consequence" onclick="runConsequence()">Analyse consequences →</button>
    </div>
  </div>
</div>

<script>
const NODE_COLORS = {
  function:         '#4dabf7',   // bright sky blue
  class:            '#cc5de8',   // vivid purple
  module:           '#51cf66',   // bright green
  config_key:       '#fcc419',   // warm amber
  tensor_contract:  '#ff6b6b',   // bright coral red
  lightning_hook:   '#ff922b',   // vivid orange
  data_format:      '#74c0fc',   // light blue
  unknown:          '#868e96',   // neutral grey
};

// Cross-repo format contract nodes coloured by mismatch risk (wmg preset)
const MISMATCH_COLORS = {
  high:   '#f85149',
  medium: '#e3b341',
  low:    '#79c0ff',
};

const EDGE_COLORS = {
  calls:               '#4dabf7',   // sky blue
  inherits:            '#cc5de8',   // purple
  produces_tensor_for: '#ff6b6b',   // coral red
  overrides_hook:      '#ff922b',   // orange
  consumes_format:     '#fcc419',   // amber
  reads_config:        '#51cf66',   // green
  writes_config:       '#69db7c',   // light green
  named_reference:     '#868e96',   // grey
  imports:             '#343a40',   // dark — structural noise
  defined_in:          '#343a40',   // dark — structural noise
  instantiates:        '#74c0fc',   // light blue
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
let viewMode = 'full';           // 'focus' | 'full'
let visibleNodeIds = new Set();  // controls which nodes are shown in focus mode
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

  // Glow filter for high-degree hub nodes
  const glow = defs.append('filter').attr('id', 'node-glow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
  glow.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
  const merge = glow.append('feMerge');
  merge.append('feMergeNode').attr('in', 'blur');
  merge.append('feMergeNode').attr('in', 'SourceGraphic');

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
      .attr('opacity', 0.7);
  });

  // Links — higher opacity, min stroke-width so thin edges are visible
  linkEl = g.append('g').selectAll('line')
    .data(validEdges).enter().append('line')
    .attr('class', 'link')
    .attr('stroke', d => EDGE_COLORS[d.type] || '#555')
    .attr('stroke-width', d => Math.max(1.0, d.severity_weight * 0.6))
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
    .attr('r', d => Math.max(5, Math.min(18, 4 + d.in_degree * 0.9)))
    .attr('fill', d => {
      if (d.is_cross_repo && d.mismatch_risk && MISMATCH_COLORS[d.mismatch_risk]) {
        return MISMATCH_COLORS[d.mismatch_risk];
      }
      return NODE_COLORS[d.type] || '#8b949e';
    })
    .attr('fill-opacity', d => {
      // High-degree hubs are fully opaque; peripheral nodes slightly translucent
      return Math.min(1.0, 0.7 + (d.in_degree || 0) * 0.03);
    })
    .attr('stroke', d => {
      if (d.is_lossiness_boundary) return '#f85149';
      if (d.is_cross_repo) return '#ffffff';
      return d.is_lightning_hook ? '#f0883e' : (NODE_COLORS[d.type] || '#8b949e');
    })
    .attr('stroke-opacity', 0.9)
    .attr('stroke-width', d => d.is_cross_repo || d.is_lightning_hook ? 2 : 1.2)
    .attr('filter', d => (d.in_degree || 0) >= 3 ? 'url(#node-glow)' : null);

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

  // Focus on load is skipped — default is full graph view.
  // Focus activates when user clicks a node or toggles manually.
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

  // Expand ego network around clicked node (2-hop in focus mode)
  if (viewMode === 'focus') {
    visibleNodeIds = computeEgoNetwork(d.id, 2);
    applyFocusVisibility();
  }

  // Wait one animation frame so simulation coordinates have settled
  requestAnimationFrame(() => flyToNode(d));

  // Pre-fill the consequence textarea with node context
  prefillConsequenceContext(d.id);

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
      ${meta.is_lightning_hook ? '<span class="meta-pill hook">Lightning hook</span>' : ''}
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

function showConsequenceError(errorData, query) {
  const title = errorData.error || 'Something went wrong';
  const hint  = errorData.hint  || '';
  const suggestions = errorData.suggestions || [];

  let html = `<div class="cq-error-card">
    <div class="cq-error-title">${_h(title)}</div>`;
  if (hint) html += `<div class="cq-error-hint">${_h(hint)}</div>`;
  if (suggestions.length) {
    html += `<div class="cq-error-suggestions">Did you mean one of these?</div>`;
    suggestions.forEach(s => {
      const fname = (s.file || '').split(/[/\\]/).pop();
      html += `<button class="cq-suggestion-btn"
        onclick="fillQueryFromSuggestion('${_h(s.id)}','${_h(s.name)}')"
        title="${_h(s.id)}">
        ${_h(s.name)}
        ${fname ? `<span class="cq-suggestion-file"> — ${_h(fname)}</span>` : ''}
      </button>`;
    });
  }
  html += `</div>`;

  document.getElementById('sidebar-title').textContent = 'Consequence analysis';
  document.getElementById('sidebar-content').innerHTML =
    `<button class="back-btn" onclick="backToExplore()">← Back to explorer</button>` + html;
}

function fillQueryFromSuggestion(nodeId, name) {
  const ta = document.getElementById('consequence-query');
  // Insert the node name at cursor position or append
  const prev = ta.value.trim();
  ta.value = prev ? `${prev} — what breaks if I change ${name}?` : `What breaks if I change ${name}?`;
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
}

async function runConsequence() {
  const ta   = document.getElementById('consequence-query');
  const query = ta.value.trim();
  if (!query) {
    showConsequenceError({ error: 'No query entered.', hint: 'Describe a change, design question, or idea — mention function and class names directly.' });
    return;
  }

  pushHistory(query);
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
    if (data.error) {
      showConsequenceError(data, query);
      return;
    }
    renderConsequenceSidebar(data);
    highlightConsequenceNodes(data);
  } catch(e) {
    showConsequenceError({ error: 'Request failed.', hint: e.message });
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
    return_format_change:  'Return format',
    signature_change:      'Signature',
    dimension_change:      'Dimension',
    rename:                'Rename',
    removal:               'Removal',
    add_tensor_component:  'New tensor',
    config_change:         '⚙ Config',
    modification:          '~ Modification',
  };

  let html = '';

  // ── Header badge ───────────────────────────────────────────
  const intentLabel  = intentLabels[data.intent]  || data.intent;
  const changeLabel  = changeLabels[data.change_type] || '';
  html += `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px">
    <span style="font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px;background:#131d2e;color:#79c0ff;text-transform:uppercase;letter-spacing:.08em">${intentLabel}</span>
    ${changeLabel ? `<span style="font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px;background:#1a1f2e;color:#8b949e;letter-spacing:.05em">${changeLabel}</span>` : ''}
  </div>`;

  // ── Lead card — adaptive per intent ────────────────────────
  const lead = data.lead || {};

  if (lead.format === 'decision') {
    // ── Comparison verdict — the answer first, evidence second ──
    const sorted = [...(lead.options || [])].sort((a,b) => a.downstream_count - b.downstream_count);

    // Recommendation sentence — replace backtick-code spans
    const recHtml = (lead.recommendation || '').replace(
      /`([^`]+)`/g,
      '<code>$1</code>'
    );

    if (sorted.length === 0) {
      html += `<div class="verdict-card"><div class="verdict-winner">${recHtml}</div></div>`;
    } else {
      const maxCount = Math.max(...sorted.map(o => o.downstream_count), 1);
      const winner = sorted[0];
      const loser  = sorted[sorted.length - 1];
      const diff   = loser.downstream_count - winner.downstream_count;

      // Build bar rows
      const barRows = sorted.map((opt, i) => {
        const isWinner = i === 0;
        const pct = Math.round((opt.downstream_count / maxCount) * 100);
        return `<div class="verdict-bar-row">
          <div class="verdict-bar-label">${_h(opt.name)}</div>
          <div class="verdict-bar-track">
            <div class="verdict-bar-fill ${isWinner ? 'winner' : 'loser'}"
                 style="width:${pct || 4}%"></div>
          </div>
          <div class="verdict-bar-count ${isWinner ? 'winner' : 'loser'}">
            ${_h(opt.downstream_count)} node${opt.downstream_count !== 1 ? 's' : ''}
          </div>
        </div>`;
      }).join('');

      html += `<div class="verdict-card">
        <div class="verdict-winner">${recHtml}</div>
        <div class="verdict-bars">${barRows}</div>
        ${diff > 0 ? `<div class="verdict-delta">
          Modifying <strong>${_h(winner.name)}</strong> cascades into
          <strong>${diff} fewer node${diff !== 1 ? 's' : ''}</strong> than
          <strong>${_h(loser.name)}</strong>.
          Structural cost ratio: ${winner.downstream_count}:${loser.downstream_count}.
        </div>` : ''}
      </div>`;
    }

  } else {
    // All non-decision intents keep the lead-card format
    html += `<div class="lead-card">`;

    if (lead.format === 'removal_plan') {
      html += `<div class="lead-type">Removal — cleanup order</div>
        <div class="lead-summary">${_h(lead.summary || '')}</div>`;
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
        <div class="lead-summary">${_h(lead.summary || '')}</div>`;
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
      const summaryHtml = _h(lead.summary || '').replace(
        /`([^`]+)`/g,
        '<code style="background:#21262d;padding:1px 4px;border-radius:3px;color:#79c0ff">$1</code>'
      );
      html += `<div class="lead-type">Structural overview</div>
        <div class="lead-summary">${summaryHtml}</div>`;
    }

    html += `</div>`; // close .lead-card
  }

  // ── Nodes anchored in query ─────────────────────────────────
  if (data.mentioned_details && data.mentioned_details.length) {
    html += `<div style="font-size:10px;color:#6e7681;margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em">
      Nodes in your query (${data.mentioned_details.length})</div>`;
    html += `<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:14px">`;
    data.mentioned_details.forEach(n => {
      const sevColor = n.severity === 'critical' ? '#f85149' : n.severity === 'high' ? '#e3b341' : '#58a6ff';
      html += `<div onclick="selectNodeById('${n.node}')"
        style="font-size:10px;padding:3px 8px;background:#0d1117;border:1px solid #30363d;border-radius:4px;color:${sevColor};cursor:pointer;white-space:nowrap">
        ${n.name}</div>`;
    });
    html += `</div>`;
  }

  // ── Arity mismatch warnings — rendered before tier nodes ──
  if (data.arity_warnings && data.arity_warnings.length) {
    html += `<div class="arity-panel">
      <div class="arity-banner">
        <span class="arity-icon">⚠</span>
        <div>
          <div class="arity-label">Tuple arity mismatch${data.arity_warnings.length > 1 ? 'es' : ''} detected</div>
          <div class="arity-sublabel">These nodes will raise ValueError at runtime — arity inferred from AST</div>
        </div>
      </div>`;
    data.arity_warnings.forEach(w => {
      const fname = (w.file || '').split(/[/\\]/).pop();
      const loc = fname ? `${fname}${w.line ? ':' + w.line : ''}` : '';
      const msg = (w.message || '').replace(/`([^`]+)`/g,
        '<code style="background:#21262d;padding:1px 4px;border-radius:3px;color:#f85149;font-size:10px">$1</code>');
      html += `<div class="arity-card" onclick="selectNodeById('${w.node}')">
        <div class="arity-node">${w.name}</div>
        <div class="arity-loc">${loc} · expects ${w.consumer_arity}-tuple, will receive ${w.source_arity}-tuple</div>
        <div class="arity-msg">${msg}</div>
      </div>`;
    });
    html += `</div>`;
  }

  // ── Tier sections — collapsed for decision, normal for others ──
  const hasTiers = (data.will_break?.length || data.likely_need?.length || data.be_aware?.length);
  const isDecision = data.intent === 'decision';

  if (hasTiers) {
    if (isDecision) {
      const _breakdownTotal = (data.will_break?.length||0) + (data.likely_need?.length||0) + (data.be_aware?.length||0);
      html += `<button class="breakdown-toggle" data-total="${_breakdownTotal}" onclick="toggleBreakdown(this)">▸ See full node breakdown (${_breakdownTotal} nodes)</button>
      <div class="breakdown-section">`;
    }

    // ── Tier 1: will break ────────────────────────────────────
    if (data.will_break && data.will_break.length) {
      html += `<div class="tier-section">
        <div class="tier-header">
          <span class="tier-label t1">Will break</span>
          <span style="font-size:10px;color:#f85149;opacity:.6">${data.will_break.length} node${data.will_break.length > 1 ? 's' : ''}</span>
          <span class="tier-action t1">must fix</span>
        </div>`;
      data.will_break.forEach(n => { html += renderCqNode(n, 1); });
      html += `</div>`;
    }

    // ── Tier 2: likely need ───────────────────────────────────
    if (data.likely_need && data.likely_need.length) {
      html += `<div class="tier-section">
        <div class="tier-header">
          <span class="tier-label t2">Likely needs update</span>
          <span style="font-size:10px;color:#6e7681">${data.likely_need.length} node${data.likely_need.length > 1 ? 's' : ''}</span>
          <span class="tier-action">review before shipping</span>
        </div>`;
      data.likely_need.forEach(n => { html += renderCqNode(n, 2); });
      html += `</div>`;
    }

    // ── Tier 3: in scope ──────────────────────────────────────
    if (data.be_aware && data.be_aware.length) {
      html += `<div class="tier-section">
        <div class="tier-header">
          <span class="tier-label t3">In scope</span>
          <span style="font-size:10px;color:#6e7681">${data.be_aware.length} node${data.be_aware.length > 1 ? 's' : ''}</span>
          <span class="tier-action">broader impact area</span>
        </div>`;
      data.be_aware.forEach(n => { html += renderCqNode(n, 3); });
      html += `</div>`;
    }

    if (isDecision) {
      html += `</div>`; // close .breakdown-section
    }
  }

  document.getElementById('sidebar-title').textContent = 'Consequence analysis';
  // Prepend back button + copy-markdown button
  const copyBtn = `<button class="copy-md-btn" id="btn-copy-md">Copy as Markdown</button>`;
  html = `<button class="back-btn" onclick="backToExplore()">← Back to explorer</button>${copyBtn}` + html;
  document.getElementById('sidebar-content').innerHTML = html;
  // Wire copy button after DOM insertion — store data on element to avoid closure issues
  const copyEl = document.getElementById('btn-copy-md');
  if (copyEl) copyEl.onclick = () => copyAsMarkdown(copyEl, data);
}

function renderCqNode(n, tier) {
  const fname  = (n.file || '').split(/[/\\]/).pop();
  const loc    = fname ? `${_h(fname)}${n.line ? ':' + _h(n.line) : ''}` : '';
  const edgeTypes = (n.edge_types || []).slice(0, 3);
  const via    = (n.via || []).map(_h).join(', ');

  // Consequence sentence — escape first, then substitute backtick→<code>
  const consequence = _h(n.consequence || '').replace(
    /`([^`]+)`/g,
    '<code style="background:#21262d;padding:1px 4px;border-radius:3px;color:#79c0ff;font-size:10px">$1</code>'
  );

  // Severity dot colour per tier
  const dotColor = tier === 1 ? '#f85149' : tier === 2 ? '#e3b341' : '#6e7681';
  const pillLabel = tier === 1 ? 'breaks' : tier === 2 ? 'review' : 'in scope';

  // Badges for detail row
  const hookBadge = n.is_hook
    ? `<span class="cq-badge hook-badge">hook<span class="badge-tip">PyTorch Lightning hook — the framework enforces this signature contract at runtime.</span></span>`
    : '';
  const shapesBadge = n.shapes && Object.keys(n.shapes).length
    ? `<span class="cq-badge shape-badge">shapes<span class="badge-tip">This node has known tensor shape contracts.</span></span>`
    : '';
  const edgeBadges = edgeTypes.map(et => {
    const tip = EDGE_TOOLTIP_TEXT[et] || '';
    return `<span class="cq-badge">${_h(et)}${tip ? `<span class="badge-tip">${_h(tip)}</span>` : ''}</span>`;
  }).join('');

  const hasDetail = loc || via || hookBadge || shapesBadge || edgeBadges;
  const blastNote = n.intersection_count > 1
    ? `<span class="cq-meta-item">· ${_h(n.intersection_count)} blast radii</span>` : '';
  const viaNote = via
    ? `<span class="cq-meta-item">via ${via}</span>` : '';

  return `<div class="cq-node tier-${_h(tier)}" id="cqnode-${_h(n.node.replace(/[^a-z0-9]/gi,'_'))}">
    <div class="cq-header" onclick="selectNodeById('${_h(n.node)}')">
      <div class="cq-sev-dot" style="background:${dotColor}"></div>
      <div class="cq-name">${_h(n.name)}</div>
      <span class="cq-tier-pill cq-pill-${_h(tier)}">${pillLabel}</span>
    </div>
    ${consequence
      ? `<div class="cq-consequence">${consequence}</div>`
      : `<div class="cq-no-consequence">No specific consequence modelled for this change type.</div>`
    }
    ${hasDetail ? `
    <button class="cq-expand-btn" onclick="toggleCqDetail(this, event)">▸ ${loc || 'details'}</button>
    <div class="cq-detail">
      <div class="cq-meta-row">
        ${loc ? `<span class="cq-meta-item file">${loc}</span>` : ''}
        ${viaNote}${blastNote}
        ${hookBadge}${shapesBadge}${edgeBadges}
      </div>
    </div>` : ''}
  </div>`;
}

function toggleCqDetail(btn, e) {
  e.stopPropagation(); // don't fire selectNodeById on the parent
  const card = btn.closest('.cq-node');
  const isExpanded = card.classList.toggle('expanded');
  btn.textContent = (isExpanded ? '▾ ' : '▸ ') + (btn.textContent.slice(2));
}

function toggleBreakdown(btn) {
  const section = btn.nextElementSibling;
  const isOpen = section.classList.toggle('open');
  btn.classList.toggle('open', isOpen);
  const total = btn.dataset.total || '';
  btn.textContent = isOpen
    ? `▾ Hide node breakdown`
    : `▸ See full node breakdown (${total} nodes)`;
}

function backToExplore() {
  document.getElementById('sidebar-title').textContent = 'Start exploring';
  const content = document.getElementById('sidebar-content');
  content.innerHTML = `
    <div class="sidebar-intro">
      <strong>Two ways to explore:</strong><br>
      Click any node in the graph to see what it affects — or describe a change in the panel below and get a ranked consequence breakdown.
    </div>
    <div class="hot-nodes-section">
      <div class="hot-nodes-label">Most structurally central nodes</div>
      <div id="hot-nodes-list"><div style="color:#6e7681;font-size:11px">Loading...</div></div>
    </div>`;
  selectedNodeId = null;
  // Restore all nodes to visible — we're back in full graph mode
  if (viewMode === 'focus') {
    viewMode = 'full';
    applyFocusVisibility();
    updateStatsBar();
  }
  // Reload hot nodes
  loadHotNodes();
  // Clear graph highlights
  if (nodeEl) nodeEl.classed('dimmed', false).classed('highlighted', false);
  if (linkEl) linkEl.classed('dimmed', false);
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
    <div class="search-result" onclick="selectNodeById('${r.id}'); searchResults.style.display='none'; searchBox.value='';">
      <span style="color:${NODE_COLORS[r.type]||'#8b949e'}">${r.name}</span>
      <span class="node-type">${r.type} · ${r.id.split('.').slice(0,-1).join('.')}</span>
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
    document.getElementById('sidebar-title').textContent = 'Diff vs main';
    document.getElementById('sidebar-content').innerHTML = `
      <div class="cq-error-card" style="border-color:#30363d;background:#161b22">
        <div class="cq-error-title" style="color:#8b949e">No changes detected</div>
        <div class="cq-error-hint">No modified Python functions found vs main branch. Commit or stage some changes first.</div>
      </div>`;
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
  document.getElementById('sidebar-title').textContent = 'Diff impact vs main';
  document.getElementById('sidebar-content').innerHTML = html;
}

// Reindex
async function reindex() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('loading').textContent = 'Re-indexing...';
  await fetch('/api/reindex', { method: 'POST' });
  location.reload();
}

// ── Ego network / focus view ──────────────────────────────────────────────────

/**
 * Returns the Set of node IDs within `hops` of any seed node.
 * Seeds can be one node ID (string) or an array of IDs.
 */
function computeEgoNetwork(seeds, hops = 1) {
  const seedArr = Array.isArray(seeds) ? seeds : [seeds];
  const visible = new Set(seedArr);

  // Build adjacency — bidirectional so ego network is symmetric
  const adj = {};
  allNodes.forEach(n => { adj[n.id] = new Set(); });
  allEdges.forEach(e => {
    const s = typeof e.source === 'object' ? e.source.id : e.source;
    const t = typeof e.target === 'object' ? e.target.id : e.target;
    if (adj[s]) adj[s].add(t);
    if (adj[t]) adj[t].add(s);
  });

  let frontier = new Set(seedArr);
  for (let h = 0; h < hops; h++) {
    const next = new Set();
    frontier.forEach(id => {
      (adj[id] || new Set()).forEach(nb => {
        if (!visible.has(nb)) { visible.add(nb); next.add(nb); }
      });
    });
    frontier = next;
  }
  return visible;
}

/** Compute the default focus: top N hubs + their 1-hop neighbours */
function computeDefaultFocus(n = 7) {
  const hubs = [...allNodes]
    .sort((a, b) => ((b.in_degree || 0) + (b.out_degree || 0)) - ((a.in_degree || 0) + (a.out_degree || 0)))
    .slice(0, n)
    .map(n => n.id);
  return computeEgoNetwork(hubs, 1);
}

/** Apply focus visibility to D3 node/link selections */
function applyFocusVisibility() {
  if (!nodeEl || !linkEl) return;

  if (viewMode === 'full') {
    nodeEl.classed('focus-hidden', false).classed('focus-visible', true).classed('ego-node', false);
    linkEl.classed('focus-hidden', false).classed('focus-visible', true);
    return;
  }

  // focus mode
  nodeEl
    .classed('focus-visible', d => visibleNodeIds.has(d.id))
    .classed('focus-hidden',  d => !visibleNodeIds.has(d.id))
    .classed('ego-node',      d => d.id === selectedNodeId);

  linkEl.classed('focus-visible', d => {
    const s = typeof d.source === 'object' ? d.source.id : d.source;
    const t = typeof d.target === 'object' ? d.target.id : d.target;
    return visibleNodeIds.has(s) && visibleNodeIds.has(t);
  }).classed('focus-hidden', d => {
    const s = typeof d.source === 'object' ? d.source.id : d.source;
    const t = typeof d.target === 'object' ? d.target.id : d.target;
    return !visibleNodeIds.has(s) || !visibleNodeIds.has(t);
  });

  updateStatsBar();
}

function updateStatsBar() {
  const total = allNodes.length;
  const shown = viewMode === 'full' ? total : visibleNodeIds.size;
  const btn = document.getElementById('btn-view-toggle');
  if (viewMode === 'focus') {
    document.getElementById('stats-bar').textContent =
      `Showing ${shown} of ${total} nodes · ${allEdges.length} edges`;
    btn.textContent = 'Focus view';
    btn.classList.remove('full-mode');
  } else {
    document.getElementById('stats-bar').textContent =
      `${total} nodes · ${allEdges.length} edges (full graph)`;
    btn.textContent = 'Full graph';
    btn.classList.add('full-mode');
  }
}

function toggleViewMode() {
  viewMode = viewMode === 'focus' ? 'full' : 'focus';
  if (viewMode === 'focus' && selectedNodeId) {
    visibleNodeIds = computeEgoNetwork(selectedNodeId, 2);
  } else if (viewMode === 'focus') {
    visibleNodeIds = computeDefaultFocus();
  }
  applyFocusVisibility();
}

// ── Onboarding ────────────────────────────────────────────────────────────────

const EDGE_TOOLTIP_TEXT = {
  produces_tensor_for: "This node outputs a tensor that feeds directly into the target — shape or format changes cascade downstream.",
  consumes_format:     "This node expects data in a specific format from the source. If the source format changes, this node breaks.",
  calls:               "Direct function call — a signature change in the source breaks this call site.",
  inherits:            "Class inheritance — the child class is affected by changes to parent methods or attributes.",
  overrides_hook:      "PyTorch Lightning hook — the framework enforces this signature contract. Changing it breaks the training loop.",
  reads_config:        "This node reads a config key — renaming or removing the key causes a silent None return or KeyError.",
  writes_config:       "This node writes a config value — changes affect all downstream readers of this key.",
  named_reference:     "String or docstring reference — won't break at runtime but becomes stale documentation on rename.",
  imports:             "Import dependency — renaming or moving the source breaks this import.",
  instantiates:        "This node creates an instance of the target class.",
};

// ── 3-step tour ───────────────────────────────────────────────────────────────

let _tourStep = 0;
const _TOUR_STEPS = 3;

function tourGoto(step) {
  // Update slides
  document.querySelectorAll('.tour-slide').forEach((el, i) => {
    el.classList.toggle('active', i === step);
  });
  // Update dots
  document.querySelectorAll('.tour-step-dot').forEach((el, i) => {
    el.classList.toggle('active', i === step);
    el.classList.toggle('done', i < step);
  });
  document.getElementById('tour-step-label').textContent = `Step ${step + 1} of ${_TOUR_STEPS}`;
  _tourStep = step;
}

function tourNext() {
  if (_tourStep < _TOUR_STEPS - 1) tourGoto(_tourStep + 1);
  else dismissWelcome();
}

function tourBack() {
  if (_tourStep > 0) tourGoto(_tourStep - 1);
}

function dismissWelcome() {
  const overlay = document.getElementById('welcome-overlay');
  overlay.style.animation = 'fadeOut 0.2s ease forwards';
  setTimeout(() => overlay.classList.add('hidden'), 200);
  if (!document.getElementById('fadeout-style')) {
    const s = document.createElement('style');
    s.id = 'fadeout-style';
    s.textContent = '@keyframes fadeOut { to { opacity:0; } }';
    document.head.appendChild(s);
  }
  // Pulse the consequence panel to draw attention after dismissal
  setTimeout(() => {
    const ta = document.getElementById('consequence-query');
    ta.classList.add('pulse-hint');
    ta.addEventListener('animationend', () => ta.classList.remove('pulse-hint'), { once: true });
  }, 400);
}

function dismissAndQuery(queryText) {
  dismissWelcome();
  setTimeout(() => {
    const ta = document.getElementById('consequence-query');
    ta.value = queryText;
    ta.dataset.anchorId = '';
    runConsequence();
  }, 300);
}

// ── Keep-alive ping (prevents Render free tier from spinning down) ─────────────
// Render spins down after 15 min of inactivity. Ping /api/stats every 10 min.
(function startKeepAlive() {
  const INTERVAL_MS = 10 * 60 * 1000; // 10 minutes
  setInterval(async () => {
    try { await fetch('/api/stats'); } catch (_) { /* silent */ }
  }, INTERVAL_MS);
})();

function fillQuery(text) {
  const ta = document.getElementById('consequence-query');
  ta.value = text;
  ta.dataset.anchorId = '';
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  // Brief pulse on the button
  const btn = document.getElementById('btn-consequence');
  btn.style.background = '#2ea043';
  setTimeout(() => btn.style.background = '', 600);
}

async function loadHotNodes() {
  if (!allNodes.length) return;
  // Score by in_degree + out_degree — most structurally central
  const scored = allNodes
    .filter(n => !n.id.startsWith('hook::') && !n.id.startsWith('config::') && !n.id.startsWith('tensor_contract::'))
    .map(n => ({ ...n, score: (n.in_degree || 0) * 2 + (n.out_degree || 0) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);

  const container = document.getElementById('hot-nodes-list');
  if (!container) return;

  // Fetch severity for each via impact endpoint
  const cards = await Promise.all(scored.map(async n => {
    try {
      const r = await fetch(`/api/impact/${encodeURIComponent(n.id)}?depth=1`);
      const d = await r.json();
      return { ...n, severity: d.severity || 'low', downstream: d.blast_radius?.downstream_count || 0 };
    } catch { return { ...n, severity: 'low', downstream: 0 }; }
  }));

  const COLOR_MAP = { function:'#58a6ff', class:'#d2a8ff', module:'#3fb950', data_format:'#79c0ff', unknown:'#8b949e' };

  container.innerHTML = cards.map(n => `
    <div class="hot-node-card" onclick="selectNodeById('${_h(n.id)}')">
      <div class="hot-node-dot" style="background:${_h(COLOR_MAP[n.type] || '#8b949e')}"></div>
      <div class="hot-node-info">
        <div class="hot-node-name">${_h(n.name)}</div>
        <div class="hot-node-meta">${_h(n.type)} · ${_h(n.downstream)} downstream</div>
      </div>
      <span class="hot-node-sev hot-sev-${_h(n.severity)}">${_h(n.severity)}</span>
    </div>`).join('');
}

// ── Help modal ────────────────────────────────────────────────────────────────

function toggleHelp() {
  const overlay = document.getElementById('help-overlay');
  overlay.classList.toggle('hidden');
}

// ── Copy as Markdown ──────────────────────────────────────────────────────────

function buildMarkdown(data) {
  const lines = [];
  const lead = data.lead || {};

  lines.push(`## Consequence analysis`);
  lines.push(`> Query: *${data.query}*`);
  lines.push('');

  if (lead.format === 'decision' && lead.recommendation) {
    lines.push(`### Decision`);
    // strip backtick-code to plain text for markdown
    lines.push(lead.recommendation.replace(/`([^`]+)`/g, '`$1`'));
    if (lead.options && lead.options.length) {
      lines.push('');
      const sorted = [...lead.options].sort((a,b) => a.downstream_count - b.downstream_count);
      sorted.forEach((opt, i) => {
        const marker = i === 0 ? '✅' : '❌';
        lines.push(`- ${marker} **${opt.name}** — ${opt.downstream_count} nodes in impact scope`);
      });
    }
  } else if (lead.summary) {
    lines.push(lead.summary.replace(/`([^`]+)`/g, '`$1`'));
  }

  lines.push('');

  const tiers = [
    { key: 'will_break',  label: 'Will break — must fix' },
    { key: 'likely_need', label: 'Likely needs update — review before shipping' },
    { key: 'be_aware',    label: 'In scope — broader impact area' },
  ];

  for (const { key, label } of tiers) {
    if (data[key] && data[key].length) {
      lines.push(`### ${label}`);
      data[key].forEach(n => {
        const fname = (n.file || '').split(/[/\\]/).pop();
        const loc = fname ? ` (${fname}${n.line ? ':' + n.line : ''})` : '';
        lines.push(`- **\`${n.name}\`**${loc}`);
        if (n.consequence) {
          lines.push(`  ${n.consequence}`);
        }
      });
      lines.push('');
    }
  }

  lines.push(`---`);
  lines.push(`*Generated by [consequencegraph](https://github.com/Raj-Taware/consequence_graph)*`);
  return lines.join('\n');
}

function copyAsMarkdown(btn, data) {
  const md = buildMarkdown(data);
  navigator.clipboard.writeText(md).then(() => {
    btn.textContent = '✓ Copied';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.textContent = 'Copy as Markdown';
      btn.classList.remove('copied');
    }, 2000);
  }).catch(() => {
    // Fallback for non-HTTPS environments
    const ta = document.createElement('textarea');
    ta.value = md;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    btn.textContent = '✓ Copied';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy as Markdown'; btn.classList.remove('copied'); }, 2000);
  });
}

// ── Query history ─────────────────────────────────────────────────────────────

const HISTORY_KEY = 'cq_query_history';
const HISTORY_MAX = 8;
let historyIndex = -1;

function getHistory() {
  try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}

function pushHistory(query) {
  if (!query.trim()) return;
  const h = getHistory().filter(q => q !== query);
  h.unshift(query);
  sessionStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, HISTORY_MAX)));
  historyIndex = -1;
}

function navigateHistory(direction) {
  const h = getHistory();
  if (!h.length) return;
  const ta = document.getElementById('consequence-query');
  if (direction === 'up') {
    historyIndex = Math.min(historyIndex + 1, h.length - 1);
  } else {
    historyIndex = Math.max(historyIndex - 1, -1);
  }
  ta.value = historyIndex >= 0 ? h[historyIndex] : '';
  ta.setSelectionRange(ta.value.length, ta.value.length);
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  const tag = document.activeElement?.tagName;
  const inInput = tag === 'INPUT' || tag === 'TEXTAREA';
  const cqFocused = document.activeElement?.id === 'consequence-query';
  const searchFocused = document.activeElement?.id === 'search-box';

  // ? — open help (only when not typing)
  if (e.key === '?' && !inInput) {
    e.preventDefault();
    toggleHelp();
    return;
  }

  // Esc — close help, or back to explorer
  if (e.key === 'Escape') {
    const help = document.getElementById('help-overlay');
    if (!help.classList.contains('hidden')) { toggleHelp(); return; }
    if (!inInput) backToExplore();
    if (cqFocused || searchFocused) document.activeElement.blur();
    return;
  }

  // / — focus search box
  if (e.key === '/' && !inInput) {
    e.preventDefault();
    document.getElementById('search-box').focus();
    return;
  }

  // Inside consequence query textarea
  if (cqFocused) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      runConsequence();
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      navigateHistory('up');
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      navigateHistory('down');
      return;
    }
  }
});

// Show history hint when textarea is focused and history exists
document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('consequence-query');
  const hint = document.getElementById('history-hint');
  if (ta && hint) {
    ta.addEventListener('focus', () => {
      if (getHistory().length > 0) hint.classList.add('visible');
    });
    ta.addEventListener('blur', () => hint.classList.remove('visible'));
  }
});

// ── Init ──────────────────────────────────────────────────────────────────────

loadGraph().then(() => loadHotNodes());
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
    parser.add_argument("--preset", default=None, choices=["neural_lam", "wmg", "wmg_neural_lam"])
    parser.add_argument("--port", type=int, default=7842)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    build_or_load_graph(args.path, preset=args.preset, force_reindex=args.reindex)
    print(f"[consequencegraph server] Open http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")  # M3: enable access logs


if __name__ == "__main__":
    main()
