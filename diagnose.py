"""
diagnose.py — run this from the neural-lam root after indexing.
Windows-friendly replacement for grep-based inspection.

Usage:
    python diagnose.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.graph import KnowledgeGraph
from core.indexer import Indexer
from core.enricher import Enricher
from core.query import QueryEngine
from presets.neural_lam import apply
from output.llm_context import format_impact_as_context

NEURAL_LAM_PATH = "./neural_lam"
CACHE = ".codegraph/cache.json"

def reindex():
    print(f"Indexing {NEURAL_LAM_PATH} ...")
    graph = KnowledgeGraph()
    graph.CACHE_FILE = CACHE
    indexer = Indexer(graph, NEURAL_LAM_PATH)
    indexer.index_path(NEURAL_LAM_PATH)
    print(f"  Raw: {graph.node_count()} nodes, {graph.edge_count()} edges")
    Enricher(graph).run()
    print(f"  After enrichment: {graph.node_count()} nodes, {graph.edge_count()} edges")
    apply(graph)
    print(f"  After preset: {graph.node_count()} nodes, {graph.edge_count()} edges")
    graph.save(CACHE)
    return graph

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def run(graph):
    engine = QueryEngine(graph)

    # 1. Node type breakdown
    print_section("NODE TYPE BREAKDOWN")
    stats = graph.stats()
    for k, v in stats["node_types"].items():
        print(f"  {k:25s} {v}")

    # 2. Key functions we care about
    print_section("KEY FUNCTIONS FOUND")
    targets = [
        "training_step", "validation_step", "forward",
        "__getitem__", "predict_step", "configure_optimizers",
        "encode", "decode", "process",
    ]
    for t in targets:
        matches = graph.resolve_node_id(t)
        for m in matches:
            d = graph.get_node(m)
            hook = " ⚡" if d.get("is_lightning_hook") else ""
            shapes = f" shapes={d['tensor_shapes']}" if d.get("tensor_shapes") else ""
            print(f"  {m}{hook}{shapes}")

    # 3. All classes
    print_section("CLASSES")
    for nid, d in sorted(graph.g.nodes(data=True)):
        if d.get("node_type") == "class":
            bases = d.get("base_classes", [])
            print(f"  {nid}")
            if bases:
                print(f"    inherits: {', '.join(bases)}")

    # 4. Impact reports for the 3 most interesting nodes
    print_section("IMPACT: ARModel.training_step")
    r = engine.impact("training_step")
    if "error" not in r and "ambiguous" not in r:
        print(format_impact_as_context(r))
    else:
        print(json.dumps(r, indent=2))

    print_section("IMPACT: BaseGraphModel (or GraphLAM) forward")
    # forward is ambiguous — find all forward functions and pick the GNN one
    forward_nodes = [
        nid for nid, d in graph.g.nodes(data=True)
        if d.get("name") == "forward" and d.get("node_type") == "function"
    ]
    if not forward_nodes:
        print("No forward() functions found in graph.")
    elif len(forward_nodes) == 1:
        r = engine.impact(forward_nodes[0])
        print(format_impact_as_context(r))
    else:
        print(f"Multiple forward() nodes found ({len(forward_nodes)}):")
        for n in forward_nodes:
            print(f"  {n}")
        # Pick the most interesting one — prefer InteractionNet (the GNN core)
        gnn_forward = next((n for n in forward_nodes if "interaction" in n.lower()), forward_nodes[0])
        print(f"\nShowing impact for: {gnn_forward}")
        r = engine.impact(gnn_forward)
        print(format_impact_as_context(r))

    print_section("IMPACT: WeatherDataset.__getitem__")
    # Find the real WeatherDataset.__getitem__ node ID
    getitem_candidates = [
        nid for nid, d in graph.g.nodes(data=True)
        if d.get("name") == "__getitem__"
        and "WeatherDataset" in nid
        and "Padded" not in nid  # exclude the utility PaddedWeatherDataset
    ]
    if getitem_candidates:
        r = engine.impact(getitem_candidates[0])
        print(format_impact_as_context(r))
    else:
        r = engine.impact("WeatherDataset.__getitem__")
        print(format_impact_as_context(r) if "error" not in r else json.dumps(r, indent=2))

    # 5. Highest centrality nodes — the load-bearing ones
    print_section("HIGHEST IN-DEGREE NODES (most depended upon)")
    scored = []
    for nid in graph.g.nodes:
        d = graph.get_node(nid)
        if d.get("node_type") in ("function", "class"):
            scored.append((graph.in_degree(nid), nid))
    scored.sort(reverse=True)
    for score, nid in scored[:15]:
        print(f"  [{score:3d} dependents] {nid}")

if __name__ == "__main__":
    if "--reindex" in sys.argv or not os.path.exists(CACHE):
        graph = reindex()
    else:
        graph = KnowledgeGraph()
        graph.CACHE_FILE = CACHE
        loaded = graph.load(CACHE)
        if not loaded:
            graph = reindex()
        else:
            print(f"Loaded from cache: {graph.node_count()} nodes, {graph.edge_count()} edges")

    run(graph)
