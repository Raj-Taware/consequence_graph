#!/usr/bin/env python3
"""
consequencegraph — Consequence-Aware Code Knowledge Graph CLI

Usage:
  consequencegraph index <path> [--preset <preset>] [--cache <cache_path>]
  consequencegraph impact <target> [--depth <n>] [--preset <preset>] [--llm <adapter>] [--cache <cache_path>]
  consequencegraph watch <path> [--preset <preset>] [--cache <cache_path>]
  consequencegraph stats [--cache <cache_path>]
  consequencegraph nodes [--type <node_type>] [--cache <cache_path>]
  consequencegraph fmt --llm <adapter> [--cache <cache_path>]
"""
import sys
import json
import argparse
import os

from core.graph import KnowledgeGraph
from core.indexer import Indexer
from core.enricher import Enricher
from core.query import QueryEngine
from output.llm_context import to_json, format_for_llm, format_impact_as_context


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_graph(cache_path: str) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.CACHE_FILE = cache_path
    loaded = graph.load(cache_path)
    if not loaded:
        print(f"[consequencegraph] ⚠  No cache found at '{cache_path}'. Run `consequencegraph index <path>` first.",
              file=sys.stderr)
    return graph, loaded


def _apply_preset(graph: KnowledgeGraph, preset: str):
    if preset == "neural_lam":
        from presets.neural_lam import apply
        apply(graph)
    elif preset:
        print(f"[consequencegraph] ⚠  Unknown preset '{preset}'. Available: neural_lam", file=sys.stderr)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_index(args):
    path = args.path
    cache_path = args.cache or os.path.join(os.getcwd(), ".consequencegraph", "cache.json")

    print(f"[consequencegraph] 🔍 Indexing: {path}")
    graph = KnowledgeGraph()
    graph.CACHE_FILE = cache_path

    indexer = Indexer(graph, path)
    indexer.index_path(path)

    print(f"[consequencegraph] 🧠 Running semantic enricher...")
    Enricher(graph).run()

    if args.preset:
        _apply_preset(graph, args.preset)

    graph.save(cache_path)
    stats = graph.stats()
    print(f"[consequencegraph] ✅ Done. {stats['total_nodes']} nodes, {stats['total_edges']} edges.")
    print(f"[consequencegraph] 💾 Cache saved to: {cache_path}")
    print(json.dumps(stats, indent=2))


def cmd_impact(args):
    cache_path = args.cache or os.path.join(os.getcwd(), ".consequencegraph", "cache.json")
    graph, loaded = _load_graph(cache_path)
    if not loaded:
        sys.exit(1)

    if args.preset:
        _apply_preset(graph, args.preset)

    depth = int(args.depth) if args.depth else None
    engine = QueryEngine(graph)
    result = engine.impact(args.target, depth=depth)

    # LLM formatter mode (piped)
    if args.llm:
        print(format_for_llm(result, llm=args.llm))
    elif args.human:
        print(format_impact_as_context(result))
    else:
        # Default: JSON to stdout (pipeable)
        print(to_json(result))


def cmd_watch(args):
    try:
        from core.watcher import Watcher
    except ImportError:
        print("[consequencegraph] ❌ watchdog not installed. Run: pip install watchdog", file=sys.stderr)
        sys.exit(1)

    path = args.path
    cache_path = args.cache or os.path.join(os.getcwd(), ".consequencegraph", "cache.json")

    # Load existing graph or build fresh
    graph = KnowledgeGraph()
    graph.CACHE_FILE = cache_path
    loaded = graph.load(cache_path)

    if not loaded:
        print(f"[consequencegraph] 🔍 No cache found. Indexing first: {path}")
        indexer = Indexer(graph, path)
        indexer.index_path(path)
        Enricher(graph).run()
        if args.preset:
            _apply_preset(graph, args.preset)
        graph.save(cache_path)
        print(f"[consequencegraph] ✅ Initial index done: {graph.node_count()} nodes")

    watcher = Watcher(graph, path)
    watcher.run_forever()


def cmd_stats(args):
    cache_path = args.cache or os.path.join(os.getcwd(), ".consequencegraph", "cache.json")
    graph, loaded = _load_graph(cache_path)
    if not loaded:
        sys.exit(1)
    print(to_json(graph.stats()))


def cmd_nodes(args):
    cache_path = args.cache or os.path.join(os.getcwd(), ".consequencegraph", "cache.json")
    graph, loaded = _load_graph(cache_path)
    if not loaded:
        sys.exit(1)

    nodes = []
    for nid, d in graph.g.nodes(data=True):
        if args.type and d.get("node_type") != args.type:
            continue
        nodes.append({
            "id": nid,
            "type": d.get("node_type"),
            "file": d.get("file_path"),
            "line": d.get("line_no"),
        })

    nodes.sort(key=lambda x: x["id"])
    print(to_json(nodes))


def cmd_fmt(args):
    """Read JSON impact from stdin, output formatted LLM context."""
    data = sys.stdin.read()
    try:
        impact = json.loads(data)
    except json.JSONDecodeError as e:
        print(f"[consequencegraph] ❌ Invalid JSON on stdin: {e}", file=sys.stderr)
        sys.exit(1)
    print(format_for_llm(impact, llm=args.llm or "generic"))


# ── CLI parser ────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="consequencegraph",
        description="Consequence-Aware Code Knowledge Graph — impact analysis for Python codebases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  consequencegraph index ./neural-lam --preset neural_lam
  consequencegraph impact GraphLAM.forward
  consequencegraph impact WeatherDataset.__getitem__ --depth 3
  consequencegraph impact ARModel.training_step --depth auto --llm claude
  consequencegraph impact training_step | consequencegraph fmt --llm openai
  consequencegraph watch ./neural-lam --preset neural_lam
  consequencegraph stats
  consequencegraph nodes --type function
        """,
    )
    parser.add_argument("--cache", default=None,
                        help="Path to cache file (default: .consequencegraph/cache.json in CWD)")

    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index a Python codebase")
    p_index.add_argument("path", help="Path to directory or file to index")
    p_index.add_argument("--preset", default=None, choices=["neural_lam"],
                         help="Apply a domain-specific preset after indexing")

    # impact
    p_impact = sub.add_parser("impact", help="Get impact report for a node")
    p_impact.add_argument("target", help="Function/class name or full node ID")
    p_impact.add_argument("--depth", default=None,
                          help="Search depth (integer or 'auto'). Default: auto")
    p_impact.add_argument("--preset", default=None, choices=["neural_lam"],
                          help="Apply domain preset before querying")
    p_impact.add_argument("--llm", default=None,
                          choices=["claude", "openai", "ollama", "generic"],
                          help="Format output for a specific LLM")
    p_impact.add_argument("--human", action="store_true",
                          help="Human-readable text output instead of JSON")

    # watch
    p_watch = sub.add_parser("watch", help="Watch a directory for changes and re-index live")
    p_watch.add_argument("path", help="Path to watch")
    p_watch.add_argument("--preset", default=None, choices=["neural_lam"])

    # stats
    sub.add_parser("stats", help="Show graph statistics")

    # nodes
    p_nodes = sub.add_parser("nodes", help="List all nodes in the graph")
    p_nodes.add_argument("--type", default=None,
                         help="Filter by node type (function, class, module, config_key, ...)")

    # fmt
    p_fmt = sub.add_parser("fmt", help="Format JSON impact (from stdin) for an LLM")
    p_fmt.add_argument("--llm", default="generic",
                       choices=["claude", "openai", "ollama", "generic"],
                       help="Target LLM adapter")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "index":  cmd_index,
        "impact": cmd_impact,
        "watch":  cmd_watch,
        "stats":  cmd_stats,
        "nodes":  cmd_nodes,
        "fmt":    cmd_fmt,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
