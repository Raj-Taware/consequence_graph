"""
LLM context formatter.
Turns an impact report dict into a structured prompt fragment
ready to inject into any LLM. Supports multiple LLM adapters.
"""
import json
from typing import Optional


# ── Adapter registry ──────────────────────────────────────────────────────────

ADAPTERS = {}

def register_adapter(name: str):
    def decorator(fn):
        ADAPTERS[name] = fn
        return fn
    return decorator


# ── Core formatter ────────────────────────────────────────────────────────────

def format_impact_as_context(impact: dict, max_nodes: int = 20) -> str:
    """
    Convert an impact report into a human+LLM readable context block.
    This is the default format — paste directly into an LLM prompt.
    """
    if "error" in impact:
        return f"[codegraph ERROR] {impact['error']}\n{impact.get('hint', '')}"

    if impact.get("ambiguous"):
        matches = "\n".join(f"  - {m}" for m in impact["matches"])
        return (
            f"[codegraph] Ambiguous query '{impact['query']}'. "
            f"Possible matches:\n{matches}\n"
            f"Hint: {impact.get('hint', '')}"
        )

    lines = []
    target = impact["target"]
    meta = impact.get("target_meta", {})
    severity = impact.get("severity", "unknown")
    score = impact.get("severity_score", 0)
    change_type = impact.get("change_type", "unknown")
    depth = impact.get("depth_used", 2)

    lines.append("=" * 70)
    lines.append(f"CODEGRAPH IMPACT CONTEXT — {target}")
    lines.append("=" * 70)
    lines.append(f"Severity   : {severity.upper()} (score {score}/10)")
    lines.append(f"Change type: {change_type}")
    lines.append(f"Search depth: {depth} hops ({impact.get('depth_mode', 'auto')})")

    if meta.get("file"):
        lines.append(f"Defined at : {meta['file']}:{meta.get('line', 0)}")
    if meta.get("signature"):
        lines.append(f"Signature  : {meta['signature']}")
    if meta.get("docstring"):
        lines.append(f"Docstring  : {meta['docstring']}")
    if meta.get("tensor_shapes"):
        lines.append(f"Tensor shapes: {meta['tensor_shapes']}")
    if meta.get("config_keys"):
        lines.append(f"Config keys read: {meta['config_keys']}")
    if meta.get("is_lightning_hook"):
        lines.append("⚡ This is a PyTorch Lightning hook — changes have framework-level consequences.")

    lines.append("")
    lines.append(f"SUMMARY: {impact.get('llm_context_hint', '')}")

    critical = impact.get("critical_path", [])
    if critical:
        lines.append(f"\nCRITICAL PATH: {' → '.join(critical)}")

    br = impact.get("blast_radius", {})
    upstream = br.get("upstream", [])[:max_nodes]
    downstream = br.get("downstream", [])[:max_nodes]

    if upstream:
        lines.append(f"\n── UPSTREAM ({br.get('upstream_count', 0)} nodes — what this depends on) ──")
        for e in upstream:
            _fmt_entry(lines, e)

    if downstream:
        lines.append(f"\n── DOWNSTREAM ({br.get('downstream_count', 0)} nodes — what depends on this) ──")
        for e in downstream:
            _fmt_entry(lines, e)

    lines.append("=" * 70)
    return "\n".join(lines)


def _fmt_entry(lines: list, e: dict):
    sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(e.get("severity", "low"), "⚪")
    node = e.get("node", "?")
    edge = e.get("edge_type", "?")
    reason = e.get("reason", "")
    depth = e.get("depth", 1)
    file_loc = f"{e.get('file', '')}:{e.get('line', 0)}" if e.get("file") else ""
    shapes = f" shapes={e['tensor_shapes']}" if e.get("tensor_shapes") else ""
    hook = " ⚡hook" if e.get("is_lightning_hook") else ""
    lines.append(
        f"  {sev_icon} [{edge}] {node}{hook}{shapes}"
        + (f"\n      → {reason}" if reason else "")
        + (f"\n      @ {file_loc} (depth {depth})" if file_loc else f" (depth {depth})")
    )


# ── JSON serializer ───────────────────────────────────────────────────────────

def to_json(impact: dict, indent: int = 2) -> str:
    return json.dumps(impact, indent=indent, default=str)


# ── LLM adapters ─────────────────────────────────────────────────────────────

@register_adapter("claude")
def claude_adapter(impact: dict) -> str:
    context = format_impact_as_context(impact)
    return (
        "<context>\n"
        "The following is a dependency impact analysis from codegraph. "
        "Use it to inform your answer about consequences of the proposed change.\n\n"
        f"{context}\n"
        "</context>"
    )


@register_adapter("openai")
def openai_adapter(impact: dict) -> str:
    context = format_impact_as_context(impact)
    return (
        "SYSTEM CONTEXT (dependency impact analysis):\n"
        "----------------------------------------------\n"
        f"{context}\n"
        "----------------------------------------------\n"
        "Use the above to answer questions about what will break or need updating."
    )


@register_adapter("ollama")
def ollama_adapter(impact: dict) -> str:
    # Ollama/local models prefer terse context
    context = format_impact_as_context(impact, max_nodes=10)
    return f"[CODEBASE CONTEXT]\n{context}\n[/CODEBASE CONTEXT]"


@register_adapter("generic")
def generic_adapter(impact: dict) -> str:
    return format_impact_as_context(impact)


def format_for_llm(impact: dict, llm: str = "generic") -> str:
    adapter = ADAPTERS.get(llm, ADAPTERS["generic"])
    return adapter(impact)
