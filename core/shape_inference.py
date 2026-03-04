"""
Shape inference engine.
Replaces hardcoded preset shapes with fully inferred ones from:
  1. jaxtyping / beartype annotations: Float[Tensor, "batch nodes features"]
  2. Docstring shape patterns: (N_send, d_h), [B, N, d_state], shape: [...]
  3. Inline comments: # shape: [B, N, d_h]
  4. Return type annotations: -> tuple[Tensor, Tensor]
  5. nn.Linear / nn.Embedding weight shape inference from __init__
  6. Dataclass field annotations
All inferred shapes carry a confidence score and source label.
"""
import ast
import re
import textwrap
from dataclasses import dataclass
from typing import Optional


# ── Shape evidence ────────────────────────────────────────────────────────────

@dataclass
class ShapeEvidence:
    param_name: str
    shape_str: str
    confidence: float        # 0.0 → 1.0
    source: str              # "annotation", "docstring", "comment", "linear_weight"
    line_no: int = 0

    def to_dict(self):
        return {
            "shape": self.shape_str,
            "confidence": self.confidence,
            "source": self.source,
            "line": self.line_no,
        }


# ── Pattern library ───────────────────────────────────────────────────────────

# Docstring shape patterns — ordered by specificity
_DOCSTRING_PATTERNS = [
    # param_name: (N, d_h) — numpy/pytorch convention
    (re.compile(r'(\w+)\s*:\s*\(([^)]+)\)'), 0.85, "docstring:tuple"),
    # param_name: [B, N, d] — bracket convention
    (re.compile(r'(\w+)\s*:\s*\[([^\]]+)\]'), 0.85, "docstring:bracket"),
    # shape: (B, N, d) — explicit shape label
    (re.compile(r'shape[:\s]+\(([^)]+)\)'), 0.80, "docstring:shape_label"),
    # (B, N, d_h) — bare tuple, context needed
    (re.compile(r'\((\s*\w+\s*(?:,\s*\w+\s*)+)\)'), 0.60, "docstring:bare_tuple"),
    # "batch nodes features" — jaxtyping string style in docstring
    (re.compile(r'"([a-z_]+(?:\s+[a-z_]+)+)"'), 0.55, "docstring:jaxtyping_str"),
]

# Inline comment patterns
_COMMENT_PATTERNS = [
    # # shape: [B, N, d_h]
    (re.compile(r'#\s*shape\s*:\s*\[([^\]]+)\]'), 0.90, "comment:explicit"),
    # # [B, N, d_h]
    (re.compile(r'#\s*\[([A-Z_][^\]]*)\]'), 0.80, "comment:bracket"),
    # # (B, N, d_h)
    (re.compile(r'#\s*\(([A-Z_][^)]*)\)'), 0.75, "comment:tuple"),
]

# Known tensor annotation types
_TENSOR_TYPES = frozenset({
    "Tensor", "torch.Tensor", "FloatTensor", "LongTensor", "BoolTensor",
    "Float", "Int", "Bool",  # jaxtyping shorthand
    "ndarray", "np.ndarray",
})


# ── Main inference functions ──────────────────────────────────────────────────

def infer_shapes_from_function(
    func_node: ast.FunctionDef,
    source_lines: list[str],
) -> dict[str, ShapeEvidence]:
    """
    Run all inference strategies on a function node.
    Returns dict of param_name → best ShapeEvidence.
    """
    all_evidence: dict[str, list[ShapeEvidence]] = {}

    def add(ev: ShapeEvidence):
        all_evidence.setdefault(ev.param_name, []).append(ev)

    # 1. Type annotations on function args
    for ev in _infer_from_annotations(func_node):
        add(ev)

    # 2. Return annotation
    for ev in _infer_from_return_annotation(func_node):
        add(ev)

    # 3. Docstring
    doc = _get_docstring(func_node)
    if doc:
        for ev in _infer_from_docstring(doc, func_node):
            add(ev)

    # 4. Inline comments in function body
    for ev in _infer_from_comments(func_node, source_lines):
        add(ev)

    # 5. nn.Linear weight shapes in __init__
    if func_node.name == "__init__":
        for ev in _infer_from_linear_layers(func_node):
            add(ev)

    # Pick highest-confidence evidence per param
    best: dict[str, ShapeEvidence] = {}
    for param, evidences in all_evidence.items():
        best[param] = max(evidences, key=lambda e: e.confidence)

    return best


def infer_shapes_from_class(
    class_node: ast.ClassDef,
    source_lines: list[str],
) -> dict[str, ShapeEvidence]:
    """Infer shapes from class-level docstring and field annotations."""
    evidence: dict[str, list[ShapeEvidence]] = {}

    doc = _get_docstring(class_node)
    if doc:
        for ev in _infer_from_docstring(doc, class_node):
            evidence.setdefault(ev.param_name, []).append(ev)

    best = {p: max(evs, key=lambda e: e.confidence) for p, evs in evidence.items()}
    return best


# ── Strategy implementations ──────────────────────────────────────────────────

def _infer_from_annotations(func_node: ast.FunctionDef) -> list[ShapeEvidence]:
    """Extract shapes from PEP 484 / jaxtyping annotations."""
    results = []
    for arg in func_node.args.args:
        if not arg.annotation:
            continue
        ann_str = ast.unparse(arg.annotation)
        ev = _parse_annotation(ann_str, arg.arg, arg.col_offset)
        if ev:
            results.append(ev)
    return results


def _parse_annotation(ann_str: str, param_name: str, line_no: int) -> Optional[ShapeEvidence]:
    """Parse a single annotation string into ShapeEvidence."""

    # jaxtyping: Float[Tensor, "batch nodes features"]
    m = re.match(r'(\w+)\[(?:Tensor|torch\.Tensor),\s*["\']([^"\']+)["\']\]', ann_str)
    if m:
        dtype, dims = m.group(1), m.group(2)
        shape = "[" + " ".join(dims.split()) + "]"
        return ShapeEvidence(param_name, shape, 0.95, "annotation:jaxtyping", line_no)

    # Annotated[Tensor, Shape["B N D"]]
    m = re.match(r'Annotated\[.*Shape\["([^"]+)"\]', ann_str)
    if m:
        shape = "[" + m.group(1) + "]"
        return ShapeEvidence(param_name, shape, 0.90, "annotation:annotated_shape", line_no)

    # Plain tensor type — no shape info but worth noting
    if any(t in ann_str for t in _TENSOR_TYPES):
        return ShapeEvidence(param_name, ann_str, 0.40, "annotation:tensor_type", line_no)

    return None


def _infer_from_return_annotation(func_node: ast.FunctionDef) -> list[ShapeEvidence]:
    """Extract return shape from -> annotation."""
    if not func_node.returns:
        return []
    ann_str = ast.unparse(func_node.returns)
    ev = _parse_annotation(ann_str, "return", func_node.lineno)
    if ev:
        return [ev]

    # tuple[Tensor, Tensor, Tensor] — record arity
    if ann_str.startswith("tuple") or ann_str.startswith("Tuple"):
        inner = re.findall(r'Tensor|ndarray', ann_str)
        if inner:
            shape = f"tuple of {len(inner)} tensors"
            return [ShapeEvidence("return", shape, 0.50, "annotation:return_tuple", func_node.lineno)]

    return []


def _infer_from_docstring(
    doc: str,
    node: ast.AST,
) -> list[ShapeEvidence]:
    """Extract shapes from structured docstring patterns."""
    results = []
    line_no = getattr(node, "lineno", 0)

    # First pass: look for explicit parameter blocks
    # "Args:" / "Parameters:" sections
    param_block = re.search(
        r'(?:Args|Arguments|Parameters|Inputs?)\s*:\s*\n(.*?)(?:\n\s*\n|\Z)',
        doc, re.DOTALL | re.IGNORECASE
    )
    if param_block:
        block = param_block.group(1)
        # param_name (shape): description
        for m in re.finditer(
            r'(\w+)\s*\(([^)]+)\)\s*:', block
        ):
            param, shape_hint = m.group(1), m.group(2)
            # Filter: looks like a shape if it contains commas or known dim names
            if re.search(r'[,xBNDd_]', shape_hint):
                results.append(ShapeEvidence(
                    param, f"({shape_hint})", 0.85, "docstring:args_block", line_no
                ))

    # Second pass: general patterns
    for pattern, confidence, source in _DOCSTRING_PATTERNS:
        for m in pattern.finditer(doc):
            groups = m.groups()
            if len(groups) == 2:
                param_name, shape = groups
                # Filter noise — shape must look like dimensions
                if re.search(r'[A-Z_\d,\s]', shape) and len(shape) < 60:
                    results.append(ShapeEvidence(
                        param_name.strip(),
                        f"[{shape.strip()}]",
                        confidence,
                        source,
                        line_no
                    ))
            elif len(groups) == 1:
                # Bare shape — attach to "return" as best guess
                shape = groups[0]
                if re.search(r'[A-Z_\d,\s]', shape) and len(shape) < 60:
                    results.append(ShapeEvidence(
                        "return", f"[{shape.strip()}]",
                        confidence * 0.8,  # penalty for ambiguity
                        source, line_no
                    ))

    return results


def _infer_from_comments(
    func_node: ast.FunctionDef,
    source_lines: list[str],
) -> list[ShapeEvidence]:
    """Extract shapes from inline comments in function body."""
    results = []
    start = func_node.lineno - 1
    end = getattr(func_node, "end_lineno", start + 50)
    func_lines = source_lines[start:end]

    for i, line in enumerate(func_lines):
        actual_line = start + i + 1
        for pattern, confidence, source in _COMMENT_PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            shape = m.group(1).strip()
            # Try to find the variable name before this comment
            var_match = re.search(r'(\w+)\s*=|(\w+)\s*:', line[:m.start()])
            param_name = "unknown"
            if var_match:
                param_name = (var_match.group(1) or var_match.group(2) or "unknown").strip()
            # Also check if it's annotating a function arg on prev lines
            results.append(ShapeEvidence(
                param_name, f"[{shape}]", confidence, source, actual_line
            ))

    return results


def _infer_from_linear_layers(init_node: ast.FunctionDef) -> list[ShapeEvidence]:
    """
    Infer dimension names from nn.Linear / nn.Embedding calls in __init__.
    nn.Linear(hidden_dim, output_dim) tells us d_in → d_out mapping.
    """
    results = []
    for node in ast.walk(init_node):
        if not isinstance(node, ast.Call):
            continue
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        if func_name in ("Linear", "Embedding", "Conv1d", "Conv2d", "LayerNorm"):
            if len(node.args) >= 2:
                d_in = ast.unparse(node.args[0])
                d_out = ast.unparse(node.args[1])
                results.append(ShapeEvidence(
                    f"{func_name}_weight",
                    f"[{d_in}, {d_out}]",
                    0.85,
                    "linear_weight",
                    node.lineno,
                ))

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_docstring(node: ast.AST) -> Optional[str]:
    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, (ast.Constant, ast.Str))
    ):
        raw = ast.literal_eval(node.body[0].value)
        return str(raw)
    return None


def format_shapes_for_node(shapes: dict[str, ShapeEvidence]) -> dict:
    """Convert ShapeEvidence dict to a JSON-serialisable format."""
    return {
        param: ev.to_dict()
        for param, ev in shapes.items()
        if ev.confidence >= 0.5  # only include reasonably confident shapes
    }
