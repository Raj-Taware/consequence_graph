"""
AST-based indexer.
Walks a Python codebase and populates a KnowledgeGraph with:
  - Module, class, function nodes
  - calls, imports, inherits, instantiates, defined_in edges
  - Lightning hook detection
  - Basic tensor shape annotation extraction (from comments/type hints)
  - Config key detection (dict access / attribute access on self.hparams / yaml keys)
"""
import ast
import os
import re
import textwrap
from typing import Optional

from .graph import (
    KnowledgeGraph, NodeData, EdgeData,
    NodeType, EdgeType, LIGHTNING_HOOKS, EDGE_SEVERITY_WEIGHTS
)
from .shape_inference import infer_shapes_from_function, format_shapes_for_node


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dotted(parts: list[str]) -> str:
    return ".".join(p for p in parts if p)


def _sig_from_args(node: ast.FunctionDef) -> str:
    args = [a.arg for a in node.args.args]
    return f"({', '.join(args)})"


def _extract_docstring(node) -> Optional[str]:
    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, (ast.Constant, ast.Str))
    ):
        raw = ast.literal_eval(node.body[0].value)
        return textwrap.shorten(str(raw), width=200, placeholder="...")
    return None


def _extract_config_keys(func_node: ast.FunctionDef) -> list[str]:
    """
    Find config key reads: self.hparams.X, cfg["X"], config["X"], self.conf.X
    """
    keys = []
    for node in ast.walk(func_node):
        # self.hparams.key  or  self.conf.key
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute):
                if node.value.attr in ("hparams", "conf", "config", "cfg"):
                    keys.append(node.attr)
        # cfg["key"] or config["key"] or self.hparams["key"]
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                keys.append(node.slice.value)
    return list(set(keys))


# Python builtins and common stdlib names — skip these in call resolution
_PYTHON_BUILTINS = frozenset({
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "list", "dict", "set", "tuple", "str", "int", "float", "bool",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
    "super", "type", "object", "property", "staticmethod", "classmethod",
    "open", "sorted", "reversed", "any", "all", "sum", "min", "max",
    "abs", "round", "repr", "hash", "id", "iter", "next",
    "append", "extend", "update", "format", "join",
    "torch", "nn", "np", "os", "sys", "math", "json", "re",
    "warnings", "logging", "pathlib", "subprocess",
})

# ── Main Indexer ──────────────────────────────────────────────────────────────

class Indexer:
    def __init__(self, graph: KnowledgeGraph, root_path: str):
        self.graph = graph
        self.root_path = os.path.abspath(root_path)

    # ── Public ────────────────────────────────────────────────────────────────

    def index_path(self, path: str):
        """Index a file or directory."""
        path = os.path.abspath(path)
        if os.path.isfile(path) and path.endswith(".py"):
            self._index_file(path)
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                # Skip hidden dirs and common noise dirs
                if any(part.startswith(".") or part in ("__pycache__", "node_modules", ".venv", "venv", "build", "dist")
                       for part in dirpath.replace("\\", "/").split("/")):
                    continue
                for fname in filenames:
                    if fname.endswith(".py"):
                        self._index_file(os.path.join(dirpath, fname))

    def reindex_file(self, file_path: str):
        """Incremental: remove old nodes for file, re-index."""
        self.graph.remove_nodes_for_file(file_path)
        self._index_file(file_path)

    # ── Private ───────────────────────────────────────────────────────────────

    def _module_id(self, file_path: str) -> str:
        rel = os.path.relpath(file_path, self.root_path)
        return rel.replace("\\", ".").replace("/", ".").removesuffix(".py")

    def _index_file(self, file_path: str):
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
            source_lines = source.splitlines()
        except SyntaxError:
            return

        module_id = self._module_id(file_path)

        # Add module node
        module_node = NodeData(
            node_id=module_id,
            node_type=NodeType.MODULE,
            name=module_id.split(".")[-1],
            module=module_id,
            file_path=file_path,
            line_no=0,
            docstring=_extract_docstring(tree),
        )
        self.graph.add_node(module_node)

        # Walk top-level imports
        self._process_imports(tree, module_id, file_path)

        # Walk top-level classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._process_class(node, module_id, file_path, [], source_lines)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._process_function(node, module_id, file_path, [], None, source_lines)

    def _process_imports(self, tree: ast.Module, module_id: str, file_path: str):
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    self._ensure_stub_module(target, file_path)
                    self.graph.add_edge(module_id, target, EdgeData(
                        edge_type=EdgeType.IMPORTS,
                        reason=f"{module_id} imports {target}",
                        severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.IMPORTS],
                    ))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    target = node.module
                    self._ensure_stub_module(target, file_path)
                    self.graph.add_edge(module_id, target, EdgeData(
                        edge_type=EdgeType.IMPORTS,
                        reason=f"{module_id} from-imports {target}",
                        severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.IMPORTS],
                    ))

    def _ensure_stub_module(self, module_id: str, file_path: str):
        if not self.graph.has_node(module_id):
            self.graph.add_node(NodeData(
                node_id=module_id,
                node_type=NodeType.MODULE,
                name=module_id.split(".")[-1],
                module=module_id,
                file_path="<external>",
                line_no=0,
            ))

    def _process_class(
        self,
        node: ast.ClassDef,
        module_id: str,
        file_path: str,
        scope: list[str],
        source_lines: list[str],
    ):
        class_id = _dotted([module_id] + scope + [node.name])
        base_names = []
        for base in node.bases:
            base_names.append(ast.unparse(base))

        is_lightning = any(
            "LightningModule" in b or "pl.LightningModule" in b
            for b in base_names
        )

        class_node = NodeData(
            node_id=class_id,
            node_type=NodeType.CLASS,
            name=node.name,
            module=module_id,
            file_path=file_path,
            line_no=node.lineno,
            docstring=_extract_docstring(node),
            base_classes=base_names,
            is_abstract=any("ABC" in b or "Abstract" in b for b in base_names),
        )
        self.graph.add_node(class_node)

        # defined_in
        self.graph.add_edge(class_id, module_id, EdgeData(
            edge_type=EdgeType.DEFINED_IN,
            reason=f"{node.name} defined in {module_id}",
            severity_weight=0,
        ))

        # inherits edges
        for base in base_names:
            # Strip common prefixes (pl., nn., pyg.) to get bare class name
            bare_base = base.split(".")[-1]

            # Resolve scoped to this module's imports first
            candidates = self.graph.resolve_node_id(bare_base, scope_module=module_id)

            # If not found scoped, try global but only if unambiguous
            if not candidates:
                candidates = self.graph.resolve_node_id(bare_base)

            if candidates:
                for c in candidates[:2]:  # max 2 — inheritance is specific
                    self.graph.add_edge(class_id, c, EdgeData(
                        edge_type=EdgeType.INHERITS,
                        reason=f"{node.name} inherits from {base}",
                        severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.INHERITS],
                    ))
            else:
                # Stub for external base (pl.LightningModule, nn.Module etc.)
                stub_id = base if "." in base else bare_base
                self._ensure_stub_module(stub_id, file_path)
                self.graph.add_edge(class_id, stub_id, EdgeData(
                    edge_type=EdgeType.INHERITS,
                    reason=f"{node.name} inherits from {base}",
                    severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.INHERITS],
                ))

        # Process methods
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._process_function(child, module_id, file_path, scope + [node.name], class_id, source_lines)
            elif isinstance(child, ast.ClassDef):
                self._process_class(child, module_id, file_path, scope + [node.name], source_lines)

    def _process_function(
        self,
        node: ast.FunctionDef,
        module_id: str,
        file_path: str,
        scope: list[str],
        parent_class_id: Optional[str],
        source_lines: list[str],
    ):
        func_id = _dotted([module_id] + scope + [node.name])
        is_hook = node.name in LIGHTNING_HOOKS and parent_class_id is not None

        # Use shape inference engine instead of old heuristic
        shape_evidence = infer_shapes_from_function(node, source_lines)
        tensor_shapes = format_shapes_for_node(shape_evidence)
        config_keys = _extract_config_keys(node)

        func_node = NodeData(
            node_id=func_id,
            node_type=NodeType.FUNCTION,
            name=node.name,
            module=module_id,
            file_path=file_path,
            line_no=node.lineno,
            docstring=_extract_docstring(node),
            signature=_sig_from_args(node),
            tensor_shapes=tensor_shapes,
            config_keys=config_keys,
            is_lightning_hook=is_hook,
        )
        self.graph.add_node(func_node)

        # defined_in
        parent_id = parent_class_id if parent_class_id else module_id
        self.graph.add_edge(func_id, parent_id, EdgeData(
            edge_type=EdgeType.DEFINED_IN,
            reason=f"{node.name} defined in {parent_id}",
            severity_weight=0,
        ))

        # Lightning hook node + edge
        if is_hook:
            hook_id = f"hook::{node.name}"
            if not self.graph.has_node(hook_id):
                self.graph.add_node(NodeData(
                    node_id=hook_id,
                    node_type=NodeType.LIGHTNING_HOOK,
                    name=node.name,
                    module="pytorch_lightning",
                    file_path="<external>",
                    line_no=0,
                ))
            self.graph.add_edge(func_id, hook_id, EdgeData(
                edge_type=EdgeType.OVERRIDES_HOOK,
                reason=f"{func_id} implements Lightning hook {node.name}",
                severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.OVERRIDES_HOOK],
            ))

        # Config key nodes + edges
        for key in config_keys:
            cfg_id = f"config::{key}"
            if not self.graph.has_node(cfg_id):
                self.graph.add_node(NodeData(
                    node_id=cfg_id,
                    node_type=NodeType.CONFIG_KEY,
                    name=key,
                    module="config",
                    file_path="<config>",
                    line_no=0,
                ))
            self.graph.add_edge(func_id, cfg_id, EdgeData(
                edge_type=EdgeType.READS_CONFIG,
                reason=f"{func_id} reads config key '{key}'",
                severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.READS_CONFIG],
            ))

        # Tensor shape contract nodes
        if tensor_shapes:
            shape_id = f"tensor_contract::{func_id}"
            if not self.graph.has_node(shape_id):
                self.graph.add_node(NodeData(
                    node_id=shape_id,
                    node_type=NodeType.TENSOR_CONTRACT,
                    name=f"shapes@{node.name}",
                    module=module_id,
                    file_path=file_path,
                    line_no=node.lineno,
                    tensor_shapes=tensor_shapes,
                ))
            self.graph.add_edge(func_id, shape_id, EdgeData(
                edge_type=EdgeType.PRODUCES_TENSOR,
                reason=f"{func_id} has tensor shape contract: {tensor_shapes}",
                severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.PRODUCES_TENSOR],
            ))

        # Walk function body for calls and instantiations
        self._process_calls(node, func_id, module_id, file_path)

    def _process_calls(
        self,
        func_node: ast.FunctionDef,
        func_id: str,
        module_id: str,
        file_path: str,
    ):
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                callee_name = None
                edge_type = EdgeType.CALLS
                # Track whether this is a self.X() call — scoped to same class
                is_self_call = False

                if isinstance(node.func, ast.Name):
                    callee_name = node.func.id
                    if callee_name[0].isupper():
                        edge_type = EdgeType.INSTANTIATES

                elif isinstance(node.func, ast.Attribute):
                    callee_name = node.func.attr
                    if callee_name[0].isupper():
                        edge_type = EdgeType.INSTANTIATES
                    # Detect self.method() — resolve within same module/class only
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                        is_self_call = True

                if not callee_name:
                    continue

                # Skip Python builtins — they're noise
                if callee_name in _PYTHON_BUILTINS:
                    continue

                # For self.method() calls, scope resolution tightly to same module
                scope = module_id if is_self_call else module_id

                candidates = self.graph.resolve_node_id(callee_name, scope_module=scope)
                if not candidates:
                    continue

                # Cap candidates and skip if still too ambiguous (>5 = noise)
                if len(candidates) > 5:
                    continue

                for c in candidates[:3]:
                    # Don't add self-loops
                    if c == func_id:
                        continue
                    verb = "instantiates" if edge_type == EdgeType.INSTANTIATES else "calls"
                    self.graph.add_edge(func_id, c, EdgeData(
                        edge_type=edge_type,
                        reason=f"{func_id} {verb} {c}",
                        severity_weight=EDGE_SEVERITY_WEIGHTS[edge_type],
                    ))
