"""
Core graph schema for consequencegraph.
Defines node types, edge types, and the central KnowledgeGraph class.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import networkx as nx


# ── Node Types ────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    FUNCTION       = "function"
    CLASS          = "class"
    MODULE         = "module"
    CONFIG_KEY     = "config_key"
    TENSOR_CONTRACT = "tensor_contract"
    LIGHTNING_HOOK = "lightning_hook"
    DATA_FORMAT    = "data_format"


# ── Edge Types ────────────────────────────────────────────────────────────────

class EdgeType(str, Enum):
    CALLS             = "calls"               # A invokes B
    INHERITS          = "inherits"            # A subclasses B
    IMPORTS           = "imports"             # module A imports B
    READS_CONFIG      = "reads_config"        # A reads config key B
    WRITES_CONFIG     = "writes_config"       # A writes config key B
    PRODUCES_TENSOR   = "produces_tensor_for" # A output shape feeds B
    OVERRIDES_HOOK    = "overrides_hook"      # A implements Lightning hook B
    CONSUMES_FORMAT   = "consumes_format"     # A expects data format from B
    NAMED_REFERENCE   = "named_reference"     # A references B by name (rename risk)
    DEFINED_IN        = "defined_in"          # entity A defined in module B
    INSTANTIATES      = "instantiates"        # A creates instance of B


EDGE_SEVERITY_WEIGHTS = {
    EdgeType.PRODUCES_TENSOR:   3,
    EdgeType.OVERRIDES_HOOK:    3,
    EdgeType.INHERITS:          3,
    EdgeType.READS_CONFIG:      2,
    EdgeType.WRITES_CONFIG:     2,
    EdgeType.CONSUMES_FORMAT:   2,
    EdgeType.CALLS:             1,
    EdgeType.INSTANTIATES:      1,
    EdgeType.NAMED_REFERENCE:   1,
    EdgeType.IMPORTS:           0,
    EdgeType.DEFINED_IN:        0,
}

LIGHTNING_HOOKS = {
    "training_step", "validation_step", "test_step",
    "configure_optimizers", "forward", "on_train_epoch_end",
    "on_validation_epoch_end", "predict_step", "configure_callbacks",
    "on_fit_start", "on_fit_end", "setup", "teardown",
}


# ── Node / Edge dataclasses ───────────────────────────────────────────────────

@dataclass
class NodeData:
    node_id: str              # e.g. "neural_lam.models.graph_lam.GraphLAM.forward"
    node_type: NodeType
    name: str                 # short name
    module: str               # dotted module path
    file_path: str
    line_no: int = 0
    docstring: Optional[str] = None
    signature: Optional[str] = None
    tensor_shapes: dict = field(default_factory=dict)   # param_name -> shape str
    config_keys: list = field(default_factory=list)
    is_lightning_hook: bool = False
    is_abstract: bool = False
    base_classes: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["node_type"] = self.node_type.value
        return d


@dataclass
class EdgeData:
    edge_type: EdgeType
    reason: str = ""
    severity_weight: int = 0

    def to_dict(self):
        return {
            "edge_type": self.edge_type.value,
            "reason": self.reason,
            "severity_weight": self.severity_weight,
        }


# ── KnowledgeGraph ────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """Central in-memory graph. Wraps NetworkX DiGraph."""

    CACHE_FILE = ".consequencegraph/cache.json"

    def __init__(self):
        self.g = nx.MultiDiGraph()

    # ── Node operations ───────────────────────────────────────────────────────

    def add_node(self, data: NodeData):
        self.g.add_node(data.node_id, **data.to_dict())

    def get_node(self, node_id: str) -> Optional[dict]:
        return self.g.nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return self.g.has_node(node_id)

    def remove_nodes_for_file(self, file_path: str):
        to_remove = [
            n for n, d in self.g.nodes(data=True)
            if d.get("file_path") == file_path
        ]
        self.g.remove_nodes_from(to_remove)

    # ── Edge operations ───────────────────────────────────────────────────────

    def add_edge(self, src: str, dst: str, data: EdgeData):
        if not self.g.has_node(src) or not self.g.has_node(dst):
            return
        self.g.add_edge(src, dst, **data.to_dict())

    # ── Query ─────────────────────────────────────────────────────────────────

    def resolve_node_id(self, query: str, scope_module: Optional[str] = None, explicit: bool = False) -> list[str]:
        """
        Fuzzy match with optional module scope.

        Priority order:
          1. Exact match
          2. Multi-part suffix match (e.g. "WeatherDataset.__getitem__" matches
             "weather_dataset.WeatherDataset.__getitem__")
          3. Suffix match scoped to modules imported by scope_module (if given)
          4. Global suffix match — only if unambiguous (exactly 1 result)
          5. Substring match — only if unambiguous (exactly 1 result)

        Dunder methods are never resolved during implicit call resolution
        (to avoid false edges) but ARE resolved when explicit=True.
        """
        # 1. Exact
        if self.g.has_node(query):
            return [query]

        # 2. Multi-part suffix — handles "ClassName.method" or "mod.Class.method"
        # Match any node whose dotted ID ends with the query (with a dot boundary)
        # Exclude meta-nodes (tensor_contract::, config::, hook::) from user-facing queries
        suffix_key = "." + query
        suffix_matches = [
            n for n in self.g.nodes
            if (n.endswith(suffix_key) or n == query)
            and not any(n.startswith(pfx) for pfx in ("tensor_contract::", "config::", "hook::"))
        ]

        # Bug 3 fix: if query is "ClassName.method", enforce that the class name segment
        # appears immediately before the method in the node ID — prevents __init__ from
        # matching every class's constructor when querying a specific one.
        if "." in query:
            parts = query.split(".")
            # Build a stricter suffix: require all parts to appear consecutively in the node ID
            strict_suffix = "." + ".".join(parts)
            strict_matches = [n for n in suffix_matches if ("." + n).endswith(strict_suffix)
                              or n == query]
            if strict_matches:
                suffix_matches = strict_matches

        # Dunder guard for implicit call resolution
        is_dunder = query.startswith("__") and query.endswith("__")
        if is_dunder and not explicit:
            return []

        # 3. Scoped suffix — prefer candidates from imported modules
        if scope_module and suffix_matches:
            imported = self._imported_modules(scope_module)
            scoped = [n for n in suffix_matches if any(n.startswith(imp) for imp in imported)]
            if scoped:
                return scoped

        # 4. Global suffix — unambiguous only (for implicit), always for explicit
        if len(suffix_matches) == 1:
            return suffix_matches
        if len(suffix_matches) > 1:
            if explicit:
                return suffix_matches  # return all, let caller disambiguate
            # implicit: ambiguous — drop it unless it's a simple name with clear winner
            return []

        # 5. Substring — only if unambiguous
        sub = [n for n in self.g.nodes if query.lower() in n.lower()]
        if len(sub) == 1:
            return sub

        return []

    def _imported_modules(self, module_id: str) -> list[str]:
        """Return dotted module IDs imported by module_id via IMPORTS edges."""
        imported = []
        if not self.g.has_node(module_id):
            return imported
        for _, dst, d in self.g.out_edges(module_id, data=True):
            if d.get("edge_type") == EdgeType.IMPORTS.value:
                imported.append(dst)
        return imported

    def upstream(self, node_id: str, depth: int = 2) -> list[dict]:
        """Nodes that node_id depends ON (predecessors)."""
        results = []
        visited = set()
        self._walk(node_id, depth, results, visited, direction="up")
        return results

    def downstream(self, node_id: str, depth: int = 2) -> list[dict]:
        """Nodes that depend ON node_id (successors)."""
        results = []
        visited = set()
        self._walk(node_id, depth, results, visited, direction="down")
        return results

    def _walk(self, node_id, depth, results, visited, direction, current_depth=1):
        if current_depth > depth or node_id in visited:
            return
        visited.add(node_id)
        if direction == "up":
            neighbors = list(self.g.predecessors(node_id))
            get_edge = lambda n: self.g.get_edge_data(n, node_id)
        else:
            neighbors = list(self.g.successors(node_id))
            get_edge = lambda n: self.g.get_edge_data(node_id, n)

        for neighbor in neighbors:
            if neighbor == node_id:
                continue
            edge_data_map = get_edge(neighbor)
            if not edge_data_map:
                continue
            # MultiDiGraph returns {0: {...}, 1: {...}} - pick highest severity
            best_edge = max(
                edge_data_map.values(),
                key=lambda e: e.get("severity_weight", 0)
            )
            node_info = dict(self.g.nodes[neighbor])
            results.append({
                "node": neighbor,
                "node_type": node_info.get("node_type"),
                "file_path": node_info.get("file_path"),
                "line_no": node_info.get("line_no"),
                "edge_type": best_edge.get("edge_type"),
                "severity_weight": best_edge.get("severity_weight", 0),
                "reason": best_edge.get("reason", ""),
                "depth": current_depth,
            })
            self._walk(neighbor, depth, results, visited, direction, current_depth + 1)

    def node_count(self) -> int:
        return self.g.number_of_nodes()

    def edge_count(self) -> int:
        return self.g.number_of_edges()

    # ── Centrality / severity ─────────────────────────────────────────────────

    def centrality_score(self, node_id: str) -> float:
        try:
            centrality = nx.degree_centrality(self.g)
            return round(centrality.get(node_id, 0.0), 4)
        except Exception:
            return 0.0

    def in_degree(self, node_id: str) -> int:
        return self.g.in_degree(node_id)

    def out_degree(self, node_id: str) -> int:
        return self.g.out_degree(node_id)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        path = path or self.CACHE_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = nx.node_link_data(self.g, edges="links")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.CACHE_FILE
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.g = nx.node_link_graph(data, directed=True, multigraph=True, edges="links")
        return True

    def stats(self) -> dict:
        node_type_counts = {}
        for _, d in self.g.nodes(data=True):
            nt = d.get("node_type", "unknown")
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1
        return {
            "total_nodes": self.node_count(),
            "total_edges": self.edge_count(),
            "node_types": node_type_counts,
        }
