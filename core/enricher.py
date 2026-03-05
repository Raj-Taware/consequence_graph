"""
Semantic enricher.
Runs after the AST indexer to add higher-level edges:
  - tensor shape propagation (produces_tensor_for between functions)
  - data format consistency edges
  - naming-change risk detection
  - cross-function config key usage collation
"""
from .graph import KnowledgeGraph, NodeType, EdgeType, EdgeData, EDGE_SEVERITY_WEIGHTS


class Enricher:
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def run(self):
        self._link_tensor_contracts()
        self._link_config_writers()
        self._link_data_formats()
        self._flag_naming_risks()

    # ── Tensor contract propagation ───────────────────────────────────────────

    def _link_tensor_contracts(self):
        """
        For every tensor_contract node, find functions that use the same
        parameter names and create produces_tensor_for edges between the
        source function and downstream functions.
        """
        contract_nodes = [
            (nid, d) for nid, d in self.graph.g.nodes(data=True)
            if d.get("node_type") == NodeType.TENSOR_CONTRACT.value
        ]
        for contract_id, contract_data in contract_nodes:
            shapes = contract_data.get("tensor_shapes", {})
            if not shapes:
                continue

            # Find the function that owns this contract
            producers = [
                src for src, dst, d in self.graph.g.edges(data=True)
                if dst == contract_id and d.get("edge_type") == EdgeType.PRODUCES_TENSOR.value
            ]

            # Find other functions that reference the same parameter names
            for func_id, func_data in self.graph.g.nodes(data=True):
                if func_data.get("node_type") != NodeType.FUNCTION.value:
                    continue
                func_shapes = func_data.get("tensor_shapes", {})
                shared_params = set(shapes.keys()) & set(func_shapes.keys())
                if shared_params and func_id not in producers:
                    for producer in producers:
                        if producer != func_id:
                            self.graph.add_edge(producer, func_id, EdgeData(
                                edge_type=EdgeType.PRODUCES_TENSOR,
                                reason=(
                                    f"Shared tensor param(s): {', '.join(shared_params)}. "
                                    f"Shape change in {producer} propagates to {func_id}."
                                ),
                                severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.PRODUCES_TENSOR],
                            ))

    # ── Config writer/reader linkage ──────────────────────────────────────────

    def _link_config_writers(self):
        """
        If a function writes a config key and another function reads it,
        add a writes_config edge from the writer to the config node.
        """
        for func_id, func_data in self.graph.g.nodes(data=True):
            if func_data.get("node_type") != NodeType.FUNCTION.value:
                continue
            # Heuristic: __init__ functions that set self.hparams or save_hyperparameters
            if func_data.get("name") in ("__init__", "save_hyperparameters", "setup"):
                config_keys = func_data.get("config_keys", [])
                for key in config_keys:
                    cfg_id = f"config::{key}"
                    if self.graph.has_node(cfg_id):
                        self.graph.add_edge(func_id, cfg_id, EdgeData(
                            edge_type=EdgeType.WRITES_CONFIG,
                            reason=f"{func_id} initialises config key '{key}'",
                            severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.WRITES_CONFIG],
                        ))

    # ── Data format edges ─────────────────────────────────────────────────────

    def _link_data_formats(self):
        """
        Inject consumes_format edges along known data pipeline stages.
        Only Dataset subclasses (torch.utils.data.Dataset) feed training hooks —
        not every class that happens to define __getitem__.
        """
        # Find __getitem__ nodes that belong to genuine Dataset subclasses
        dataset_getitem_nodes = [
            nid for nid, d in self.graph.g.nodes(data=True)
            if d.get("name") == "__getitem__"
            and d.get("node_type") == NodeType.FUNCTION.value
            and self._parent_is_dataset(nid)
        ]

        training_nodes = [
            nid for nid, d in self.graph.g.nodes(data=True)
            if d.get("name") in ("training_step", "validation_step", "test_step")
            and d.get("node_type") == NodeType.FUNCTION.value
        ]

        for src in dataset_getitem_nodes:
            for dst in training_nodes:
                self.graph.add_edge(src, dst, EdgeData(
                    edge_type=EdgeType.CONSUMES_FORMAT,
                    reason=(
                        f"Dataset.__getitem__ produces batches consumed by "
                        f"{dst.split('.')[-1]}. Data format changes cascade here."
                    ),
                    severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.CONSUMES_FORMAT],
                ))

        # Bug 1 fix: also connect to common_step and unroll_prediction —
        # Lightning routes batch through training_step → common_step implicitly.
        # Static analysis can't see this, so we inject the edge explicitly.
        common_step_nodes = [
            nid for nid, d in self.graph.g.nodes(data=True)
            if d.get("name") in ("common_step", "unroll_prediction", "predict_step",
                                  "on_validation_epoch_end", "on_train_epoch_end")
            and d.get("node_type") == NodeType.FUNCTION.value
        ]
        for src in dataset_getitem_nodes:
            for dst in common_step_nodes:
                self.graph.add_edge(src, dst, EdgeData(
                    edge_type=EdgeType.CONSUMES_FORMAT,
                    reason=(
                        f"Dataset.__getitem__ batch flows through Lightning's lifecycle into "
                        f"{dst.split('.')[-1]}. Tuple unpacking here breaks if __getitem__ "
                        f"return structure changes."
                    ),
                    severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.CONSUMES_FORMAT],
                ))

        # create_graph / build_graph → forward (graph topology feeds GNN)
        graph_builders = [
            nid for nid, d in self.graph.g.nodes(data=True)
            if d.get("name") in ("create_graph", "build_graph", "_create_mesh_graph",
                                  "_create_graph", "build_graph_for_domain")
            and d.get("node_type") == NodeType.FUNCTION.value
        ]
        forward_nodes = [
            nid for nid, d in self.graph.g.nodes(data=True)
            if d.get("name") == "forward"
            and d.get("node_type") == NodeType.FUNCTION.value
        ]
        for src in graph_builders:
            for dst in forward_nodes:
                self.graph.add_edge(src, dst, EdgeData(
                    edge_type=EdgeType.CONSUMES_FORMAT,
                    reason=(
                        f"Graph builder {src.split('.')[-1]} produces graph structure "
                        f"consumed by {dst}. Edge index / feature shape changes cascade."
                    ),
                    severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.CONSUMES_FORMAT],
                ))

    def _parent_is_dataset(self, func_node_id: str) -> bool:
        """
        Check if the function's parent class is a genuine training Dataset subclass.
        Excludes utility classes like PaddedWeatherDataset that inherit Dataset
        but aren't part of the training data pipeline.
        """
        parts = func_node_id.rsplit(".", 1)
        if len(parts) < 2:
            return False
        parent_id = parts[0]
        parent_data = self.graph.get_node(parent_id)
        if not parent_data:
            return False

        # Exclude known utility/internal Dataset subclasses
        excluded_names = {"PaddedWeatherDataset", "IterDataset", "SubsetDataset"}
        if parent_data.get("name") in excluded_names:
            return False

        base_classes = parent_data.get("base_classes", [])
        dataset_bases = {
            "Dataset", "torch.utils.data.Dataset",
            "IterableDataset", "torch.utils.data.IterableDataset",
        }
        return any(
            any(db in b for db in dataset_bases)
            for b in base_classes
        )

    # ── Naming risk flagging ──────────────────────────────────────────────────

    def _flag_naming_risks(self):
        """
        Add named_reference edges between functions that reference each other
        by name in their docstrings (heuristic for rename risk).
        """
        all_func_ids = {
            nid: d for nid, d in self.graph.g.nodes(data=True)
            if d.get("node_type") == NodeType.FUNCTION.value
        }
        name_to_ids: dict[str, list[str]] = {}
        for nid, d in all_func_ids.items():
            name = d.get("name", "")
            name_to_ids.setdefault(name, []).append(nid)

        for nid, d in all_func_ids.items():
            doc = d.get("docstring") or ""
            for name, targets in name_to_ids.items():
                if name in doc and name != d.get("name"):
                    for target in targets:
                        if target != nid:
                            self.graph.add_edge(nid, target, EdgeData(
                                edge_type=EdgeType.NAMED_REFERENCE,
                                reason=(
                                    f"Docstring of {nid} references '{name}'. "
                                    f"Renaming {name} requires updating this docstring."
                                ),
                                severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.NAMED_REFERENCE],
                            ))
