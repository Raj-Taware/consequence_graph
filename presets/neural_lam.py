"""
Neural-LAM preset.
Hardcodes domain knowledge that pure AST cannot infer:
  - Known tensor shape contracts per function
  - Critical data flow paths
  - Config schema keys
  - Lightning hook contracts specific to neural-lam
  - Data format pipeline stages
"""
try:
    from ..core.graph import (
        KnowledgeGraph, NodeData, EdgeData,
        NodeType, EdgeType, EDGE_SEVERITY_WEIGHTS
    )
except ImportError:
    from core.graph import (
        KnowledgeGraph, NodeData, EdgeData,
        NodeType, EdgeType, EDGE_SEVERITY_WEIGHTS
    )


# ── Domain knowledge ──────────────────────────────────────────────────────────

NEURAL_LAM_HOOKS = [
    "training_step", "validation_step", "test_step",
    "configure_optimizers", "forward", "on_train_epoch_end",
    "on_validation_epoch_end", "predict_step", "setup",
]

# Known tensor shape contracts in neural-lam (function_name_suffix → {param: shape})
KNOWN_SHAPE_CONTRACTS = {
    "GraphLAM.forward":     {"x": "[B, N_grid, d_h]", "edge_index": "[2, E]", "edge_attr": "[E, d_e]"},
    "HiLAM.forward":        {"x": "[B, N_grid, d_h]", "mesh_x": "[B, N_mesh, d_h]"},
    "ARModel.forward":      {"x": "[B, pred_steps, N_grid, d_state]"},
    "WeatherDataset.__getitem__": {
        "return": "[init_states: [2, N_grid, d_state], target: [pred_steps, N_grid, d_state], "
                  "forcing: [pred_steps, N_grid, d_forcing], static: [N_grid, d_static]]"
    },
    "GraphLAM._mesh_graph_init": {"mesh_feat": "[N_mesh, d_mesh_static]", "m2g": "[2, E_m2g]"},
}

# Config keys that are critical (high-severity if renamed/removed)
CRITICAL_CONFIG_KEYS = [
    "hidden_dim", "num_processor_layers", "mesh_aggr", "processor_layers",
    "encoder_layers", "decoder_layers", "output_std", "loss",
    "lr", "val_steps_to_log", "n_example_pred",
    "num_input_steps", "num_pred_steps_train", "num_pred_steps_val_test",
]

# Known data pipeline stages in order
DATA_PIPELINE = [
    "datastore",          # raw data source
    "__getitem__",        # dataset item fetch
    "collate_fn",         # batch assembly
    "training_step",      # model consumes batch
    "forward",            # model forward pass
    "training_step",      # loss computation
    "on_train_epoch_end", # epoch-level logging
]

# Neural-lam specific module paths
NEURAL_LAM_MODULES = [
    "neural_lam.models.graph_lam",
    "neural_lam.models.hi_lam",
    "neural_lam.models.hi_lam_parallel",
    "neural_lam.models.ar_model",
    "neural_lam.weather_dataset",
    "neural_lam.create_graph",
    "neural_lam.graphs",
    "neural_lam.loss_weighting",
    "neural_lam.metrics",
    "neural_lam.train_model",
    "neural_lam.config",
]


# ── Preset application ────────────────────────────────────────────────────────

def apply(graph: KnowledgeGraph):
    """
    Apply neural-lam specific enrichment to an existing graph.
    Call this after indexing + base enrichment.
    """
    _inject_shape_contracts(graph)
    _inject_critical_config_keys(graph)
    _inject_pipeline_edges(graph)
    _flag_critical_nodes(graph)
    print("[codegraph:neural_lam preset] Applied domain enrichment.")


def _inject_shape_contracts(graph: KnowledgeGraph):
    """
    For each known shape contract, find the matching function node
    and either update its tensor_shapes or create a contract node.
    """
    for func_suffix, shapes in KNOWN_SHAPE_CONTRACTS.items():
        candidates = graph.resolve_node_id(func_suffix, explicit=True)
        for node_id in candidates:
            node_data = graph.get_node(node_id)
            if not node_data:
                continue
            # Merge known shapes into existing
            existing = node_data.get("tensor_shapes", {})
            merged = {**shapes, **existing}
            graph.g.nodes[node_id]["tensor_shapes"] = merged

            # Create/update tensor contract node
            contract_id = f"tensor_contract::{node_id}"
            if not graph.has_node(contract_id):
                graph.add_node(NodeData(
                    node_id=contract_id,
                    node_type=NodeType.TENSOR_CONTRACT,
                    name=f"shapes@{func_suffix}",
                    module="neural_lam.preset",
                    file_path="<preset>",
                    line_no=0,
                    tensor_shapes=merged,
                ))
            else:
                graph.g.nodes[contract_id]["tensor_shapes"] = merged

            graph.add_edge(node_id, contract_id, EdgeData(
                edge_type=EdgeType.PRODUCES_TENSOR,
                reason=f"[neural_lam preset] Known tensor contract for {func_suffix}",
                severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.PRODUCES_TENSOR],
            ))


def _inject_critical_config_keys(graph: KnowledgeGraph):
    """
    Ensure all critical config keys exist as nodes and are marked high-severity.
    """
    for key in CRITICAL_CONFIG_KEYS:
        cfg_id = f"config::{key}"
        if not graph.has_node(cfg_id):
            graph.add_node(NodeData(
                node_id=cfg_id,
                node_type=NodeType.CONFIG_KEY,
                name=key,
                module="neural_lam.config",
                file_path="<preset:config>",
                line_no=0,
            ))
        # Mark as critical in node data
        graph.g.nodes[cfg_id]["is_critical"] = True


def _inject_pipeline_edges(graph: KnowledgeGraph):
    """
    Inject explicit consumes_format edges along the known data pipeline.
    E.g. WeatherDataset.__getitem__ → training_step → forward
    """
    pairs = [
        ("WeatherDataset.__getitem__", "ARModel.training_step",
         "Dataset batch format feeds training_step — "
         "tensor shape changes (N_grid, d_state) cascade here."),
        ("WeatherDataset.__getitem__", "GraphLAM.training_step",
         "Dataset batch format feeds GraphLAM training_step."),
        ("ARModel.training_step", "ARModel.forward",
         "training_step calls forward — signature or tensor changes propagate."),
        ("GraphLAM.forward", "ARModel.training_step",
         "GraphLAM.forward result used in loss computation inside training_step."),
        ("create_graph", "GraphLAM.forward",
         "Graph structure (edge_index, edge_attr) produced by create_graph "
         "is loaded into GraphLAM.forward — topology changes break forward."),
    ]

    for src_suffix, dst_suffix, reason in pairs:
        srcs = graph.resolve_node_id(src_suffix, explicit=True)
        dsts = graph.resolve_node_id(dst_suffix, explicit=True)
        for src in srcs:
            for dst in dsts:
                graph.add_edge(src, dst, EdgeData(
                    edge_type=EdgeType.CONSUMES_FORMAT,
                    reason=f"[neural_lam preset] {reason}",
                    severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.CONSUMES_FORMAT],
                ))


def _flag_critical_nodes(graph: KnowledgeGraph):
    """
    Mark certain nodes as critical in their metadata — used by query engine
    to boost severity scores.
    """
    critical_suffixes = [
        "ARModel.forward", "GraphLAM.forward", "HiLAM.forward",
        "WeatherDataset.__getitem__", "ARModel.training_step",
        "ARModel.validation_step",
    ]
    for suffix in critical_suffixes:
        for node_id in graph.resolve_node_id(suffix, explicit=True):
            graph.g.nodes[node_id]["is_critical_preset"] = True
