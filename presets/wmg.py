"""
weather-model-graphs (wmg) preset.

Injects domain knowledge that pure AST cannot infer:

  1. Format contract nodes — the .pt files on disk that wmg writes and
     neural-lam reads. These are the invisible coupling between the two
     codebases: nowhere in the AST of either repo does a Python call
     connect to_pyg() to load_graph(). The contracts exist only as
     filename conventions, tensor wrapping rules, and node-index ordering
     assumptions. This preset makes them first-class graph nodes so the
     consequence engine can surface cross-repo breakage.

  2. Lossiness edge — from_networkx() in PyG discards networkx node labels
     and re-indexes to [0, N). This is an irreversible information loss:
     a round-trip through to_pyg() → from_pyg() cannot reconstruct the
     original graph. Any function in wmg that relies on globally-unique
     node labels is silently broken after this conversion. We inject an
     explicit edge to surface this.

  3. wmg→neural-lam pipeline edges — the four concrete mismatches from
     the prajwal-tech07 audit (filename conventions, List[Tensor] wrapping,
     hierarchical edge naming, aggregated mesh features) are injected as
     explicit consumes_format edges so the consequence engine flags them
     when to_pyg() or load_graph() changes.

  4. Critical wmg functions — to_pyg(), from_networkx(), create_*_graph()
     archetypes, split_graph_by_edge_attribute(), sort_nodes_in_graph() —
     marked critical because changes break the format contract.

Usage:
    python server.py --path ./repos --preset wmg_neural_lam
    (where ./repos/ contains both neural-lam/ and weather-model-graphs/)
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


# ── Format contracts on disk ──────────────────────────────────────────────────
#
# These are the files that wmg writes and neural-lam reads.
# Each entry: (contract_id, filename, description, produced_by, consumed_by)
#
# produced_by / consumed_by are function name suffixes used to resolve
# nodes in whichever repo has been indexed.
#
# The four mismatches from the prajwal-tech07 audit are modelled here:
#   1. {name}_node_features.pt  ≠  mesh_features.pt       (filename)
#   2. Tensor(2,E)              ≠  List[Tensor]            (wrapping)
#   3. m2m_up_*.pt              ≠  mesh_up_*.pt            (hierarchical naming)
#   4. per-component features   ≠  aggregated mesh_features.pt (aggregation)

_FORMAT_CONTRACTS = [
    # ── g2m ──────────────────────────────────────────────────────────────────
    {
        "id":       "format_contract::g2m_edge_index.pt",
        "filename": "g2m_edge_index.pt",
        "desc":     "Grid-to-mesh edge indices. wmg writes via to_pyg(g2m). "
                    "neural-lam reads in load_graph() → g2m_edge_index.",
        "produced_by": ["to_pyg", "to_neural_lam"],
        "consumed_by": ["load_graph"],
        "mismatch_risk": "low",   # naming is already correct
    },
    {
        "id":       "format_contract::g2m_edge_features.pt",
        "filename": "g2m_edge_features.pt",
        "desc":     "Grid-to-mesh edge features (len, vdiff concatenated). "
                    "Column ordering must match what load_graph() expects.",
        "produced_by": ["to_pyg", "to_neural_lam", "_concat_pyg_features"],
        "consumed_by": ["load_graph"],
        "mismatch_risk": "medium",  # column order is implicit
    },
    # ── m2g ──────────────────────────────────────────────────────────────────
    {
        "id":       "format_contract::m2g_edge_index.pt",
        "filename": "m2g_edge_index.pt",
        "desc":     "Mesh-to-grid edge indices for decode step.",
        "produced_by": ["to_pyg", "to_neural_lam"],
        "consumed_by": ["load_graph"],
        "mismatch_risk": "low",
    },
    {
        "id":       "format_contract::m2g_edge_features.pt",
        "filename": "m2g_edge_features.pt",
        "desc":     "Mesh-to-grid edge features.",
        "produced_by": ["to_pyg", "to_neural_lam", "_concat_pyg_features"],
        "consumed_by": ["load_graph"],
        "mismatch_risk": "medium",
    },
    # ── m2m flat ──────────────────────────────────────────────────────────────
    # MISMATCH 2: wmg saves Tensor(2,E), neural-lam BufferList expects List[Tensor]
    {
        "id":       "format_contract::m2m_edge_index.pt",
        "filename": "m2m_edge_index.pt",
        "desc":     "Mesh-to-mesh edge indices. "
                    "MISMATCH RISK: wmg to_pyg() saves a bare Tensor(2,E) when "
                    "list_from_attribute=None. neural-lam load_graph() always "
                    "passes to BufferList() which calls len() — expects List[Tensor]. "
                    "Breaks with TypeError on load.",
        "produced_by": ["to_pyg", "to_neural_lam"],
        "consumed_by": ["load_graph"],
        "mismatch_risk": "high",   # audit item #2
    },
    # MISMATCH 1+4: wmg writes m2m_node_features.pt, neural-lam expects mesh_features.pt
    # as a List[(N_mesh[l], d)] across all levels
    {
        "id":       "format_contract::mesh_features.pt",
        "filename": "mesh_features.pt",
        "desc":     "Aggregated mesh node features across all levels — List[(N_mesh[l], d)]. "
                    "MISMATCH RISK: wmg to_pyg() writes {name}_node_features.pt per call "
                    "(e.g. m2m_node_features.pt). neural-lam hardcodes mesh_features.pt. "
                    "Breaks with FileNotFoundError.",
        "produced_by": ["to_pyg", "to_neural_lam"],
        "consumed_by": ["load_graph"],
        "mismatch_risk": "high",   # audit items #1 and #4
    },
    # ── m2m hierarchical ──────────────────────────────────────────────────────
    # MISMATCH 3: wmg saves m2m_up_*.pt, neural-lam expects mesh_up_*.pt
    {
        "id":       "format_contract::mesh_up_edge_index.pt",
        "filename": "mesh_up_edge_index.pt",
        "desc":     "Hierarchical upward mesh edges — List[Tensor] per level. "
                    "MISMATCH RISK: wmg test flow saves m2m_up_edge_index.pt. "
                    "neural-lam expects mesh_up_edge_index.pt. FileNotFoundError.",
        "produced_by": ["to_pyg", "to_neural_lam", "split_graph_by_edge_attribute"],
        "consumed_by": ["load_graph"],
        "mismatch_risk": "high",   # audit item #3
    },
    {
        "id":       "format_contract::mesh_down_edge_index.pt",
        "filename": "mesh_down_edge_index.pt",
        "desc":     "Hierarchical downward mesh edges — List[Tensor] per level. "
                    "Same naming mismatch as mesh_up: m2m_down_*.pt vs mesh_down_*.pt.",
        "produced_by": ["to_pyg", "to_neural_lam", "split_graph_by_edge_attribute"],
        "consumed_by": ["load_graph"],
        "mismatch_risk": "high",   # audit item #3
    },
    {
        "id":       "format_contract::m2m_edge_features.pt",
        "filename": "m2m_edge_features.pt",
        "desc":     "Same-level m2m edge features — List[Tensor] per level.",
        "produced_by": ["to_pyg", "to_neural_lam"],
        "consumed_by": ["load_graph"],
        "mismatch_risk": "medium",
    },
    # ── decode mask ───────────────────────────────────────────────────────────
    {
        "id":       "format_contract::decode_mask.npz",
        "filename": "decode_mask.npz",
        "desc":     "Mask defining which grid nodes are decoded to. "
                    "In LAM settings separates interior from boundary. "
                    "Currently a side-channel outside the graph structure — "
                    "grid/in+grid/out split would eliminate this.",
        "produced_by": ["export_decode_mask", "export"],
        "consumed_by": ["load_decode_mask", "load_graph", "ARModel.__init__"],
        "mismatch_risk": "medium",
    },
]


# ── Node-label lossiness ──────────────────────────────────────────────────────
#
# from_networkx() in PyG re-indexes all nodes to [0, N).
# This destroys wmg's globally-unique node labels.
# Consequence: after to_pyg(), you cannot recover the original node identity.
# In a LAM g2m graph, grid node 0 may be a boundary node that does NOT appear
# in m2g. If the merged graph is converted once, the correct per-component
# indices are lost — this is why return_components=True is necessary.

_LOSSINESS_NOTE = (
    "PyG from_networkx() re-indexes nodes to [0, N), discarding wmg's globally-unique "
    "node labels. Round-trip to_pyg() → from_pyg() cannot reconstruct the original graph. "
    "In LAM settings: boundary grid nodes (present in g2m, absent from m2g) get "
    "wrong indices if the full graph is converted at once. "
    "return_components=True is required to convert g2m, m2m, m2g separately."
)


# ── wmg shape contracts ───────────────────────────────────────────────────────

WMG_SHAPE_CONTRACTS = {
    # Edge features: concatenation of len (scalar) and vdiff (3D vector) → d=4
    "to_pyg":        {"edge_features": "[E, 4]  # len:1 + vdiff:3 per edge"},
    "to_neural_lam": {"g2m_edge_index": "[2, E_g2m]", "m2m_edge_index": "List[[2, E_m2m[l]]]",
                      "mesh_features": "List[[N_mesh[l], d_mesh]]"},
    # from create_keisler_graph — single mesh level
    "create_keisler_graph": {"return": "nx.DiGraph (g2m+m2m+m2g combined or components)"},
    "create_oskarsson_hierarchical_graph": {
        "return": "nx.DiGraph with edge attr 'direction'∈{up,down,same} and 'level'∈int"
    },
}


# ── Critical wmg functions ────────────────────────────────────────────────────
#
# Changes to any of these functions risk breaking the format contract.

WMG_CRITICAL_FUNCTIONS = [
    "to_pyg",                        # primary serialisation — any change cascades to neural-lam
    "to_neural_lam",                 # P0 shim — the bridge function
    "_concat_pyg_features",          # determines edge feature column ordering
    "from_networkx",                 # PyG conversion — causes node-label lossiness
    "split_graph_by_edge_attribute", # determines hierarchical m2m splitting
    "sort_nodes_in_graph",           # determines node ordering before conversion
    "create_keisler_graph",          # creates flat graph — format contract consumer
    "create_oskarsson_hierarchical_graph",  # creates hierarchical graph
    "load_graph",                    # neural-lam side — the other end of every contract
    "export_decode_mask",            # decode mask contract
]


# ── apply() ──────────────────────────────────────────────────────────────────

def apply(graph: KnowledgeGraph):
    """
    Entry point. Called after indexing completes.
    Injects all wmg domain knowledge into the knowledge graph.
    """
    _inject_format_contract_nodes(graph)
    _inject_cross_repo_edges(graph)
    _inject_lossiness_edges(graph)
    _inject_shape_contracts(graph)
    _flag_critical_nodes(graph)
    print("[wmg preset] Format contract nodes and cross-repo edges injected.")


def _inject_format_contract_nodes(graph: KnowledgeGraph):
    """
    Create DATA_FORMAT nodes representing each .pt / .npz file on disk.
    These are the invisible cross-repo coupling made visible.
    """
    for contract in _FORMAT_CONTRACTS:
        node_id = contract["id"]
        if not graph.has_node(node_id):
            graph.add_node(NodeData(
                node_id=node_id,
                node_type=NodeType.DATA_FORMAT,
                name=contract["filename"],
                module="format_contract",
                file_path="<format_contract>",
                line_no=0,
                docstring=contract["desc"],
            ))
        # Tag mismatch risk level so consequence engine can weight it
        graph.g.nodes[node_id]["mismatch_risk"] = contract["mismatch_risk"]
        graph.g.nodes[node_id]["is_cross_repo"] = True


def _inject_cross_repo_edges(graph: KnowledgeGraph):
    """
    Inject produces_tensor_for / consumes_format edges from wmg write functions
    to format contract nodes, and from format contract nodes to neural-lam
    load functions.

    This makes the cross-codebase format dependency explicit:
      to_pyg() ──produces──► format_contract::mesh_features.pt ──consumes──► load_graph()

    Without these edges, changing to_pyg() in wmg shows zero consequence in
    neural-lam's AST — the coupling only exists through the filename on disk.
    """
    for contract in _FORMAT_CONTRACTS:
        node_id = contract["id"]

        # wmg producer functions → format contract node
        for producer_suffix in contract["produced_by"]:
            candidates = graph.resolve_node_id(producer_suffix, explicit=True)
            for src in candidates:
                src_data = graph.get_node(src) or {}
                # Only wire functions from wmg (not neural-lam's own references)
                fp = src_data.get("file_path", "")
                if "neural_lam" in fp or "neural-lam" in fp:
                    continue
                graph.add_edge(src, node_id, EdgeData(
                    edge_type=EdgeType.PRODUCES_TENSOR,
                    reason=(
                        f"[wmg preset] {producer_suffix} writes {contract['filename']} "
                        f"to disk (mismatch_risk={contract['mismatch_risk']}). "
                        f"{contract['desc'][:120]}"
                    ),
                    severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.PRODUCES_TENSOR],
                ))

        # format contract node → neural-lam consumer functions
        for consumer_suffix in contract["consumed_by"]:
            candidates = graph.resolve_node_id(consumer_suffix, explicit=True)
            for dst in candidates:
                dst_data = graph.get_node(dst) or {}
                fp = dst_data.get("file_path", "")
                # Prefer neural-lam side for consumers
                graph.add_edge(node_id, dst, EdgeData(
                    edge_type=EdgeType.CONSUMES_FORMAT,
                    reason=(
                        f"[wmg preset] {consumer_suffix} reads {contract['filename']} "
                        f"from disk (mismatch_risk={contract['mismatch_risk']}). "
                        + ("⚠ KNOWN MISMATCH — see prajwal-tech07 audit in #339."
                           if contract["mismatch_risk"] == "high" else "")
                    ),
                    severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.CONSUMES_FORMAT],
                ))


def _inject_lossiness_edges(graph: KnowledgeGraph):
    """
    Inject explicit edges representing the node-label lossiness from
    PyG's from_networkx(). Any function that calls from_networkx() on
    a merged graph loses wmg's globally-unique node indexing.

    This surfaces as a consequence when any function that depends on
    correct node-label ordering is downstream of from_networkx().
    """
    # from_networkx is the lossiness source
    from_nx_candidates = graph.resolve_node_id("from_networkx", explicit=True)

    # Functions whose correctness depends on correct node ordering
    ordering_sensitive = [
        "to_pyg",
        "to_neural_lam",
        "sort_nodes_in_graph",
        "load_graph",         # neural-lam end — wrong indices cause silent errors
        "ARModel.__init__",   # builds BufferLists indexed by node
        "GraphLAM.forward",   # edge_index tensor references node positions
    ]

    for from_nx in from_nx_candidates:
        for sensitive_suffix in ordering_sensitive:
            candidates = graph.resolve_node_id(sensitive_suffix, explicit=True)
            for dst in candidates:
                if dst == from_nx:
                    continue
                graph.add_edge(from_nx, dst, EdgeData(
                    edge_type=EdgeType.CONSUMES_FORMAT,
                    reason=(
                        f"[wmg preset] from_networkx() re-indexes nodes to [0,N), "
                        f"discarding wmg node labels. {sensitive_suffix} depends on "
                        f"correct node ordering — silently wrong results if a merged "
                        f"g2m+m2m+m2g graph is converted without return_components=True. "
                        + _LOSSINESS_NOTE[:100]
                    ),
                    severity_weight=EDGE_SEVERITY_WEIGHTS[EdgeType.CONSUMES_FORMAT],
                ))

    # Also tag from_networkx nodes themselves so they appear in consequence output
    for from_nx in from_nx_candidates:
        graph.g.nodes[from_nx]["is_lossiness_boundary"] = True
        graph.g.nodes[from_nx]["docstring"] = (
            (graph.g.nodes[from_nx].get("docstring") or "") +
            f"\n⚠ WMG LOSSINESS: {_LOSSINESS_NOTE}"
        )


def _inject_shape_contracts(graph: KnowledgeGraph):
    """
    Inject known tensor shape contracts for wmg functions.
    """
    for func_suffix, shapes in WMG_SHAPE_CONTRACTS.items():
        candidates = graph.resolve_node_id(func_suffix, explicit=True)
        for node_id in candidates:
            node_data = graph.get_node(node_id)
            if not node_data:
                continue
            existing = node_data.get("tensor_shapes", {})
            merged = {**shapes, **existing}
            graph.g.nodes[node_id]["tensor_shapes"] = merged


def _flag_critical_nodes(graph: KnowledgeGraph):
    """
    Mark critical wmg functions — changes to these risk breaking the
    format contract and causing silent or loud failures in neural-lam.
    """
    for func_suffix in WMG_CRITICAL_FUNCTIONS:
        for node_id in graph.resolve_node_id(func_suffix, explicit=True):
            graph.g.nodes[node_id]["is_critical_preset"] = True
            # Boost severity so consequence engine surfaces these first
            graph.g.nodes[node_id]["preset_severity_boost"] = 2
