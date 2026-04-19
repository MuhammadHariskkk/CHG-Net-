"""CHG graph construction: node/edge features and directed clique adjacency (pre-mask)."""

from chgnet.graph.adjacency import directed_complete_edge_index, edge_count_complete_directed
from chgnet.graph.features import (
    cos_phi_heading_alignment,
    cos_theta_motion_direction,
    edge_geometric_attr,
    node_geometric_tensor,
    sanitize_agent_kinematics,
)
from chgnet.graph.graph_builder import (
    CHGGraph,
    build_chg_graph,
    build_chg_graph_from_batch,
    build_chg_graph_from_sample_dict,
)
from chgnet.graph.mask import (
    apply_direction_aware_mask,
    compute_direction_aware_mask,
    compute_direction_aware_mask_from_config,
    mask_edge_features,
)
from chgnet.graph.stgcn_block import STGCNBlock, edge_softmax_incoming, stgcn_hidden_dim_from_config

__all__ = [
    "apply_direction_aware_mask",
    "CHGGraph",
    "build_chg_graph",
    "build_chg_graph_from_batch",
    "build_chg_graph_from_sample_dict",
    "compute_direction_aware_mask",
    "compute_direction_aware_mask_from_config",
    "cos_phi_heading_alignment",
    "cos_theta_motion_direction",
    "directed_complete_edge_index",
    "edge_count_complete_directed",
    "edge_softmax_incoming",
    "edge_geometric_attr",
    "mask_edge_features",
    "node_geometric_tensor",
    "STGCNBlock",
    "stgcn_hidden_dim_from_config",
    "sanitize_agent_kinematics",
]
