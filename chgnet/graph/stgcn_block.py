"""Single ST-GCN block: spatial attention over masked CHG edges + temporal Conv1d.

**paper-specified (high level):** spatial message passing on the heterogeneous graph using edge
information, followed by temporal convolution along the observation trajectory.

**engineering assumption (PDF not in-repo):**
- Directed edges follow Phase 3 (*i* → *j*); node *j* aggregates **incoming** messages (all *i* with
  edges to *j*).
- **Attention** *a*_{i→j} is a **softmax over incoming neighbors** of *j* (separately for each batch
  row and time step), with logits produced by a linear layer on Φ(**e**_{ij}), where Φ is a learned
  edge MLP on masked edge attributes **ẽ**_{ij}. This yields normalized non-negative weights that sum
  to 1 over {(i,j) ∈ E | dst=j} for each (*b*, *t*, *j*).
- Message body: MLP([**h**_i ‖ Φ(**ẽ**_{ij})]) (concatenate then linear + ReLU).
- Residual **self** term: Linear(**h**_j) added before ReLU.
- **Temporal**: Conv1d with kernel *K* on the observation axis *T* (per node, shared across agents),
  ``padding=K//2`` so *T* matches ``data.obs_len`` (odd *K*).
- If ``edge_attr`` is 2D ``(E, D)``, it is **broadcast** across ``(B, T)``; optional 4D
  ``(B, T, E, D)`` supports time-varying edge features (**recommended extension** for strict
  geometry refresh each frame).

**Shapes**
    ``x``: ``(B, T, N, F_in)`` node inputs (e.g. per-timestep kinematics + optional static channels).
    ``edge_index``: ``(2, E)`` long, ``src=i``, ``dst=j``.
    ``edge_attr``: ``(E, D_edge)`` or ``(B, T, E, D_edge)`` (Phase 4 masked **ẽ**).
    ``node_mask``: optional ``(B, N)`` bool, True = valid agent; invalid nodes zeroed after update.

**Output:** ``(B, T, N, F_hidden)`` with ``F_hidden`` = ``hidden_dim`` (typically ``graph_hidden_dim``).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_softmax_incoming(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Softmax over edges that share the same ``dst`` (incoming to each node).

    **a_tij derivation:** for fixed batch-time row ``bt`` and destination ``j``,
    ``a_{i→j} = exp(logit_{ij}) / Σ_{i': (i',j) ∈ E} exp(logit_{i'j})``.

    Args:
        logits: ``(BT, E)`` unnormalized attention scores per edge.
        edge_index: ``(2, E)`` with ``edge_index[0]=src``, ``edge_index[1]=dst``.
        num_nodes: ``N`` (index dimension).

    Returns:
        ``(BT, E)`` nonnegative weights; each column group for a fixed ``(bt, j)`` sums to 1 over
        incoming edges to ``j`` (numerically stable per-destination max).
    """
    if logits.shape[1] == 0:
        return logits.new_zeros(logits.shape)
    BT, E = logits.shape
    device = logits.device
    dtype = logits.dtype
    edge_index = edge_index.to(device=device)
    dst = edge_index[1].long()

    max_val = torch.full((BT, num_nodes), float("-inf"), device=device, dtype=dtype)
    dst_exp = dst.view(1, E).expand(BT, E)
    max_val.scatter_reduce_(1, dst_exp, logits, reduce="amax", include_self=False)

    max_per_edge = max_val.gather(1, dst_exp)
    exp = torch.exp(logits - max_per_edge)

    sum_val = torch.zeros(BT, num_nodes, device=device, dtype=dtype)
    sum_val.scatter_add_(1, dst_exp, exp)

    denom = sum_val.gather(1, dst_exp).clamp_min(1e-12)
    return exp / denom


class STGCNBlock(nn.Module):
    """One spatial-temporal graph convolution block (single layer stack)."""

    def __init__(
        self,
        node_in_dim: int,
        edge_dim: int = 6,
        hidden_dim: int = 64,
        temporal_kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if temporal_kernel_size % 2 != 1:
            raise ValueError("temporal_kernel_size must be odd for same-length padding.")
        self.node_in_dim = node_in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.temporal_kernel_size = temporal_kernel_size

        self.edge_phi = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.attn_logit = nn.Linear(hidden_dim, 1, bias=False)
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_in_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.self_lin = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.temporal_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run spatial attention + aggregation, then temporal Conv1d.

        Args:
            x: ``(B, T, N, F_in)``.
            edge_index: ``(2, E)`` on same device as ``x``.
            edge_attr: ``(E, D_edge)`` or ``(B, T, E, D_edge)``.
            node_mask: ``(B, N)`` optional; invalid nodes zeroed after spatial + temporal step.
        """
        if x.dim() != 4:
            raise ValueError(f"x must be (B, T, N, F_in), got {tuple(x.shape)}")
        B, T, N, F_in = x.shape
        if F_in != self.node_in_dim:
            raise ValueError(f"x last dim {F_in} != node_in_dim {self.node_in_dim}")
        device = x.device
        dtype = x.dtype
        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_attr = edge_attr.to(device=device)
        src = edge_index[0]
        dst = edge_index[1]
        E = int(edge_index.shape[1])

        BT = B * T
        h = x.reshape(BT, N, F_in)

        def mask_bt(nm: torch.Tensor) -> torch.Tensor:
            """``(B, N)`` → ``(B*T, N, 1)`` repeating the same agents over time."""
            return nm.unsqueeze(1).expand(B, T, N).reshape(BT, N, 1)

        if E == 0:
            out_s = self.self_lin(h)
            out_s = F.relu(out_s)
            if node_mask is not None:
                m = mask_bt(node_mask).to(dtype=out_s.dtype)
                out_s = out_s * m
            out_s = self.dropout(out_s)
            y = out_s.reshape(B, T, N, self.hidden_dim)
            y = y.permute(0, 2, 3, 1).contiguous().view(B * N, self.hidden_dim, T)
            y = self.temporal_conv(y)
            y = y.view(B, N, self.hidden_dim, T).permute(0, 3, 1, 2).contiguous()
            if node_mask is not None:
                y = y * node_mask.unsqueeze(1).unsqueeze(-1).to(dtype=y.dtype)
            return y

        if edge_attr.dim() == 2:
            if edge_attr.shape[0] != E:
                raise ValueError("edge_attr rows must match edge_index E.")
            if edge_attr.shape[1] != self.edge_dim:
                raise ValueError(f"edge_attr dim 1 {edge_attr.shape[1]} != edge_dim {self.edge_dim}")
            enc_e = self.edge_phi(edge_attr.to(dtype=dtype))
            enc_e = enc_e.unsqueeze(0).expand(BT, -1, -1)
        elif edge_attr.dim() == 4:
            if edge_attr.shape[0] != B or edge_attr.shape[1] != T or edge_attr.shape[2] != E:
                raise ValueError("edge_attr (B,T,E,D) shape mismatch.")
            if edge_attr.shape[3] != self.edge_dim:
                raise ValueError(f"edge_attr dim 3 {edge_attr.shape[3]} != edge_dim {self.edge_dim}")
            ea = edge_attr.reshape(BT, E, self.edge_dim).to(dtype=dtype)
            enc_e = self.edge_phi(ea)
        else:
            raise ValueError("edge_attr must be (E, D) or (B, T, E, D).")

        h_src = h[:, src, :]
        logits = self.attn_logit(enc_e).squeeze(-1)
        attn = edge_softmax_incoming(logits, edge_index, N)

        msg_in = torch.cat([h_src, enc_e], dim=-1)
        msg = self.msg_mlp(msg_in)
        weighted = attn.unsqueeze(-1) * msg

        agg = torch.zeros(BT, N, self.hidden_dim, device=device, dtype=dtype)
        idx = dst.view(1, E, 1).expand(BT, E, self.hidden_dim)
        agg.scatter_add_(1, idx, weighted)

        out_s = agg + self.self_lin(h)
        out_s = F.relu(out_s)

        if node_mask is not None:
            m = mask_bt(node_mask).to(dtype=out_s.dtype)
            out_s = out_s * m

        out_s = self.dropout(out_s)
        y = out_s.reshape(B, T, N, self.hidden_dim)
        y = y.permute(0, 2, 3, 1).contiguous().view(B * N, self.hidden_dim, T)
        y = self.temporal_conv(y)
        y = y.view(B, N, self.hidden_dim, T).permute(0, 3, 1, 2).contiguous()

        if node_mask is not None:
            y = y * node_mask.unsqueeze(1).unsqueeze(-1).to(dtype=y.dtype)

        return y


def stgcn_hidden_dim_from_config(cfg: Mapping[str, Any] | None) -> int:
    """Resolve ST-GCN hidden size: ``model.stgcn.hidden_dim`` if set, else ``model.graph_hidden_dim``."""
    if cfg is None:
        return 64
    m = cfg.get("model", {})
    if not isinstance(m, dict):
        return 64
    st = m.get("stgcn", {})
    if isinstance(st, dict) and st.get("hidden_dim") is not None:
        return int(st["hidden_dim"])
    return int(m.get("graph_hidden_dim", 64))
