"""Full CHG-Net: class embeddings, masked CHG, ST-GCN, Micro-TCN, GMM trajectory decoder.

**paper-specified (high level):** heterogeneous class-aware graph, direction-aware edge masking,
ST-GCN-style spatial attention on masked edges, dilated temporal conv (Micro-TCN), mixture of
Gaussians over future positions.

**engineering assumption:**
- Collated batches use a fixed ``N_max``; graphs are built on all ``N_max`` slots so
  ``edge_index`` is shared across the batch row dimension (see ``build_chg_graph_from_batch(..., num_nodes=...)``).
- Node inputs concatenate per-timestep kinematics ``(x, y, v_x, v_y, heading)`` with a **time-constant**
  class embedding per agent (invalid / padded classes are zeroed).
- Edge geometry and direction-aware masks use the **last observation timestep** (Phase 3–4 graph) and
  are **broadcast across all ``T``** for ST-GCN ``edge_attr`` (time-varying CHG per frame is not
  implemented here; see :class:`~chgnet.graph.stgcn_block.STGCNBlock` docstring).
- ``CHGNetOutput.node_mask`` matches :class:`~chgnet.graph.graph_builder.CHGGraph` ``node_valid``
  (finite kinematics, ``class_idx >= 0``, ``agent_id >= 0``, and last-step ``obs_valid``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from chgnet.datasets.label_mapping import build_label_mapper_from_config
from chgnet.graph.adjacency import directed_complete_edge_index
from chgnet.graph.graph_builder import build_chg_graph_from_batch
from chgnet.graph.mask import apply_direction_aware_mask
from chgnet.graph.stgcn_block import STGCNBlock, stgcn_hidden_dim_from_config
from chgnet.models.gmm_decoder import GMMDecoderOutput, gmm_decoder_from_config
from chgnet.models.micro_tcn import micro_tcn_from_config


# Kinematics channels from Phase 2 tensors (no rel_disp in the ST-GCN node vector).
_NODE_GEOM_DIM = 5


@dataclass
class CHGNetOutput:
    """Forward bundle for loss (Phase 8), metrics, and export (Phase 9)."""

    gmm: GMMDecoderOutput
    node_mask: torch.Tensor
    stgcn_hidden: torch.Tensor
    micro_hidden: torch.Tensor

    @property
    def deterministic_trajectory(self) -> torch.Tensor:
        """``(B, N, pred_len, 2)`` — highest-mixture-weight mode (paper default)."""
        return self.gmm.deterministic_trajectory

    @property
    def multimodal_means(self) -> torch.Tensor:
        """``(B, N, K, L, 2)`` — all GMM component means (export-friendly)."""
        return self.gmm.means

    @property
    def multimodal_std(self) -> torch.Tensor:
        return self.gmm.std

    @property
    def mix_logits(self) -> torch.Tensor:
        return self.gmm.mix_logits


class CHGNet(nn.Module):
    """End-to-end CHG-Net for batched SDD-style windows."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.get("model", {})
        if not isinstance(m, dict):
            m = {}
        d = cfg.get("data", {})
        if not isinstance(d, dict):
            d = {}

        self._obs_len = int(d.get("obs_len", 8))
        self._pred_len = int(d.get("pred_len", 12))

        class_emb_dim = int(m.get("class_embedding_dim", 16))
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self._num_classes = int(num_classes)
        self.class_embed = nn.Embedding(self._num_classes, class_emb_dim)

        node_in_dim = _NODE_GEOM_DIM + class_emb_dim
        hidden_dim = stgcn_hidden_dim_from_config(cfg)

        stg = m.get("stgcn", {})
        if not isinstance(stg, dict):
            stg = {}
        tk = int(stg.get("temporal_kernel_size", 3))
        st_dropout = float(stg.get("dropout", m.get("dropout", 0.0)))

        self.stgcn = STGCNBlock(
            node_in_dim=node_in_dim,
            edge_dim=6,
            hidden_dim=hidden_dim,
            temporal_kernel_size=tk,
            dropout=st_dropout,
        )
        self.micro = micro_tcn_from_config(cfg, in_channels=hidden_dim)
        self.gmm_head = gmm_decoder_from_config(cfg, input_dim=hidden_dim, pred_len=self._pred_len)

    def forward(self, batch: Mapping[str, Any]) -> CHGNetOutput:
        device = next(self.parameters()).device
        param_dtype = next(self.parameters()).dtype
        if batch["obs_xy"].device != device:
            raise ValueError(
                "CHGNet expects batch tensors on the same device as model parameters "
                f"(batch on {batch['obs_xy'].device}, model on {device})."
            )

        obs_xy = batch["obs_xy"].to(device=device, dtype=param_dtype)
        obs_vel = batch["obs_vel"].to(device=device, dtype=param_dtype)
        obs_heading = batch["obs_heading"].to(device=device, dtype=param_dtype)
        obs_valid = batch["obs_valid"].to(device=device)
        class_idx = batch["class_idx"].to(device=device)
        agent_id = batch["agent_id"].to(device=device)
        num_agents = batch["num_agents"].to(device=device)

        batch_graph: dict[str, Any] = dict(batch)
        batch_graph.update(
            {
                "obs_xy": obs_xy,
                "obs_vel": obs_vel,
                "obs_heading": obs_heading,
                "obs_valid": obs_valid,
                "class_idx": class_idx,
                "agent_id": agent_id,
                "num_agents": num_agents,
            }
        )

        B, T, N, _ = obs_xy.shape
        if T != self._obs_len:
            raise ValueError(f"obs_len mismatch: batch T={T}, config data.obs_len={self._obs_len}")

        cls = class_idx.long()
        valid_cls = cls >= 0
        cls_safe = cls.clamp(min=0)
        ce = self.class_embed(cls_safe)
        ce = ce * valid_cls.unsqueeze(-1).to(dtype=ce.dtype)

        base = torch.cat(
            [obs_xy, obs_vel, obs_heading.unsqueeze(-1)],
            dim=-1,
        )
        if base.shape[-1] != _NODE_GEOM_DIM:
            raise ValueError(f"expected {_NODE_GEOM_DIM} kinematic channels, got {base.shape[-1]}")
        ce_t = ce.unsqueeze(1).expand(-1, T, -1, -1)
        x = torch.cat([base, ce_t], dim=-1)

        edge_index = directed_complete_edge_index(N, device=device)
        edge_attr_list: list[torch.Tensor] = []
        node_mask_list: list[torch.Tensor] = []
        for b in range(B):
            g = build_chg_graph_from_batch(batch_graph, b, cfg=self.cfg, num_nodes=N)
            gm, _ = apply_direction_aware_mask(g, self.cfg)
            edge_attr_list.append(gm.edge_attr.to(device=device, dtype=param_dtype))
            node_mask_list.append(g.node_valid.to(device=device))

        edge_attr_b = torch.stack(edge_attr_list, dim=0)
        node_mask = torch.stack(node_mask_list, dim=0)
        edge_attr_bt = edge_attr_b.unsqueeze(1).expand(-1, T, -1, -1).contiguous()

        h_st = self.stgcn(x, edge_index, edge_attr_bt, node_mask=node_mask)

        hd = self.stgcn.hidden_dim
        h_perm = h_st.permute(0, 2, 3, 1).contiguous().view(B * N, hd, T)
        h_micro = self.micro(h_perm)
        h_ctx = h_micro[:, :, -1].view(B, N, hd)

        gmm_out = self.gmm_head(h_ctx)
        return CHGNetOutput(
            gmm=gmm_out,
            node_mask=node_mask,
            stgcn_hidden=h_st,
            micro_hidden=h_micro.view(B, N, hd, T),
        )


def chg_net_from_config(cfg: Mapping[str, Any]) -> CHGNet:
    """Build :class:`CHGNet` with ``num_classes = len(label vocabulary)`` from ``cfg['labels']``."""
    mapper = build_label_mapper_from_config(cfg)
    return CHGNet(cfg, num_classes=len(mapper.vocabulary))
