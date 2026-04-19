"""Directed adjacency for a fully connected CHG (before interaction masking).

**paper-specified:** heterogeneous graph edges exist between agents; masking is applied later (Phase 4).

**engineering assumption:** edges are **directed** (*i* → *j*), all *i* ≠ *j*, ordered by increasing
``i``, then increasing ``j``. :func:`chgnet.graph.features.edge_geometric_attr` uses the same
``edge_index`` for Δp, Δv, cos θ, and cos φ.
"""

from __future__ import annotations

import torch


def directed_complete_edge_index(num_nodes: int, device: torch.device | None = None) -> torch.Tensor:
    """Return ``edge_index`` of shape ``(2, E)`` with ``E = N * (N - 1)``.

    Column ``k`` is ``(src, dst)`` with ``src`` in ``[0, N)``, ``dst`` in ``[0, N)``, ``src != dst``,
    sorted by ``src`` major, ``dst`` minor.
    """
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    rows: list[int] = []
    cols: list[int] = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    ei = torch.tensor([rows, cols], dtype=torch.long, device=device)
    return ei


def edge_count_complete_directed(num_nodes: int) -> int:
    """Number of directed edges in a simple clique without self-loops."""
    if num_nodes < 2:
        return 0
    return num_nodes * (num_nodes - 1)
