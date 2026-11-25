"""GNN and baseline model architectures (Section 17).

Three GNNs (GCN, GraphSAGE, GATv2) plus a feature-only MLP baseline and a GCN encoder for
the unsupervised Graph-Autoencoder (Approach B). All output raw logits (use BCEWithLogitsLoss).

Weighted-edge support (verified per layer, Section 17):
- ``GCNConv``   : supports ``edge_weight`` (scalar per edge) -> we pass STRING confidence.
- ``SAGEConv``  : does NOT take edge weights -> topology only (documented limitation).
- ``GATv2Conv`` : learns attention; edge weights not used as scalars (can use ``edge_dim`` for
                  edge *features*, out of scope here) -> attention != causal evidence.

Oversmoothing/oversquashing: stacking many message-passing layers makes node representations
converge (oversmoothing) and bottlenecks long-range signal (oversquashing). We default to 2-3
layers with dropout + normalization + optional residual connections to mitigate both.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

logger = logging.getLogger(__name__)

CONV_TYPES = ("gcn", "sage", "gatv2")
_SUPPORTS_EDGE_WEIGHT = {"gcn": True, "sage": False, "gatv2": False}


def _make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "batchnorm":
        return nn.BatchNorm1d(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind in ("none", None):
        return nn.Identity()
    raise ValueError(f"Unknown norm {kind!r}.")


class GNN(nn.Module):
    """Configurable message-passing network (GCN / GraphSAGE / GATv2) + linear classifier."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        *,
        num_layers: int = 2,
        dropout: float = 0.4,
        conv_type: str = "gcn",
        norm: str = "batchnorm",
        residual: bool = False,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if conv_type not in CONV_TYPES:
            raise ValueError(f"conv_type must be one of {CONV_TYPES}, got {conv_type!r}.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.conv_type = conv_type
        self.dropout = dropout
        self.residual = residual
        self.supports_edge_weight = _SUPPORTS_EDGE_WEIGHT[conv_type]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            cin = in_channels if i == 0 else hidden_channels
            self.convs.append(self._make_conv(conv_type, cin, hidden_channels, heads))
            self.norms.append(_make_norm(norm, hidden_channels))
        self.head = nn.Linear(hidden_channels, out_channels)

    @staticmethod
    def _make_conv(conv_type: str, cin: int, hidden: int, heads: int) -> nn.Module:
        if conv_type == "gcn":
            return GCNConv(cin, hidden, add_self_loops=True, normalize=True)
        if conv_type == "sage":
            return SAGEConv(cin, hidden)
        # gatv2: keep output dim == hidden by splitting across heads then concatenating
        if hidden % heads != 0:
            raise ValueError(f"hidden_channels ({hidden}) must be divisible by heads ({heads}).")
        return GATv2Conv(cin, hidden // heads, heads=heads, concat=True, dropout=0.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_in = h
            if self.conv_type == "gcn" and edge_weight is not None:
                h = conv(h, edge_index, edge_weight)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            if i < len(self.convs) - 1:
                h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and h.shape == h_in.shape:
                h = h + h_in
        return self.head(h)

    @torch.no_grad()
    def node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return pre-classifier hidden representations (for explainability/inspection)."""
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.conv_type == "gcn" and edge_weight is not None:
                h = conv(h, edge_index, edge_weight)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            if i < len(self.convs) - 1:
                h = F.gelu(h)
        return h


class MLP(nn.Module):
    """Feature-only baseline (no message passing) — isolates the graph's contribution."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 1,
                 *, num_layers: int = 2, dropout: float = 0.4, norm: str = "batchnorm") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [in_channels] + [hidden_channels] * (num_layers - 1)
        for cin, cout in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(cin, cout), _make_norm(norm, cout), nn.GELU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(dims[-1], out_channels)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: D401
        return self.head(self.body(x))


class GCNEncoder(nn.Module):
    """Encoder for the unsupervised Graph-Autoencoder (Approach B).

    Use with ``torch_geometric.nn.GAE(GCNEncoder(...))``; the inner-product decoder
    reconstructs the adjacency. Node embeddings then feed network propagation / module
    detection without any disease labels.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h = F.gelu(self.conv1(x, edge_index, edge_weight))
        return self.conv2(h, edge_index, edge_weight)


def build_model(name: str, in_channels: int, cfg_model: dict, *, out_channels: int = 1) -> nn.Module:
    """Factory: name in {'gcn','sage','gatv2','mlp'} using the ``model`` config section."""
    common = dict(
        in_channels=in_channels,
        hidden_channels=int(cfg_model.get("hidden_channels", 128)),
        out_channels=out_channels,
        num_layers=int(cfg_model.get("num_layers", 2)),
        dropout=float(cfg_model.get("dropout", 0.4)),
        norm=cfg_model.get("norm", "batchnorm"),
    )
    if name == "mlp":
        return MLP(**common)
    name = {"graphsage": "sage"}.get(name, name)  # config uses 'graphsage'; conv key is 'sage'
    return GNN(**common, conv_type=name, residual=bool(cfg_model.get("residual", False)),
               heads=int(cfg_model.get("heads", 4)))
