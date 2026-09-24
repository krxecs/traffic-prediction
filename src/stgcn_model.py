"""Ablatable STGCN encoder variants used by the METR-LA experiments.

Defaults intentionally retain A0: a physical graph and single temporal GLUs.
"""

import math

import torch
from torch import nn
from torch_geometric.nn import ChebConv
from torch_geometric.utils import coalesce, get_laplacian


class TemporalConvLayer(nn.Module):
    """A0 GLU temporal convolution. It shortens time by ``kt - 1``."""

    def __init__(self, c_in, c_out, kt=3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kt, 1))

    def forward(self, x):
        linear, gate = self.conv(x).chunk(2, dim=1)
        return linear * torch.sigmoid(gate)


class MultiScaleTemporalConvLayer(nn.Module):
    """Causal aligned GLU branches that retain the A0 output length."""

    def __init__(self, c_in, c_out, kt=3, dilations=(1, 2, 4)):
        super().__init__()
        if kt != 3:
            raise ValueError("MultiScaleTemporalConvLayer currently supports kt=3.")
        self.dilations = tuple(dilations)
        self.branches = nn.ModuleList(
            nn.Conv2d(c_in, 2 * c_out, kernel_size=(kt, 1), dilation=(d, 1))
            for d in self.dilations
        )
        self.fuse = nn.Conv2d(len(self.dilations) * c_out, c_out, kernel_size=1)

    def forward(self, x):
        outputs = []
        for dilation, conv in zip(self.dilations, self.branches):
            padded = torch.nn.functional.pad(x, (0, 0, 2 * (dilation - 1), 0))
            linear, gate = conv(padded).chunk(2, dim=1)
            outputs.append(linear * torch.sigmoid(gate))
        expected_steps = x.shape[2] - 2
        if any(branch.shape[2] != expected_steps for branch in outputs):
            raise RuntimeError("Multi-scale temporal branches lost time alignment.")
        return self.fuse(torch.cat(outputs, dim=1))


class AdaptiveAdjacency(nn.Module):
    """Static sparse learned topology mixed differentiably with physical edges."""

    def __init__(self, num_nodes, physical_edge_index, physical_edge_weight,
                 embed_dim=16, top_k=16, alpha=0.8, edge_dropout=0.05):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("physical_graph_alpha must be in [0, 1].")
        if not 0 < top_k < num_nodes:
            raise ValueError("adaptive_top_k must be between 1 and num_nodes - 1.")
        self.num_nodes, self.embed_dim = int(num_nodes), int(embed_dim)
        self.top_k, self.alpha, self.edge_dropout = int(top_k), float(alpha), float(edge_dropout)
        self.node_src = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.node_dst = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_src)
        nn.init.xavier_uniform_(self.node_dst)
        self.register_buffer("physical_edge_index", physical_edge_index.long())
        self.register_buffer("physical_edge_weight", physical_edge_weight.float())

    def adaptive_edges(self):
        scores = torch.relu(self.node_src @ self.node_dst.T / math.sqrt(self.embed_dim))
        scores = scores.masked_fill(torch.eye(self.num_nodes, dtype=torch.bool, device=scores.device), float("-inf"))
        values, neighbors = scores.topk(self.top_k, dim=1)
        weights = torch.softmax(values, dim=1)
        if self.training and self.edge_dropout:
            keep = (torch.rand_like(weights) >= self.edge_dropout).to(weights.dtype)
            weights = weights * keep / (1.0 - self.edge_dropout)
        source = torch.arange(self.num_nodes, device=scores.device).unsqueeze(1).expand_as(neighbors)
        edge_index = torch.stack((source.reshape(-1), neighbors.reshape(-1)))
        edge_weight = weights.reshape(-1)
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        edge_weight = torch.cat((edge_weight, edge_weight), dim=0)
        return coalesce(edge_index, edge_weight, self.num_nodes, reduce="mean")

    def mixed_edges(self):
        adaptive_index, adaptive_weight = self.adaptive_edges()
        edge_index = torch.cat((self.physical_edge_index, adaptive_index), dim=1)
        edge_weight = torch.cat((self.alpha * self.physical_edge_weight, (1.0 - self.alpha) * adaptive_weight))
        return coalesce(edge_index, edge_weight, self.num_nodes, reduce="sum")

    @torch.no_grad()
    def diagnostics(self):
        adaptive_index, adaptive_weight = self.adaptive_edges()
        physical_ids = self.physical_edge_index[0] * self.num_nodes + self.physical_edge_index[1]
        adaptive_ids = adaptive_index[0] * self.num_nodes + adaptive_index[1]
        novel = ~torch.isin(adaptive_ids, physical_ids)
        return {
            "adaptive_embed_dim": self.embed_dim, "adaptive_top_k": self.top_k,
            "physical_graph_alpha": self.alpha, "adaptive_edges": int(adaptive_weight.numel()),
            "adaptive_nonphysical_fraction": float(novel.float().mean().cpu()),
            "adaptive_weight_mean": float(adaptive_weight.mean().cpu()),
            "adaptive_weight_max": float(adaptive_weight.max().cpu()),
        }


class PyGChebGraphConv(nn.Module):
    """Chebyshev convolution with static or differentiable graph weights."""

    def __init__(self, c_in, c_out, ks, edge_index, edge_weight, adaptive_graph=None):
        super().__init__()
        self.conv = ChebConv(c_in, c_out, K=ks, normalization="sym")
        self.num_nodes = int(edge_index.max()) + 1
        self.adaptive_graph = adaptive_graph
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        if adaptive_graph is None:
            cached_edge_index, cached_norm = self._laplacian(edge_index, edge_weight)
            self.register_buffer("cached_edge_index", cached_edge_index)
            self.register_buffer("cached_norm", cached_norm)

    def _laplacian(self, edge_index, edge_weight):
        lap_index, lap_norm = get_laplacian(edge_index, edge_weight, normalization="sym", num_nodes=self.num_nodes)
        lap_norm = lap_norm.clone()
        lap_norm[lap_index[0] == lap_index[1]] -= 1.0
        return lap_index, lap_norm

    def forward(self, x):
        batch_size, channels, steps, nodes = x.shape
        if nodes != self.num_nodes:
            raise ValueError("Graph node count does not match the input tensor.")
        if self.adaptive_graph is None:
            edge_index, norm = self.cached_edge_index, self.cached_norm
        else:
            edge_index, edge_weight = self.adaptive_graph.mixed_edges()
            edge_index, norm = self._laplacian(edge_index, edge_weight)
        node_features = x.permute(0, 2, 3, 1).reshape(batch_size * steps, nodes, channels)
        tx_0 = node_features
        convolved = self.conv.lins[0](tx_0)
        if len(self.conv.lins) > 1:
            tx_1 = self.conv.propagate(edge_index, x=tx_0, norm=norm)
            convolved = convolved + self.conv.lins[1](tx_1)
        for linear in self.conv.lins[2:]:
            tx_2 = 2.0 * self.conv.propagate(edge_index, x=tx_1, norm=norm) - tx_0
            convolved = convolved + linear(tx_2)
            tx_0, tx_1 = tx_1, tx_2
        if self.conv.bias is not None:
            convolved = convolved + self.conv.bias
        return convolved.reshape(batch_size, steps, nodes, -1).permute(0, 3, 1, 2)


class STConvBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, edge_index, edge_weight, dropout=0.1,
                 temporal_mode="single", adaptive_graph=None):
        super().__init__()
        if temporal_mode not in {"single", "multiscale"}:
            raise ValueError("temporal_mode must be 'single' or 'multiscale'.")
        temporal_cls = TemporalConvLayer if temporal_mode == "single" else MultiScaleTemporalConvLayer
        self.temporal1 = temporal_cls(c_in, c_mid)
        self.graph = PyGChebGraphConv(c_mid, c_mid, 3, edge_index, edge_weight, adaptive_graph)
        self.dropout = nn.Dropout(dropout)
        self.temporal2 = temporal_cls(c_mid, c_out)
        self.residual = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.c_out, self.nodes = c_out, int(edge_index.max()) + 1

    def forward(self, x):
        y = self.temporal1(x)
        y = self.graph(y)
        y = self.dropout(torch.relu(y))
        y = self.temporal2(y)
        y = y + self.residual(x)[:, :, -y.shape[2]:, :]
        return torch.nn.functional.layer_norm(y.permute(0, 2, 1, 3), (self.c_out, self.nodes)).permute(0, 2, 1, 3)


class STGCNEncoder(nn.Module):
    def __init__(self, edge_index, edge_weight, num_time_features=4, dropout=0.1,
                 graph_mode="physical", temporal_mode="single", use_daily_lag=False,
                 use_weekly_lag=False, adaptive_embed_dim=16, adaptive_top_k=16,
                 physical_graph_alpha=0.8, adaptive_edge_dropout=0.05):
        super().__init__()
        if graph_mode not in {"physical", "adaptive"}:
            raise ValueError("graph_mode must be 'physical' or 'adaptive'.")
        self.num_time_features, self.use_daily_lag = num_time_features, use_daily_lag
        self.use_weekly_lag, self.graph_mode, self.temporal_mode = use_weekly_lag, graph_mode, temporal_mode
        self.num_nodes = int(edge_index.max()) + 1
        self.adaptive_graph = AdaptiveAdjacency(self.num_nodes, edge_index, edge_weight, adaptive_embed_dim,
            adaptive_top_k, physical_graph_alpha, adaptive_edge_dropout) if graph_mode == "adaptive" else None
        periodic_channels = 2 * int(use_daily_lag) + 2 * int(use_weekly_lag)
        self.blocks = nn.ModuleList([
            STConvBlock(1 + num_time_features + periodic_channels, 32, 64, edge_index, edge_weight, dropout, temporal_mode, self.adaptive_graph),
            STConvBlock(64, 32, 128, edge_index, edge_weight, dropout, temporal_mode, self.adaptive_graph),
        ])
        self.output_temporal = TemporalConvLayer(128, 128)

    def forward(self, recent, time_feat, periodic_feat=None):
        if time_feat.shape[:2] != recent.shape[:2] or time_feat.shape[2] != self.num_time_features:
            raise ValueError("time_feat must have shape (batch, history, num_time_features).")
        features = [recent.unsqueeze(1), time_feat.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, recent.shape[2])]
        if self.use_daily_lag or self.use_weekly_lag:
            if periodic_feat is None or periodic_feat.shape[:3] != recent.shape or periodic_feat.shape[3] != 4:
                raise ValueError("periodic_feat must have shape (batch, history, nodes, 4).")
            channels = ([] if not self.use_daily_lag else [0, 1]) + ([] if not self.use_weekly_lag else [2, 3])
            features.append(periodic_feat[..., channels].permute(0, 3, 1, 2))
        x = torch.cat(features, dim=1)
        for block in self.blocks:
            x = block(x)
        return self.output_temporal(x)

    def graph_diagnostics(self):
        return self.adaptive_graph.diagnostics() if self.adaptive_graph is not None else {"graph_mode": "physical"}


class STGCN(nn.Module):
    """Original single-horizon regression baseline."""

    def __init__(self, edge_index, edge_weight, num_time_features=4, dropout=0.1, **encoder_options):
        super().__init__()
        self.encoder = STGCNEncoder(edge_index, edge_weight, num_time_features, dropout, **encoder_options)
        self.output = nn.Conv2d(128, 1, kernel_size=(2, 1))

    def forward(self, recent, time_feat, periodic_feat=None):
        return self.output(self.encoder(recent, time_feat, periodic_feat)).squeeze(1).squeeze(1)
