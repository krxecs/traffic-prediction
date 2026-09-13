"""Shared STGCN encoder and the baseline and multitask output variants."""

import torch
from torch import nn
from torch_geometric.nn import ChebConv
from torch_geometric.utils import get_laplacian


class TemporalConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kt=3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kt, 1))

    def forward(self, x):
        linear, gate = self.conv(x).chunk(2, dim=1)
        return linear * torch.sigmoid(gate)


class PyGChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, ks, edge_index, edge_weight):
        super().__init__()
        self.conv = ChebConv(c_in, c_out, K=ks, normalization="sym")
        cached_edge_index, cached_norm = get_laplacian(
            edge_index,
            edge_weight,
            normalization="sym",
            num_nodes=int(edge_index.max()) + 1,
        )
        cached_norm[cached_edge_index[0] == cached_edge_index[1]] -= 1.0
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.register_buffer("cached_edge_index", cached_edge_index)
        self.register_buffer("cached_norm", cached_norm)

    def forward(self, x):
        batch_size, channels, steps, nodes = x.shape
        node_features = x.permute(0, 2, 3, 1).reshape(batch_size * steps, nodes, channels)
        tx_0 = node_features
        convolved = self.conv.lins[0](tx_0)
        if len(self.conv.lins) > 1:
            tx_1 = self.conv.propagate(self.cached_edge_index, x=tx_0, norm=self.cached_norm)
            convolved = convolved + self.conv.lins[1](tx_1)
        for linear in self.conv.lins[2:]:
            tx_2 = self.conv.propagate(self.cached_edge_index, x=tx_1, norm=self.cached_norm)
            tx_2 = 2.0 * tx_2 - tx_0
            convolved = convolved + linear(tx_2)
            tx_0, tx_1 = tx_1, tx_2
        if self.conv.bias is not None:
            convolved = convolved + self.conv.bias
        return convolved.reshape(batch_size, steps, nodes, -1).permute(0, 3, 1, 2)


class STConvBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, edge_index, edge_weight, dropout=0.1):
        super().__init__()
        self.temporal1 = TemporalConvLayer(c_in, c_mid)
        self.graph = PyGChebGraphConv(c_mid, c_mid, 3, edge_index, edge_weight)
        self.dropout = nn.Dropout(dropout)
        self.temporal2 = TemporalConvLayer(c_mid, c_out)
        self.residual = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.c_out = c_out
        self.nodes = int(edge_index.max()) + 1

    def forward(self, x):
        y = self.temporal1(x)
        y = self.graph(y)
        y = self.dropout(torch.relu(y))
        y = self.temporal2(y)
        y = y + self.residual(x)[:, :, -y.shape[2]:, :]
        return torch.nn.functional.layer_norm(
            y.permute(0, 2, 1, 3), (self.c_out, self.nodes)
        ).permute(0, 2, 1, 3)


class STGCNEncoder(nn.Module):
    def __init__(self, edge_index, edge_weight, num_time_features=4, dropout=0.1):
        super().__init__()
        self.num_time_features = num_time_features
        self.blocks = nn.ModuleList(
            [
                STConvBlock(1 + num_time_features, 32, 64, edge_index, edge_weight, dropout),
                STConvBlock(64, 32, 128, edge_index, edge_weight, dropout),
            ]
        )
        self.output_temporal = TemporalConvLayer(128, 128)

    def forward(self, recent, time_feat):
        if time_feat.shape[:2] != recent.shape[:2] or time_feat.shape[2] != self.num_time_features:
            raise ValueError("time_feat must have shape (batch, history, num_time_features).")
        time_feat = time_feat.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, recent.shape[2])
        x = torch.cat((recent.unsqueeze(1), time_feat), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.output_temporal(x)


class STGCN(nn.Module):
    """Original single-horizon regression baseline."""

    def __init__(self, edge_index, edge_weight, num_time_features=4, dropout=0.1):
        super().__init__()
        self.encoder = STGCNEncoder(edge_index, edge_weight, num_time_features, dropout)
        self.output = nn.Conv2d(128, 1, kernel_size=(2, 1))

    def forward(self, recent, time_feat):
        return self.output(self.encoder(recent, time_feat)).squeeze(1).squeeze(1)

