import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder
from torch_geometric.graphgym.register import register_network

from graphgps.layer.mpnn_layer import MPNNLayer


@register_network('mpnn')
class MPNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        layers = []
        for i in range(cfg.gnn.layers_mp):
            layers.append(MPNNLayer(dim_in))
        self.gnn_layers = torch.nn.ModuleList(layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)

        for layer in self.gnn_layers:
            batch = layer(batch)

        return self.post_mp(batch)
