import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pygnn
from torch_geometric.nn import LayerNorm
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.virtual_node import VirtualNode


class MPNNLayer(nn.Module):

    def __init__(self, dim_h, **kwargs):
        super().__init__(**kwargs)

        self.gnn_type = cfg.gnn.layer_type
        self.use_vn = cfg.gnn.use_vn

        self.supports_edge_attr = True
        self.activation = register.act_dict[cfg.gnn.act]
        self.dropout = cfg.gnn.dropout
        post_conv_dropout_p = self.dropout

        if self.gnn_type == "GCN":
            self.supports_edge_attr = False
            self.conv = pygnn.GCNConv(dim_h, dim_h)
        elif self.gnn_type == 'GIN':
            self.supports_edge_attr = False
            gin_nn = nn.Sequential(
                nn.BatchNorm1d(dim_h),
                self.activation(),
                nn.Dropout(self.dropout),
                pygnn.Linear(dim_h, dim_h)
            )
            self.conv = pygnn.GINConv(gin_nn)
            post_conv_dropout_p = 0.0
        elif self.gnn_type == 'GENConv':
            self.conv = pygnn.GENConv(dim_h, dim_h)
        elif self.gnn_type == 'GINE':
            gin_nn = nn.Sequential(
                nn.BatchNorm1d(dim_h),
                self.activation(),
                nn.Dropout(self.dropout),
                nn.Linear(dim_h, dim_h),
            )
            self.conv = pygnn.GINEConv(gin_nn)
            post_conv_dropout_p = 0.0
        elif self.gnn_type == 'GAT':
            num_heads = cfg.gnn.heads
            self.conv = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h
            )
        elif self.gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(cfg.gt.pna_degrees))
            self.conv = pygnn.PNAConv(
                dim_h, dim_h,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=min(128, dim_h),
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        elif self.gnn_type == 'CustomGatedGCN':
            self.conv = GatedGCNLayer(
                dim_h, dim_h,
                dropout=self.dropout,
                residual=False,
                act=cfg.gnn.act,
            )
            post_conv_dropout_p = 0.0
        else:
            raise ValueError(f"Unsupported local GNN model: {self.gnn_type}")

        self.post_conv_dropout = nn.Dropout(post_conv_dropout_p)

        self.norm_type = cfg.gnn.norm_type
        if self.norm_type == 'layer':
            self.norm_gnn = LayerNorm(dim_h)
            self.norm_out = LayerNorm(dim_h)
            self.norm_vn = LayerNorm(dim_h)
        else:
            self.norm_gnn = nn.BatchNorm1d(dim_h)
            self.norm_out = nn.BatchNorm1d(dim_h)
            self.norm_vn = nn.BatchNorm1d(dim_h)

        if self.use_vn:
            self.vn = VirtualNode(dim_h)
            self.dropout_vn = nn.Dropout(self.dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim_h, 2 * dim_h),
            self.activation(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * dim_h, dim_h),
            nn.Dropout(self.dropout),
        )

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        if self.gnn_type == 'CustomGatedGCN':
            batch = self.conv(batch)
            h_gnn = batch.x
        else:
            if self.supports_edge_attr:
                h_gnn = self.conv(x, batch.edge_index, batch.edge_attr)
            else:
                h_gnn = self.conv(x, batch.edge_index)

        h_gnn = self.post_conv_dropout(h_gnn)

        if self.norm_type == 'layer':
            h_gnn = self.norm_gnn(x + h_gnn, batch.batch)
        else:
            h_gnn = self.norm_gnn(x + h_gnn)

        if self.use_vn:
            h_vn = self.vn(x, batch.batch)
            h_vn = self.dropout_vn(h_vn)

            if self.norm_type == 'layer':
                h_vn = self.norm_vn(h_vn + x, batch.batch)
            else:
                h_vn = self.norm_vn(h_vn + x)

            h_gnn = h_gnn + h_vn

        x = h_gnn
        h = self.ffn(x)

        if self.norm_type == 'layer':
            x = self.norm_out(x + h, batch.batch)
        else:
            x = self.norm_out(x + h)

        batch.x = x
        return batch
