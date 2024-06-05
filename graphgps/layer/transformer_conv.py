import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import LayerNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register

from graphgps.optim.utils import GradClipper
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.virtual_node import ResidualVirtualNode


class CustomTransformerConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.,
        attn_dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.edge_dim = edge_dim

        self.head_dim = self.out_channels // self.heads
        self.coeff = math.sqrt(self.head_dim)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        assert out_channels % heads == 0

        self.lin_kv = Linear(in_channels[0], 2 * out_channels, bias=bias)
        self.lin_q = Linear(in_channels[1], out_channels, bias=bias)
        self.lin_out = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            register.act_dict[cfg.gnn.act](),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout),
        )

        self.lin_res = Linear(in_channels[0], out_channels, bias=False)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, 2 * out_channels, bias=bias)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_kv.reset_parameters()
        self.lin_q.reset_parameters()
        #for m in self.lin_out:
        #    m.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels // self.heads

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_q(x[1]).view(-1, H, C)

        key_value = self.lin_kv(x[0]).view(-1, H, 2*C)
        key, value = torch.tensor_split(key_value, 2, dim=-1)

        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        out = out.view(-1, self.out_channels)
        out = out + self.lin_res(x[1])

        out = self.lin_out(out) # + x[1]
        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_key_value = self.lin_edge(edge_attr).view(-1, self.heads, 2 * self.head_dim)
            edge_key, edge_value = torch.tensor_split(edge_key_value, 2, dim=-1)
            key_j += edge_key
            value_j += edge_value

        #alpha = torch.einsum('ehd,ehd->eh', query_i, key_j)
        alpha = query_i + key_j
        alpha = torch.sigmoid(alpha)

        #alpha /= self.coeff
        #alpha = softmax(alpha, index, ptr, size_i)
        #alpha = softmax(alpha.view(value_j.shape[0], -1), index, ptr, size_i)
        #alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        out = value_j
        out = out * alpha.view(-1, self.heads, out.shape[-1])
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class TransformerConvLayer(nn.Module):

    def __init__(self, h_dim, edge_dim=None, dropout=0.0, norm_first=False, residual=True, layer_id=0, **kwargs):
        super().__init__(**kwargs)

        heads = cfg.gnn.heads
        attn_dropout = cfg.gnn.attn_dropout
        #self.conv = CustomTransformerConv(
        #    in_channels=h_dim,
        #    out_channels=h_dim,
        #    edge_dim=edge_dim,
        #    heads=heads,
        #    dropout=dropout,
        #    attn_dropout=attn_dropout
        #)
        self.conv = GatedGCNLayer(
            h_dim, h_dim,
            dropout=dropout,
            residual=False,  # True,
            act=cfg.gnn.act,
        )

        self.vn = ResidualVirtualNode(h_dim)

        self.ffn = nn.Sequential(
            nn.Linear(h_dim, 2 * h_dim),
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(dropout),
            nn.Linear(2 * h_dim, h_dim),
            nn.Dropout(dropout),
        )

        self.norm_first = norm_first
        self.norm1 = LayerNorm(h_dim)
        self.norm2 = LayerNorm(h_dim)

        self.reset_parameters()

        self.res1_weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        self.res2_weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float))

        self.grad_clip_conv = GradClipper(1.0) #/ (2 * (cfg.gnn.layers_mp - layer_id) + 2))
        self.grad_clip_ffn = GradClipper(1.0) #/ (2 * (cfg.gnn.layers_mp - layer_id) + 1))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.ffn[0].weight)
        nn.init.xavier_normal_(self.ffn[3].weight)

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        r1 = 0.5 #torch.sigmoid(self.res1_weight)
        r2 = 0.5 #torch.sigmoid(self.res2_weight)

        if self.norm_first:
            h = self.norm1(x, batch.batch)
        else:
            h = x

        batch.x = h
        batch = self.conv(batch)
        h = batch.x

        #h = self.conv(x=h, edge_index=edge_index, edge_attr=edge_attr)
        #h = self.grad_clip_conv.set_output(h)

        if self.norm_first:
            x = x + h
            h = x
            h = self.norm2(h, batch.batch)
        else:
            x = self.norm1(r1 * x + (1.0 - r1) * h, batch.batch)
            h = x

        h = self.vn(h, batch.batch)

        h = self.ffn(h)
        #h = self.grad_clip_ffn.set_output(h)

        if self.norm_first:
            x = x + h
        else:
            x = self.norm2(r2 * x + (1.0 - r2) * h, batch.batch)

        batch.x = x
        return batch
