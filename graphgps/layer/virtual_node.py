import torch
from torch.nn import Module, Linear, Sequential, Dropout, BatchNorm1d, Parameter, LayerNorm, Identity
from torch_geometric.nn import LayerNorm as PyGLayerNorm
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register


class VirtualNode(Module):

    def __init__(self, dim_h):
        super(VirtualNode, self).__init__()
        self.aggr = register.pooling_dict[cfg.gnn.vn_pooling]
        self.ffn = Sequential(
            Linear(dim_h, 2 * dim_h),
            register.act_dict[cfg.gnn.act](),
            Dropout(cfg.gnn.dropout),
            Linear(2 * dim_h, dim_h),
        )

    def forward(self, h, batch):
        h_vn = self.aggr(h, batch)
        h_vn = self.ffn(h_vn)
        return h_vn[batch]


class ResidualVirtualNode(Module):

    def __init__(self, dim_h):
        super(ResidualVirtualNode, self).__init__()
        self.aggr = register.pooling_dict[cfg.gnn.vn_pooling]

        if cfg.gnn.vn_norm == 'layer':
            norm = LayerNorm(dim_h)
        elif cfg.gnn.vn_norm == 'batch':
            norm = BatchNorm1d(dim_h)
        else:
            norm = Identity()

        self.ffn = Sequential(
            Linear(dim_h, 2 * dim_h),
            register.act_dict[cfg.gnn.act](),
            Dropout(cfg.gnn.dropout),
            Linear(2 * dim_h, dim_h),
            norm,
        )

    def forward(self, batch):
        h_vn = self.aggr(batch.x, batch.batch)
        
        #if 'x_vn' in batch:
        #    h_vn += batch.x_vn
        
        h_vn = self.ffn(h_vn)
        batch.x = batch.x + h_vn[batch.batch]
        batch.x_vn = h_vn
        return batch
