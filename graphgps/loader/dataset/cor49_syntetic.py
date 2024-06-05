import os.path as osp
import pickle

import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx


class Cor49_Syntetic(InMemoryDataset):

    def __init__(self, root, split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)
            
            graph_list = graphs[0]
            l = torch.tensor(graphs[1])
            r = torch.tensor(graphs[2])
            num_nodes_w = torch.tensor(graphs[3])
        
            indices = range(len(l))
            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                
                nx_graph = graph_list[idx]
                data = from_networkx(nx_graph)
                assert l[idx] + num_nodes_w[idx] == data.num_nodes
                
                # compute node features
                x_u = torch.tensor([[2, 1]], dtype=torch.float).repeat(int(l[idx]),1)
                x_w = torch.tensor([[2, 2]], dtype=torch.float).repeat(int(num_nodes_w[idx]),1)
                
                x = torch.cat((x_u, x_w), dim=0)
                
                # compute target
                e_9 = torch.exp(torch.tensor([9]))
                e_12 = torch.exp(torch.tensor([12]))
                
                f_u = (3+2*r[idx]*e_9)/(1+r[idx]*e_9)
                f_w = (3+2*r[idx]*e_12)/(1+r[idx]*e_12)
            
                y = l[idx]*f_u + l[idx]*r[idx]*f_w
                
                # save info to data
                data.x = x
                data.y = y
                data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.long)
                   
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
