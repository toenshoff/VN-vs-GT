import os
import os.path as osp
import shutil
import pickle

import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from torch_geometric.utils import from_networkx


class Cor44_Syntetic(InMemoryDataset):

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
            
            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                nx_graph = graphs[idx]
                data = from_networkx(nx_graph)
                x = torch.zeros((data.num_nodes, 1), dtype=torch.long)
                y = torch.tensor([data.num_nodes**2], dtype=torch.float)
                edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.long)
                data.x = x
                data.y = y
                data.edge_attr = edge_attr

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))
