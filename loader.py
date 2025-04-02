import torch
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.utils import get_ppr, k_hop_subgraph
from utils import process_ppr, get_dfs, get_bfs, get_distance

class TransformerDataset(Dataset):
    def __init__(self, graph, config) -> None:
        super().__init__()
        self.graph = graph
        self.distance_encoding = config['distance_encoding']
        self.degree_encoding = config['degree_encoding']
        self.n_hops = config['bfs_num_hops']
        self.n_walks = config['dfs_num_walks']
        self.sequence_type = config['sequence_type']
        if config['sequence_type'] == 'ppr':
            self.max_sequence_length = config['ppr_num_neighbors'] + 1
        elif config['sequence_type'] == 'bfs':
            self.max_sequence_length = config['max_sequence_length']
        elif config['sequence_type'] == 'dfs':
            self.max_sequence_length = config['dfs_walk_length']
 
        if self.sequence_type == 'ppr':
            path = f"{config['root']}/{config['dataset']}_ppr.pt"
            if osp.exists(path):
                ppr = torch.load(path)
            else:
                ppr = get_ppr(graph.edge_index, num_nodes=len(graph.x))
                torch.save(ppr, path)
            self.neighbors, self.weights = process_ppr(ppr, len(graph.x), config['ppr_num_neighbors'])
        elif self.sequence_type == 'dfs':
            self.neighbors = get_dfs(self.graph, config['dfs_num_walks'], config['dfs_walk_length'])
        elif self.sequence_type == 'bfs':
            self.neighbors = get_bfs(self.graph, config['bfs_num_hops'])
        
        if self.distance_encoding:
            self.distance = get_distance(self.graph, self.neighbors)
            self.max_distance = self.distance.max()
        
        if self.degree_encoding:
            self.degree = graph.deg
    
    def __len__(self):
        return len(self.graph.x)

    def __getitem__(self, idx):
        if self.sequence_type == 'ppr':
            neighbors = self.neighbors[idx]
        elif self.sequence_type == 'dfs':
            rnd_path = torch.randint(0, self.n_walks, (1, )).item()
            neighbors = self.neighbors[idx, rnd_path, :]
        elif self.sequence_type == 'bfs':
            neighbors = self.neighbors[idx]

        if self.distance_encoding:
            distances = self.distance[idx, rnd_path, :] if self.sequence_type == 'dfs' else self.distance[idx]
        else:
            distances = torch.zeros_like(neighbors)
        
        if self.degree_encoding:
            degree = self.degree[neighbors]
        else:
            degree = torch.zeros_like(neighbors)

        return neighbors[:self.max_sequence_length], degree[:self.max_sequence_length], distances[:self.max_sequence_length]
        
def get_tokenizer_loader(config, data):
    tokenizer_train_loader = RandomNodeLoader(
        Data(x=data.x, edge_index=data.edge_index, y=data.y, sign_feat=data.sign_feat if config['tokenizer_input']=='sign' else None),
        num_parts=config['n_parts'],
        shuffle=True,
        num_workers=12,
        persistent_workers=True,
    )
    tokenizer_test_loader = RandomNodeLoader(
        Data(x=data.x, edge_index=data.edge_index, y=data.y, sign_feat=data.sign_feat if config['tokenizer_input']=='sign' else None),
        num_parts=config['n_parts'],
        shuffle=False,
        num_workers=12,
        persistent_workers=True,
    )
    return tokenizer_train_loader, tokenizer_test_loader

def get_tokenizer_loader_sign(config, data):
    train_loader = DataLoader(data.sign_feat, config['batch_size'], shuffle=True, num_workers=12)
    test_loader = DataLoader(data.sign_feat, config['batch_size'], shuffle=False, num_workers=12)
    # tokenizer_train_loader = RandomNodeLoader(
    #     Data(x=data.x, edge_index=data.edge_index, y=data.y, sign_feat=data.sign_feat if config['tokenizer_input']=='sign' else None),
    #     num_parts=config['n_parts'],
    #     shuffle=True,
    #     num_workers=12,
    #     persistent_workers=True,
    # )
    # tokenizer_test_loader = RandomNodeLoader(
    #     Data(x=data.x, edge_index=data.edge_index, y=data.y, sign_feat=data.sign_feat if config['tokenizer_input']=='sign' else None),
    #     num_parts=config['n_parts'],
    #     shuffle=False,
    #     num_workers=12,
    #     persistent_workers=True,
    # )
    return train_loader, test_loader

def get_transformer_loader(config, data, train_idx, val_idx, test_idx):
    train_mask = torch.zeros(len(data.x), dtype=torch.bool)
    train_mask[train_idx] = True

    def collate(batched_data):
        sequence = torch.nn.utils.rnn.pad_sequence([d[0] for d in batched_data], batch_first=True, padding_value=-1)
        degree = torch.nn.utils.rnn.pad_sequence([d[1] for d in batched_data], batch_first=True, padding_value=-1)
        distances = torch.nn.utils.rnn.pad_sequence([d[2] for d in batched_data], batch_first=True, padding_value=-1)
        mask = train_mask[sequence]
        label = data.y[sequence]
        transformer_input = data.transformer_input[sequence]
        return (sequence, degree, distances, weight, transformer_input, label, mask)

    ds = TransformerDataset(data, config)
    kwargs = dict(collate_fn=collate, batch_size=config['batch_size']) #, num_workers=12, prefetch_factor=2, persistent_workers=True)
    if config['first_node_only']:
        train_loader = DataLoader(Subset(ds, train_idx), shuffle=True, **kwargs)
    else:
        train_loader = DataLoader(ds, shuffle=True, **kwargs)
    val_loader = DataLoader(Subset(ds, val_idx), shuffle=False, **kwargs)
    test_loader = DataLoader(Subset(ds, test_idx), shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader