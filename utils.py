import copy
import torch
import walker
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch_geometric.transforms import Compose
from models.transformer import TransformerConfig
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.utils import k_hop_subgraph, to_networkx

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi / (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))

class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)

class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)

def setup_config(config):
    if config['llm_embedding'] and 'llm_emb' in config['transformer_input']:
        dict_size, input_dim = config['vq_codebook_size'] * config['vq_num_codebook'], 768
    elif config['transformer_input'] == 'token_id':
        dict_size, input_dim = config['vq_codebook_size'] * config['vq_num_codebook'], config['num_features']
    elif 'quantized_emb' in config['transformer_input'] or 'encoder_emb' in config['transformer_input']: 
        dict_size, input_dim = config['vq_codebook_size'] * config['vq_num_codebook'], config['vq_dim']
    elif 'sign_emb' in config['transformer_input']: 
        dict_size, input_dim = config['vq_codebook_size'] * config['vq_num_codebook'], config['num_features'] * (config['num_sign_hops+1'])
    elif 'input_emb' in config['transformer_input']: 
        dict_size, input_dim = config['vq_codebook_size'] * config['vq_num_codebook'], config['num_features']

    #Temporary added for sign ids
    # dict_size, input_dim = config['vq_codebook_size'] * (config['num_sign_hops'] + 1), config['vq_dim']
    
    if config['sequence_type'] == 'ppr':
        node_len = config['ppr_num_neighbors'] + 1
    elif config['sequence_type'] == 'bfs':
        node_len = config['max_sequence_length']
    elif config['sequence_type'] == 'dfs':
        node_len = config['dfs_walk_length']

    num_token_each_node = 0
    if '_emb' in config['transformer_input']:
        num_token_each_node += 1
    if 'token_id' in config['transformer_input']:
        num_token_each_node += config['vq_num_codebook']
        #temporary for sign
        # num_token_each_node = config['num_sign_hops'] + 1
    
    transformer_config = TransformerConfig(dict_size=dict_size, num_layers=config['transformer_num_layers'], in_dim=input_dim,
                                           ffn_dim=config['transformer_ffn'], hidden_dim=config['transformer_hidden_dim'], 
                                           num_classes=config['num_classes'], num_heads=config['transformer_num_heads'],
                                           activation=config['transformer_activation'], dropout=config['transformer_dropout'], 
                                           num_token_each_node=num_token_each_node, node_len=node_len, max_degree=config['max_degree'],
                                           max_distance=config['max_distance'], degree_encoding=config['degree_encoding'], 
                                           distance_encoding=config['distance_encoding'], hierarchy_encoding=config['hierarchy_encoding'],
                                           position_encoding=config['position_encoding'], 
                                           aggr_tokens=config['transformer_aggr_tokens'], aggr_neighbors=config['transformer_aggr_neighbors'],
                                           use_codebook=config['use_codebook'], use_gating=config['use_gating'])

    return transformer_config

def sparsify(neighbors, weights, topk):
    new_neighbors = []
    for n, w in zip(neighbors, weights):
        idx_topk = np.argsort(w)[-topk:]
        new_neighbor = n[idx_topk]
        new_neighbors.append(new_neighbor)
    return new_neighbors

def process_ppr(ppr, n_nodes, n_neighbors):
    neighbors, weights = ppr
    _, neighbor_counts = neighbors[0].unique(return_counts=True)
    neighbors = [n.numpy() for n in torch.split(neighbors[1], neighbor_counts.tolist(), dim=0)]
    weights = [n.numpy() for n in torch.split(weights, neighbor_counts.tolist(), dim=0)]
    neighbors = sparsify(neighbors, weights, topk=n_neighbors)
    neighbors = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in neighbors], batch_first=True, padding_value=-1)
    neighbors = torch.cat((torch.arange(n_nodes).unsqueeze(dim=-1), neighbors), dim=1) # N * (1 + topk)
    return neighbors, weights

def get_bfs(graph, n_hops):
    bfs = []
    n_nodes = graph.x.shape[0]
    for i in tqdm(range(n_nodes)):
        neighbors, __, __, __ = k_hop_subgraph(node_idx=i, num_hops=n_hops, edge_index=graph.edge_index, num_nodes=n_nodes)
        bfs.append(neighbors)
    return bfs

def get_dfs(graph, n_walks, walk_len, p=1.0, q=1.0):
    g = to_networkx(graph)
    x = walker.random_walks(g, n_walks=n_walks, walk_len=walk_len, p=p, q=q)
    x = torch.from_numpy(x.astype(np.int32))
    x = x.view(n_walks, graph.x.shape[0],  walk_len).transpose(1, 0)
    return x

def get_distance(graph, sequence):
    g = to_networkx(graph)
    seq = sequence.reshape(-1, sequence.shape[-1])
    seq_np = seq.numpy()

    unique_pairs = set()
    for path in tqdm(seq_np):
        unique_pairs.update(set([tuple((path[0], target)) for target in path[1:]]))
    
    dist_dict = {}
    for src, target in tqdm(unique_pairs, desc='Computing Shortest Path...'):
        dist_dict[tuple((src, target))] = nx.shortest_path_length(g, src, target)
    
    distance = torch.zeros_like(seq)
    for idx, path in enumerate(seq_np):
        distance[idx, 1:] = torch.LongTensor([dist_dict[tuple((path[0], target))] for target in path[1:]])
    
    return distance.view(*sequence.shape)