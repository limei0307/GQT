import copy
import torch
import pickle
import os.path as osp
import networkx as nx
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_undirected, to_networkx, from_dgl
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.datasets import (Planetoid, WikiCS, Coauthor, Amazon, LRGBDataset, 
                                      GNNBenchmarkDataset, Yelp, Flickr,
                                      Reddit2, PPI)

import dgl
from dgl.data import PubmedGraphDataset, CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.data import WikiCSDataset, ChameleonDataset, SquirrelDataset, ActorDataset, CornellDataset, TexasDataset, WisconsinDataset
from dgl.data import AmazonRatingsDataset, RomanEmpireDataset, MinesweeperDataset, QuestionsDataset

GRAPH_DICT = {
    "pubmed-vcr-graphormer": PubmedGraphDataset,
    "corafull-vcr-graphormer": CoraFullDataset,
    "computer-vcr-graphormer": AmazonCoBuyComputerDataset,
    "photo-vcr-graphormer": AmazonCoBuyPhotoDataset,
    "cs-vcr-graphormer": CoauthorCSDataset,
    "physics-vcr-graphormer": CoauthorPhysicsDataset,
    "squirrel-vcr-graphormer": SquirrelDataset,

    "squirrel-dgl-random-split": SquirrelDataset,
    "amazonratings-hetero-dgl": AmazonRatingsDataset, 
    "romanempire-hetero-dgl": RomanEmpireDataset, 
    "minesweeper-hetero-dgl": MinesweeperDataset, 
    "questions-hetero-dgl": QuestionsDataset,
}

def index2mask(idx, size):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

def get_data(config):
    dataset_name = config['dataset']

    if 'vcr-graphormer' in dataset_name or 'hetero-dgl' in dataset_name:
        if dataset_name == "squirrel-vcr-graphormer":
            file_path = "dataset/squirrel.pt"
            data_list = torch.load(file_path)
            adj_tensor = data_list[0]
            features = data_list[1]
            labels = data_list[2]
            import scipy.sparse as sp
            adj_matrix = sp.coo_matrix(adj_tensor)
            graph = dgl.from_scipy(adj_matrix)
            graph.ndata["feat"] = features
            graph.ndata["label"] = labels
            graph = graph.remove_self_loop()
            graph = graph.add_self_loop()
        else:
            dataset = GRAPH_DICT[dataset_name]()
            graph = dataset[0]
            graph = graph.remove_self_loop()
            graph = graph.add_self_loop()
        if config['to_undirected']:
            graph = dgl.to_bidirected(graph, copy_ndata=True)
        
        if 'vcr-graphormer' in dataset_name:
            print('======== following random 6/2/2 split ========')
            from dgl.data.utils import split_dataset
            train_idx, val_idx, test_idx = split_dataset(range(len(graph.ndata["feat"])), frac_list = [0.6, 0.2, 0.2], shuffle=True, random_state=42) # train_idx.indices is an array
            train_idx, val_idx, test_idx = train_idx.indices, val_idx.indices, test_idx.indices
        else:
            train_idx, val_idx, test_idx = graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"]
        data = from_dgl(graph)
        data.x = data.feat
        data.y = data.label
        data.num_nodes = len(data.x)

        config['num_features'] = graph.ndata["feat"].shape[1]
        # config['num_classes'] = dataset.num_classes
        try:
            config['num_classes'] = dataset.num_classes
        except:
            config['num_classes'] = max(graph.ndata["label"]) + 1

    else:
        root = config['root']

        transform = [T.AddSelfLoops()]
        if config['to_undirected']:
            transform.append(T.ToUndirected())
        if config['num_sign_hops'] > 0:
            transform.append(T.SIGN(config['num_sign_hops']))
        if config['normalize_feature']:
            transform.append(T.NormalizeFeatures())
        
        transform
        transform = T.Compose(transform)

        if dataset_name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(dataset_name, f'{root}/OGB', transform=transform)
            data = dataset[0]
            # data.y = data.y.view(-1)
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train']
            val_idx = split_idx['valid']
            test_idx = split_idx['test']
            data.train_mask = index2mask(split_idx['train'], data.num_nodes)
            data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
            data.test_mask = index2mask(split_idx['test'], data.num_nodes)
        elif dataset_name == 'flickr':
            dataset = Flickr(f'{root}/Flickr', pre_transform=transform)
            data = dataset[0]
            data.y = torch.unsqueeze(data.y, dim=1)
            train_idx = torch.arange(len(data.x))[data.train_mask]
            val_idx = torch.arange(len(data.x))[data.val_mask]
            test_idx = torch.arange(len(data.x))[data.test_mask]
        elif dataset_name == 'ogbn-products':
            dataset = PygNodePropPredDataset(dataset_name, f'{root}/OGB', transform=transform)
            data = dataset[0]
            # data.y = data.y.view(-1)
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train']
            val_idx = split_idx['valid']
            test_idx = split_idx['test']
            data.train_mask = index2mask(split_idx['train'], data.num_nodes)
            data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
            data.test_mask = index2mask(split_idx['test'], data.num_nodes)
        elif dataset_name in ('computers', 'photo'):
            dataset = Amazon(f'{root}/Amazon', dataset_name, transform=transform)
            data = dataset[0]
            num_nodes = data.x.size(0)
            train_size, val_size = int(num_nodes * .6), int(num_nodes * .2)
            indices = torch.randperm(num_nodes)
            train_idx = indices[:train_size]
            val_idx = indices[train_size: train_size + val_size]
            test_idx = indices[train_size + val_size:]
        elif dataset_name in ('cs', 'physics'):
            dataset = Coauthor(root, dataset_name, transform=transform)
            data = dataset[0]
            num_nodes = data.x.size(0)
            train_size, val_size = int(num_nodes * .6), int(num_nodes * .2)
            indices = torch.randperm(num_nodes)
            train_idx = indices[:train_size]
            val_idx = indices[train_size: train_size + val_size]
            test_idx = indices[train_size + val_size:]
        elif dataset_name in ('cora', 'citeseer', 'pubmed'):
            dataset = Planetoid(root, dataset_name, transform=transform)
            data = dataset[0]
            train_idx = torch.where(data.train_mask)[0]
            val_idx = torch.where(data.val_mask)[0]
            test_idx = torch.where(data.test_mask)[0]
        elif dataset_name == 'wikics':
            dataset = WikiCS(root, transform)
            data = dataset[0]
            train_idx = torch.where(data.train_mask[:, 0])[0]
            val_idx = torch.where(data.val_mask[:, 0])[0]
            test_idx = torch.where(data.test_mask)[0]
        elif dataset_name in ('pascalvoc-sp', 'coco-sp', 'pcqm-contact', 'peptides-func', 'peptides-struct'):
            train_dataset = LRGBDataset(root, dataset_name, split='train', transform=transform)
            val_dataset = LRGBDataset(root, dataset_name, split='val', transform=transform)
            test_dataset = LRGBDataset(root, dataset_name, split='test', transform=transform)

        if config['scale_feature']:
            epsilon = 1e-15
            std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
            data.x = (data.x - mean) / (std + epsilon)

        config['root'] = '/'.join(dataset.processed_dir.split('/')[:-1])

        config['num_features'] = dataset.num_features
        config['num_classes'] = dataset.num_classes
    
    pred_train_loader = DataLoader(train_idx, batch_size=config['batch_size'], shuffle=True)
    pred_val_loader = DataLoader(val_idx, batch_size=2 * config['batch_size'], shuffle=False)
    pred_test_loader = DataLoader(test_idx, batch_size=2 * config['batch_size'], shuffle=False)

    data.deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
    config['max_degree'] = min(data.deg.max().item(), config['max_degree'])
    data.deg[data.deg > config['max_degree']] = config['max_degree']

    if config['llm_embedding']:
        data = get_llm_embedding(dataset_name, data, root, config['device'])
    
    if config['tokenizer_input'] == 'sign':
        data = get_sign_features(data, config)
    
    if config['tokenizer_input'] == 'sign':
        config['vq_in_dim'] = config['num_features'] * (config['num_sign_hops']+1)
    elif config['tokenizer_input'] == 'feature':
        config['vq_in_dim'] = config['num_features']
    elif config['tokenizer_input'] == 'llm':
        config['vq_in_dim'] = 768

    return data, pred_train_loader, pred_val_loader, pred_test_loader, train_idx, val_idx, test_idx

def get_llm_embedding(dataset_name, data, root, device):
    path = osp.join(root, f'llm_embedding_{dataset_name}.pkl')

    if osp.exists(path):
        with open(path, 'rb') as f:
           embedding=pickle.load(f)
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('sentence-transformers/sentence-t5-base')
        edge_index = to_undirected(data.edge_index)
        x = data.x if isinstance(data.x, torch.Tensor) else torch.empty(edge_index[0].unique().shape[-1], 0)
        degrees = degree(edge_index[0])
        global_template = f'name: {dataset_name} nodes: {x.shape[0]} features: {x.shape[1]} edges: {len(edge_index[0])} degree: {degrees.mean().item():.2f} '
        text = [global_template + f'degree: {degrees[idx]} feature: [' + ' '.join([f'{s.item():0.4f}' for s in x[idx]]) + ']' for idx in range(len(x))]
        embedding = encoder.encode(sentences=text, batch_size=1024, show_progress_bar=True, convert_to_tensor=True, device=device).to('cpu')
        with open(path, 'wb') as f:
            pickle.dump(embedding, f)

    data.llm_embedding = embedding

    return data

def get_sign_features(data, config):
    data.sign_feat = []
    for hop in range(config['num_sign_hops'] + 1):
        if hop > 0:
            data.sign_feat.append(data[f'x{hop}'])
        else:
            data.sign_feat.append(data.x)
    data.sign_feat = torch.cat(data.sign_feat, dim=1)
    return data

def get_all_data(config):
    graphs = []
    # ['ogbn-arxiv', 'ogbn-products', 'computers', 'photo', 'cs', 'physics', 
    #                 'cora', 'citeseer', 'pubmed', 'wikics', 'pascalvoc-sp', 'coco-sp', 
    #                 'pcqm-contact', 'peptides-func', 'peptides-struct']:
    for dataset in ['ogbn-arxiv', 'computers', 'photo', 'cs', 'physics', 'pubmed']:
        tmp_config = copy.deepcopy(config)
        tmp_config['dataset'] = dataset
        tmp_config['num_sign_hops'] = 0
        graph, __, __, __, __, __, __ = get_data(tmp_config)
        graphs.append(Data(x=graph.llm_embedding, edge_index=graph.edge_index))
    graphs = Batch.from_data_list(graphs)
    return graphs


if __name__ == '__main__':
    get_data('cora', 1, 16)