import logging

import torch

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.data import WikiCSDataset, ChameleonDataset, SquirrelDataset, ActorDataset, CornellDataset, TexasDataset, WisconsinDataset
from dgl.data import AmazonRatingsDataset, RomanEmpireDataset, MinesweeperDataset, QuestionsDataset

from ogb.nodeproppred import DglNodePropPredDataset

from sklearn.preprocessing import StandardScaler


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,

    "pubmed-vcr-graphormer": PubmedGraphDataset,
    "corafull-vcr-graphormer": CoraFullDataset,
    "computer-vcr-graphormer": AmazonCoBuyComputerDataset,
    "photo-vcr-graphormer": AmazonCoBuyPhotoDataset,
    "cs-vcr-graphormer": CoauthorCSDataset,
    "physics-vcr-graphormer": CoauthorPhysicsDataset,
    "squirrel-vcr-graphormer": SquirrelDataset,
    "squirrel-dgl-random-split": SquirrelDataset,

    "wikics-ssl": WikiCSDataset,
    "computer-ssl": AmazonCoBuyComputerDataset,
    "photo-ssl": AmazonCoBuyPhotoDataset,
    "chameleon-ssl": ChameleonDataset,
    "squirrel-ssl": SquirrelDataset,
    "actor-ssl": ActorDataset,
    "cornell-ssl": CornellDataset,
    "texas-ssl": TexasDataset,
    "wisconsin-ssl": WisconsinDataset,

    "amazonratings-hetero-dgl": AmazonRatingsDataset, 
    "romanempire-hetero-dgl": RomanEmpireDataset, 
    "minesweeper-hetero-dgl": MinesweeperDataset, 
    "questions-hetero-dgl": QuestionsDataset,
}

def load_small_dataset(dataset_name, split_seed=42, to_undirected=True):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    elif dataset_name == "squirrel-vcr-graphormer":
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
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    try:
        num_classes = dataset.num_classes
    except:
        num_classes = max(graph.ndata["label"]) + 1

    if 'vcr-graphormer' in dataset_name:
        print('======== following random 6/2/2 split ========')
        if to_undirected:
            graph = dgl.to_bidirected(graph, copy_ndata=True)
        from dgl.data.utils import split_dataset
        
        idx_train, idx_val, idx_test = split_dataset(range(len(graph.ndata["feat"])), frac_list = [0.6, 0.2, 0.2], shuffle=True, random_state=split_seed)
        new_train_mask = torch.zeros(len(graph.ndata["feat"]), dtype=torch.bool)
        new_train_mask[idx_train] = True
        new_val_mask = torch.zeros(len(graph.ndata["feat"]), dtype=torch.bool)
        new_val_mask[idx_val] = True
        new_test_mask = torch.zeros(len(graph.ndata["feat"]), dtype=torch.bool)
        new_test_mask[idx_test] = True

        graph.ndata["train_mask"] = new_train_mask
        graph.ndata["val_mask"] = new_val_mask
        graph.ndata["test_mask"] = new_test_mask
    
    elif dataset_name in ["wikics-ssl", "computer-ssl", "photo-ssl"]: # random 1/1/8
        print('======== following random 1/1/8 split ========')
        if to_undirected:
            graph = dgl.to_bidirected(graph, copy_ndata=True)
        from dgl.data.utils import split_dataset
        
        idx_train, idx_val, idx_test = split_dataset(range(len(graph.ndata["feat"])), frac_list = [0.1, 0.1, 0.8], shuffle=True, random_state=split_seed)
        new_train_mask = torch.zeros(len(graph.ndata["feat"]), dtype=torch.bool)
        new_train_mask[idx_train] = True
        new_val_mask = torch.zeros(len(graph.ndata["feat"]), dtype=torch.bool)
        new_val_mask[idx_val] = True
        new_test_mask = torch.zeros(len(graph.ndata["feat"]), dtype=torch.bool)
        new_test_mask[idx_test] = True

        graph.ndata["train_mask"] = new_train_mask
        graph.ndata["val_mask"] = new_val_mask
        graph.ndata["test_mask"] = new_test_mask

    elif dataset_name in ["chameleon-ssl", "squirrel-ssl", "actor-ssl", "cornell-ssl", "texas-ssl", "wisconsin-ssl"]: # following given splits
        print('======== following given splits ========')
        if to_undirected:
            graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph.ndata["train_mask_all"] = graph.ndata["train_mask"]
        graph.ndata["val_mask_all"] = graph.ndata["val_mask"]
        graph.ndata["test_mask_all"] = graph.ndata["test_mask"]
    
    elif dataset_name in ["amazonratings", "romanempire", "minesweeper", "questions"]: # following given splits
        print('======== following given splits ========')
        if to_undirected:
            graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph.ndata["train_mask_all"] = graph.ndata["train_mask"]
        graph.ndata["val_mask_all"] = graph.ndata["val_mask"]
        graph.ndata["test_mask_all"] = graph.ndata["test_mask"]
    
    return graph, (num_features, num_classes)

def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    src, dst = graph.all_edges()
    # graph.add_edges(dst, src)
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph


def scale_feats(x):
    logging.info("### scaling features ###")
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
