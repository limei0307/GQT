
from args import parse_args

import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GCN, GraphSAGE, GIN, GAT, PNA
from torch_geometric.nn.inits import reset, uniform

import copy
from typing import Callable, Tuple
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from tqdm import tqdm
from utils import CosineDecayScheduler

EPS = 1e-15

from torch_geometric.utils import k_hop_subgraph, get_ppr
import numpy as np

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0), device=x.device)], edge_index

def sparsify(neighbors, weights, topk):
    new_neighbors = []
    for n, w in zip(neighbors, weights):
        idx_topk = np.argsort(w)[-topk:]
        new_neighbor = n[idx_topk]
        new_neighbors.append(new_neighbor)
    return new_neighbors

def process_ppr(ppr, n_nodes, n_neighbors, padding_value=-1):
    neighbors, weights = ppr
    _, neighbor_counts = neighbors[0].unique(return_counts=True)
    neighbors = [n.numpy() for n in torch.split(neighbors[1], neighbor_counts.tolist(), dim=0)]
    weights = [n.numpy() for n in torch.split(weights, neighbor_counts.tolist(), dim=0)]
    topk_neighbors = sparsify(neighbors, weights, topk=n_neighbors)
    topk_neighbors = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in topk_neighbors], batch_first=True, padding_value=padding_value)
    # neighbors = torch.cat((torch.arange(n_nodes).unsqueeze(dim=-1), neighbors), dim=1) # N * (1 + topk)
    return neighbors, weights, topk_neighbors

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x

class GCN_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.5):
        super(GCN_Encoder, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # return x.log_softmax(dim=-1)
        return x

class SAGE_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.5):
        super(SAGE_Encoder, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # return x.log_softmax(dim=-1)
        return x

class DeepGraphInfomax(torch.nn.Module):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (torch.nn.Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """
    def __init__(
        self,
        hidden_channels: int,
        encoder: Module,
        summary: Callable,
        corruption: Callable,
        config,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = Parameter(torch.empty(hidden_channels, hidden_channels))

        self.quantizer = ResidualVQ(
            dim = config['vq_dim'],
            num_quantizers = config['vq_num_codebook'],
            codebook_size = config['vq_codebook_size'],
            kmeans_init = True,
            use_cosine_sim = True
        )

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation.
        """
        pos_z = self.encoder(*args, **kwargs)
        pos_z_quantized, pos_indices, pos_commit_loss = self.quantizer(pos_z)

        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        cor_args = cor[:len(args)]
        cor_kwargs = copy.copy(kwargs)
        for key, value in zip(kwargs.keys(), cor[len(args):]):
            cor_kwargs[key] = value

        neg_z = self.encoder(*cor_args, **cor_kwargs)
        neg_z_quantized, neg_indices, neg_commit_loss = self.quantizer(neg_z)

        summary = self.summary(pos_z, *args, **kwargs)
        summary_quantized = self.summary(pos_z_quantized, *args, **kwargs)

        return pos_z, neg_z, summary, pos_z_quantized, neg_z_quantized, summary_quantized, torch.sum(pos_commit_loss), torch.sum(neg_commit_loss), pos_indices
    
    @torch.no_grad()
    def encode(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        self.encoder.eval()
        self.quantizer.eval()
        x = self.encoder(*args, **kwargs)
        quantized, indices, __, all_codes = self.quantizer(x, return_all_codes=True)
        codebooks = torch.cat([layer.codebook for layer in self.quantizer.layers])
        return x, quantized, indices, all_codes, codebooks

    def discriminate(self, z: Tensor, summary: Tensor,
                     sigmoid: bool = True) -> Tensor:
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (torch.Tensor): The latent space.
            summary (torch.Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z: Tensor, neg_z: Tensor, summary: Tensor) -> Tensor:
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 -
                              self.discriminate(neg_z, summary, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss

    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = 'lbfgs',
        *args,
        **kwargs,
    ) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hidden_channels})'

class Tokenizer:
    def __init__(self, config):
        self.config = config
        if config['gnn_type'] == 'gcn':
            encoder = GCN(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
        elif config['gnn_type'] == 'graphsage':
            encoder = GraphSAGE(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
        elif config['gnn_type'] == 'gin':
            encoder = GIN(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
        elif config['gnn_type'] == 'gat':
            encoder = GAT(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
        elif config['gnn_type'] == 'pna':
            encoder = PNA(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
    
        self.model = DeepGraphInfomax(
            hidden_channels=config['vq_dim'] if self.config['vq_method'] != 'none' else config['gnn_hidden_dim'],
            encoder=encoder,
            summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(),
            corruption=corruption, config=config,
        ).to(config['device'])
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr_tokenizer'], weight_decay=config['weight_decay_tokenizer'])
        self.scheduler = CosineDecayScheduler(config['lr_tokenizer'], config['warmup_tokenizer'], config['epochs_tokenizer'])
    
    def train(self, loader, epoch):
        if self.config['scheduler_tokenizer']:
            lr = self.scheduler .get(epoch)
            for param_group in self.optimizer .param_groups:
                param_group['lr'] = lr

        self.model.train()
        total_loss, total_commit_loss, total_infomax_loss = [], [], []
        for batch in tqdm(loader, disable=True):
            self.optimizer.zero_grad(set_to_none=True)
            batch = batch.to(self.config['device'])
            pos_z, neg_z, summary, pos_z_quantized, neg_z_quantized, summary_quantized, pos_commit_loss, neg_commit_loss, pos_indices = self.model(batch.x, batch.edge_index)
            commit_loss = (pos_commit_loss + neg_commit_loss) / 2.
            if self.config['vq_method'] != 'none':
                infomax_loss = self.model.loss(pos_z_quantized, neg_z_quantized, summary_quantized)
                loss = infomax_loss + commit_loss * self.config['commit_loss_weight']
            else:
                infomax_loss = self.model.loss(pos_z, neg_z, summary)
                loss = infomax_loss
            total_loss.append(loss.item())
            total_commit_loss.append(commit_loss.item())
            total_infomax_loss.append(infomax_loss.item())
            loss.backward()
            self.optimizer.step()

        return {'loss': np.mean(total_loss), 'commit_loss': np.mean(total_commit_loss), 'infomax_loss': np.mean(total_infomax_loss)}

    @torch.no_grad()
    def encode(self, loader):
        x_all, quantized_all, indices_all = [], [], []
        self.model.eval()
        for batch in tqdm(loader, disable=True):
            batch = batch.to(self.config['device'])
            x, quantized, indices, all_codes, codebooks = self.model.encode(batch.x, batch.edge_index)
            x_all.append(x)
            quantized_all.append(quantized)
            indices_all.append(indices)
        return torch.cat(x_all, dim=0), torch.cat(quantized_all, dim=0), torch.cat(indices_all, dim=0), all_codes, codebooks



# def train(model, data, optimizer):
#     model.train()
#     optimizer.zero_grad()
#     pos_z, neg_z, summary, pos_z_quantized, neg_z_quantized, summary_quantized, pos_commit_loss, neg_commit_loss, pos_indices = model(data.x, data.edge_index)
#     if config['vq_method'] != 'none':
#         loss = model.loss(pos_z_quantized, neg_z_quantized, summary_quantized) + config['commit_loss_weight'] * pos_commit_loss + config['commit_loss_weight'] * neg_commit_loss
#     else:
#         loss = model.loss(pos_z, neg_z, summary)
#     loss.backward()
#     optimizer.step()
#     return loss.item(), pos_commit_loss.item(), neg_commit_loss.item()


def test(model, data):
    model.eval()
    z_encoder, _, _, z_quantized, _, _, _, _, pos_indices = model(data.x, data.edge_index)
    pos_indices_ppr = pos_indices[data.topk_ppr_neighbors].reshape(len(data.x), -1)
    

    acc_encoder = model.test(z_encoder[data.train_mask], data.y[data.train_mask],
                     z_encoder[data.test_mask], data.y[data.test_mask], max_iter=5000)
    acc_quantized = model.test(z_quantized[data.train_mask], data.y[data.train_mask],
                     z_quantized[data.test_mask], data.y[data.test_mask], max_iter=5000)
    acc_indices = model.test(pos_indices[data.train_mask], data.y[data.train_mask],
                     pos_indices[data.test_mask], data.y[data.test_mask], max_iter=500)
    acc_indices_ppr = model.test(pos_indices_ppr[data.train_mask], data.y[data.train_mask],
                     pos_indices_ppr[data.test_mask], data.y[data.test_mask], max_iter=5000)
    acc_input = model.test(data.x[data.train_mask], data.y[data.train_mask],
                     data.x[data.test_mask], data.y[data.test_mask], max_iter=5000)
    return acc_encoder, acc_quantized, acc_indices, acc_indices_ppr, acc_input, z_encoder, z_quantized, pos_indices


# def main(config):
#     data, _, _, _, _, _, _ = get_data(config)
#     path = f"{config['dataset']}/{config['dataset']}_ppr.pt"
#     if osp.exists(path):
#         ppr = torch.load(path)
#     else:
#         raise ValueError
#     _, _, data.topk_ppr_neighbors = process_ppr(ppr, len(data.x), 30, padding_value=0)

#     if config['gnn_type'] == 'gcn':
#         encoder = GCN(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
#     elif config['gnn_type'] == 'graphsage':
#         encoder = GraphSAGE(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
#     elif config['gnn_type'] == 'gin':
#         encoder = GIN(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
#     elif config['gnn_type'] == 'gat':
#         encoder = GAT(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
#     elif config['gnn_type'] == 'pna':
#         encoder = PNA(config['num_features'], config['gnn_hidden_dim'], config['gnn_num_layers'], config['vq_dim'], dropout=0, act='relu')
    
#     model = DeepGraphInfomax(
#         hidden_channels=config['gnn_hidden_dim'],
#         encoder=encoder,
#         summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(),
#         corruption=corruption, config=config,
#     ).to(config['device'])
    
#     data = data.to(config['device'])
#     optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_tokenizer'])
#     pre_loss = 1000
#     early_stop = 0
#     for epoch in tqdm(range(1,config['epochs_tokenizer'] + 1)):
#         loss, pos_commit_loss, neg_commit_loss = train(model, data, optimizer)
#         # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, pos commit Loss: {pos_commit_loss:.4f}, neg commit Loss: {neg_commit_loss:.4f}')
#         if loss > pre_loss:
#             early_stop += 1
#         else:
#             early_stop = 0
#         pre_loss = loss
#         if early_stop > config['early_stop_epochs']:
#             break
#         if epoch % config['save_epochs'] == 0:
#             acc_encoder, acc_quantized, acc_indices, acc_indices_ppr, acc_input, z_encoder, z_quantized, pos_indices = test(model, data)
#             # save_path =  config['dataset'] +'_dgi_tune' \
#             #         + '_' + str(config['vq_dim']) + '_' + str(config['vq_codebook_size']) + '_' + str(config['vq_num_codebook']) \
#             #         + '_' + str(epoch)
#             # torch.save(z_encoder.cpu(), save_path + '_encoder_emb.pt')
#             # torch.save(z_quantized.cpu(), save_path + '_token_emb.pt')
#             # torch.save(pos_indices.cpu(), save_path + '_id.pt')
#             # save_tokens(config, epoch, z_encoder, z_quantized, pos_indices)
#             print(f'Epoch: {epoch:03d}, Acc encoder: {acc_encoder:.4f}, Acc quantized: {acc_quantized:.4f}, Acc indices: {acc_indices:.4f}, Acc indices ppr: {acc_indices_ppr:.4f}, Acc input: {acc_input:.4f}')

# def search(config):
#     from ray import tune
#     from ray.tune.search.optuna import OptunaSearch

#     config['gnn_type'] = tune.choice(['gcn']) # , 'graphsage', 'gin', 'gat', 'pna'
#     config['gnn_hidden_dim'] = tune.choice([128, 256, 512, 1024])
#     config['gnn_num_layers'] = tune.choice([1,2,3,4])

#     # config['vq_dim'] = tune.choice([32, 64, 128, 256, 512, 1024])
#     config['vq_codebook_size'] = tune.choice([64, 128, 256, 512, 1024])
#     config['vq_num_codebook'] = tune.choice([1])

#     config['lr_tokenizer'] = tune.choice([1e-4, 1e-3, 1e-2, 1e-1])
#     # config['epochs_tokenizer'] = tune.choice(list(range(200, 1000, 100)))

#     analysis = tune.run(
#         main,
#         num_samples=100,
#         resources_per_trial={"cpu": 16, "gpu": 0.5},
#         search_alg=OptunaSearch(mode='max'),
#         config=config,)
#     print(analysis)

if __name__ == '__main__':
    config = parse_args().__dict__
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['vq_dim'] = config['gnn_hidden_dim']
    if config['search']:
        search(config)
    else:
        main(config)