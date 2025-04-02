import torch
import torch.nn as nn
from tqdm import tqdm
import torch_geometric
from typing import Union
import torch.nn.functional as F
from dataclasses import dataclass
from models.mlp import MLPConfig, MLP
from models.gnn import GNNConfig, GNN
from torch_geometric.nn import DMoNPooling
from vector_quantize_pytorch import VectorQuantize, ResidualVQ


@dataclass
class RQVAEConfig:
    encoder_type: str
    decoder_type: str
    encoder_config: Union[MLPConfig, GNNConfig]
    decoder_config: Union[MLPConfig, GNNConfig]
    node_feature: str
    quantization: str
    feature_loss_type: str
    edge_loss_type: str
    rec_feature: bool
    rec_edge: bool
    cluster_loss: bool
    num_cluster: int
    decay: float
    num_codebook: int 
    codebook_size: int
    codebook_dim: int
    commitment_weight: float
    edge_rec_loss_weight: float
    feature_rec_loss_weight: float

class RQVAE(nn.Module):
    def __init__(self, config: RQVAEConfig):
        super().__init__()
        self.config = config

        if isinstance(config.encoder_config, GNNConfig):
            self.encoder = GNN(config.encoder_config)
        elif isinstance(config.encoder_config, MLPConfig):
            self.encoder = MLP(config.encoder_config)

        if config.quantization == 'vq':
            self.vq = VectorQuantize(
                dim = config.codebook_dim,
                codebook_size = config.codebook_size,  # codebook size
                decay = config.decay,                  # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = config.commitment_weight,     # the weight on the commitment loss
                kmeans_init = True,
                kmeans_iters = 10
            )
        elif config.quantization == 'rq':
            self.vq = ResidualVQ(
                dim = config.codebook_dim,
                num_quantizers = config.num_codebook,     # specify number of quantizers
                codebook_size = config.codebook_size,  # codebook size
                kmeans_init = True,
                kmeans_iters = 10,
                shared_codebook = False,
            )
        
        if config.rec_feature:
            if config.decoder_type == 'gnn':
                self.decoder_node = GNN(config.decoder_config)
            elif config.decoder_type == 'mlp':
                self.decoder_node = MLP(config.decoder_config)
        if config.rec_edge:
            if config.decoder_type == 'gnn':
                self.decoder_edge = GNN(config.decoder_config)
            elif config.decoder_type == 'mlp':
                self.decoder_edge = MLP(config.decoder_config)

        if config.cluster_loss:
            self.pool = DMoNPooling(config.encoder_config.out_dim, config.num_cluster)

    def forward(self, batch):
        if self.config.encoder_type == 'gnn':
            if self.config.node_feature == 'sign':
                out = self.encoder(batch.sign_feat, batch.edge_index)
            elif self.config.node_feature == 'feature':
                out = self.encoder(batch.x, batch.edge_index)
            elif self.config.node_feature == 'llm':
                out = self.encoder(batch.llm_embedding, batch.edge_index)
        elif self.config.encoder_type == 'mlp':
            if self.config.node_feature == 'sign':
                out = self.encoder(batch.sign_feat)
            elif self.config.node_feature == 'feature':
                out = self.encoder(batch.x)
        quantized, indices, commit_loss, __ = self.vq(out, return_all_codes=True)
        
        if self.config.quantization == 'rq':
            commit_loss = torch.sum(commit_loss)
        
        loss = 0 + commit_loss

        edge_rec_loss = feature_rec_loss = cluster_loss = torch.tensor(0)
        
        if self.config.cluster_loss:
            adj = torch_geometric.utils.to_torch_sparse_tensor(batch.edge_index, size=batch.num_nodes).to_dense()
            s, _, _, sp, o, c = self.pool(torch.unsqueeze(out, 0), adj)
            cluster_loss = sp + o + c
            loss += cluster_loss
            # s as kind of positional encoding 
        quantized_node, quantized_edge = None, None
        if self.config.rec_edge:
            if self.config.decoder_type == 'gnn':
                quantized_edge = self.decoder_edge(quantized, batch.edge_index)
            elif self.config.decoder_type == 'mlp':
                quantized_edge = self.decoder_edge(quantized)
            adj_quantized = torch.matmul(quantized_edge, quantized_edge.t())
            adj = torch_geometric.utils.to_torch_sparse_tensor(batch.edge_index, size=batch.num_nodes).to_dense()
            edge_rec_loss = self.compute_edge_loss(adj_quantized, adj)
            loss += self.config.edge_rec_loss_weight * edge_rec_loss
        
        if self.config.rec_feature:
            if self.config.decoder_type == 'gnn':
                quantized_node = self.decoder_node(quantized, batch.edge_index)
            elif self.config.decoder_type == 'mlp':
                quantized_node = self.decoder_node(quantized)
            if self.config.node_feature == 'sign': 
                feature_rec_loss = self.compute_feature_loss(batch.sign_feat, quantized_node)
            elif self.config.node_feature == 'feature':
                feature_rec_loss = self.compute_feature_loss(batch.x, quantized_node)
            elif self.config.node_feature == 'llm':
                feature_rec_loss = self.compute_feature_loss(batch.llm_embedding, quantized_node)
            loss += self.config.feature_rec_loss_weight * feature_rec_loss
        # loss = F.cross_entropy(out, y)
        return out, quantized_node, indices, loss, commit_loss, feature_rec_loss, edge_rec_loss, cluster_loss

    def compute_feature_loss(self, yt, yp):
        if self.config.feature_loss_type == 'l2':
            return F.mse_loss(yp, yt)
        elif self.config.feature_loss_type == 'l1':
            return F.l1_loss(yp, yt)
        elif self.config.feature_loss_type == 'cosine':
            return 1 - F.cosine_similarity(yp, yt, dim=-1).mean()
    
    def compute_edge_loss(self, yt, yp):
        if self.config.edge_loss_type == 'ce':
            return F.binary_cross_entropy_with_logits(yt, yp)

    @torch.no_grad()
    def inference(self, loader, device):
        x_all = []
        quantized_all = []
        indices_all = []
        for batch in tqdm(loader, disable=True):
            if self.config.encoder_type == 'gnn':
                if self.config.node_feature == 'sign':
                    out = self.encoder(batch.sign_feat.to(device), batch.edge_index.to(device))
                elif self.config.node_feature == 'feature':
                    out = self.encoder(batch.x.to(device), batch.edge_index.to(device))
                elif self.config.node_feature == 'llm':
                    out = self.encoder(batch.llm_embedding.to(device), batch.edge_index.to(device))
            elif self.config.encoder_type == 'mlp':
                if self.config.node_feature == 'sign':
                    out = self.encoder(batch.sign_feat.to(device))
                elif self.config.node_feature == 'feature':
                    out = self.encoder(batch.x.to(device))
            quantized, indices, _, all_codes = self.vq(out, return_all_codes=True)
            x_all.append(out)
            quantized_all.append(quantized)
            indices_all.append(indices.cpu())
        return torch.cat(x_all, dim=0), torch.cat(quantized_all, dim=0), torch.cat(indices_all, dim=0), all_codes
