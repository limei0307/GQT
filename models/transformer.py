
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from models.utils import create_activation, compute_positional_encoding

@dataclass
class TransformerConfig:
    dict_size: int
    num_layers: int
    in_dim: int
    ffn_dim: int
    hidden_dim: int
    num_classes: int
    num_heads: int
    activation: str
    dropout: float
    num_token_each_node: int 
    node_len: int # sequence should be num_token_each_node * node_len
    max_degree: int
    max_distance: int
    use_codebook: bool
    degree_encoding: bool
    distance_encoding: bool
    hierarchy_encoding: bool
    position_encoding: bool
    aggr_tokens: str
    aggr_neighbors: str
    use_gating: bool


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig, codebooks: torch.Tensor):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.sequence_length = config.num_token_each_node * config.node_len
        self.activation = create_activation(config.activation)
        self.dropout = nn.Dropout(config.dropout)

        self.linear_feature = nn.Linear(config.in_dim, config.hidden_dim)
        
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.num_heads,
            dim_feedforward=config.ffn_dim, dropout=config.dropout,
            activation=config.activation, norm_first=True, batch_first=True), 
            config.num_layers)

        self.out_norm = nn.LayerNorm(config.hidden_dim)
        
        # self.decoder = nn.Linear(config.hidden_dim, config.dict_size) # for masked token prediction

        if config.aggr_tokens == 'cat':
            classifier_input_dim = config.num_token_each_node * config.hidden_dim
        else:
            classifier_input_dim = config.hidden_dim
        if config.aggr_neighbors == 'attn':
            self.aggr_attn = nn.Linear(classifier_input_dim, 1)

        self.classifier = nn.Sequential(nn.Linear(classifier_input_dim, int(classifier_input_dim/2)),
                                        nn.PReLU(),
                                        nn.Linear(int(classifier_input_dim/2), self.config.num_classes))

        self.learnable_mask = nn.Parameter(torch.rand(1, config.hidden_dim))

        if config.use_codebook:
            self.id_emb = nn.Embedding(num_embeddings=config.dict_size, embedding_dim=codebooks.shape[-1]).from_pretrained(codebooks, freeze=False) # add 1 for masked token
            self.id_projection = nn.Linear(codebooks.shape[-1], config.hidden_dim)
        else:
            self.id_emb = nn.Embedding(num_embeddings=config.dict_size, embedding_dim=config.hidden_dim) # add 1 for masked token
            self.id_projection = nn.Identity()

        if self.config.position_encoding:
            self.position_table = torch.arange(self.config.node_len).unsqueeze(0).repeat_interleave(self.config.num_token_each_node, dim=1)
            self.position_encoding = nn.Embedding(self.sequence_length, config.hidden_dim).from_pretrained(compute_positional_encoding(config.hidden_dim, self.sequence_length), freeze=False)
        if self.config.hierarchy_encoding:
            self.hierarchy_table = torch.arange(self.config.num_token_each_node).unsqueeze(0).repeat_interleave(self.config.node_len, dim=0).view(-1, self.sequence_length)
            self.hierarchy_encoding = nn.Embedding(config.num_token_each_node, config.hidden_dim).from_pretrained(compute_positional_encoding(config.hidden_dim, config.num_token_each_node), freeze=False)
        
        if self.config.degree_encoding:
            self.degree_encoding = nn.Embedding(config.max_degree + 1, config.hidden_dim).from_pretrained(compute_positional_encoding(config.hidden_dim, config.max_degree + 1), freeze=False)
        if self.config.distance_encoding:
            self.distance_encoding = nn.Embedding(config.max_distance, config.hidden_dim).from_pretrained(compute_positional_encoding(config.hidden_dim, config.max_distance), freeze=False)
        
    def forward(self, batch_data):
        
        x, f, mask, degree, distances, weight = batch_data 
        # x: node id (B, num_token_each_node * node_len)
        # f: node feature (B, num_token_each_node, feature_dim)
        # mask: mask (B, node_len)
        # degree: node degree (B, node_len)
        # distances: (B, node_len)
        
        if x != None and f != None:
            x = self.id_projection(self.id_emb(x))
            f = self.linear_feature(f)
            x = torch.cat((x.reshape(x.shape[0], self.config.node_len, -1, x.shape[-1]), f.unsqueeze(2)), dim=2).reshape(x.shape[0], self.config.num_token_each_node * self.config.node_len, x.shape[-1])
            if mask != None:
                mask = mask.repeat_interleave(self.config.num_token_each_node, dim=1)
        elif x != None and f == None:
            x = self.id_projection(self.id_emb(x))
            if mask != None:
                mask = mask.repeat_interleave(self.config.num_token_each_node, dim=1)
        elif x == None and f != None:
            x = self.linear_feature(f)
        
        if mask != None:
            x[mask] = self.learnable_mask

        if self.config.use_gating:
            x = x * weight

        if self.config.position_encoding:
            x = x + self.position_encoding(self.position_table.repeat_interleave(x.shape[0], dim=0).to(x.device))

        if self.config.hierarchy_encoding:
            x = x + self.hierarchy_encoding(self.hierarchy_table.repeat_interleave(x.shape[0], dim=0).to(x.device))

        if self.config.degree_encoding:
            x = x + self.degree_encoding(degree).repeat_interleave(self.config.num_token_each_node, dim=1).to(x.device)
        
        if self.config.distance_encoding:
            x = x + self.distance_encoding(distances).repeat_interleave(self.config.num_token_each_node, dim=1).to(x.device)
        
        h = self.encoder(x)
        h = self.out_norm(h)
        h = self.dropout(self.activation(h))

        # token_pred = self.decoder(h).reshape(x.shape[0], self.config.node_len, -1, self.config.dict_size)[:,:, :self.config.num_token_each_node-1, :].reshape(x.shape[0], -1, self.config.dict_size)
        # token_pred = self.decoder(h).reshape(x.shape[0], self.config.node_len * self.config.num_token_each_node, self.config.dict_size)
        token_pred = None
        
        if self.config.aggr_tokens == 'sum':
            h = h.reshape(h.shape[0], -1, self.config.num_token_each_node, h.shape[-1]).sum(dim=2)
        elif self.config.aggr_tokens == 'cat':
            h = h.reshape(h.shape[0], self.config.node_len, -1)

        if self.config.aggr_neighbors == 'attn':
                a = self.aggr_attn(h).squeeze()
                a = F.softmax(a, dim=-1)
                h = h * a.unsqueeze(-1)
                h = h.sum(dim=1)

        # if self.config.num_token_each_node > 1 and self.config.aggr_tokens != 'cat':
        #     h = h.reshape(h.shape[0], -1, self.config.num_token_each_node, h.shape[-1])
        #     h = h[:, 0, 0, :]
            # if self.config.aggr_tokens == 'sum':
            #     h = h.sum(dim=2) # sum num_token_each_node tokens for each node
            # elif self.config.aggr_tokens == 'mean':
            #     h = h.mean(dim=2) # sum num_token_each_node tokens for each node
            # elif self.config.aggr_tokens == 'max':
            #     h = h.max(dim=2)[0]
            # elif self.config.aggr_tokens == 'attn':
            #     a = self.aggr_attn(h).squeeze()
            #     a = F.softmax(a, dim=-1)
            #     h = h * a.unsqueeze(-1)
            #     h = h.sum(dim=2)

        pred = self.classifier(h)

        # if self.config.aggr_tokens == 'vote':
        #     pred = pred.max(dim=2)[0]
            
        return token_pred, pred, mask
