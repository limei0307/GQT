import torch.nn as nn
from dataclasses import dataclass, field
from models.utils import create_activation, create_norm
from torch_geometric.nn import GINConv, GCNConv, GraphSAGE, GATConv

@dataclass
class GNNConfig:
    layer_type: str
    in_dim: int
    num_layers: int
    hidden_dim: int
    out_dim: int
    num_heads: int
    activation: str
    norm: str
    dropout: float
    skip: bool


class GNN(nn.Module):
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        if config.layer_type == 'graphsage':
            self.model = GraphSAGE(config.in_dim, config.hidden_dim, config.num_layers, config.out_dim, config.dropout, config.activation, config.norm)
        else:
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.dropout = nn.Dropout(config.dropout)
            self.linear_in = nn.Linear(config.in_dim, config.hidden_dim)
            self.linear_out = nn.Linear(config.hidden_dim, config.out_dim)
            
            for idx in range(config.num_layers):
                if self.config.layer_type == 'gin':
                    self.layers.append(GINConv(nn.Sequential(nn.Linear(config.hidden_dim, 2 * config.hidden_dim), 
                                            nn.LeakyReLU(), 
                                            nn.Linear(2 * config.hidden_dim, config.hidden_dim))))
                elif self.config.layer_type == 'gcn':
                    self.layers.append(GCNConv(config.hidden_dim, config.hidden_dim))
                elif self.config.layer_type == 'gat':
                    self.layers.append(GATConv(config.hidden_dim, config.hidden_dim, config.num_heads, concat=False))                    
                self.norms.append(create_norm(self.config.norm)(config.hidden_dim))
                self.activations.append(create_activation(self.config.activation))

    def forward(self, x, edge_index):
        if self.config.layer_type == 'graphsage':
            h = self.model(x, edge_index)
        else:
            h = self.linear_in(x)
            h = self.dropout(h)
            for idx in range(self.config.num_layers):
                x = h
                h = self.layers[idx](h, edge_index)
                h = self.norms[idx](h)
                h = self.activations[idx](h)
                h = self.dropout(h)
                if self.config.skip:
                    h = h + x
            h = self.linear_out(h)
        return h


if __name__ == '__main__':  
    pass