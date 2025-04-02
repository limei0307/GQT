import torch.nn as nn
from dataclasses import dataclass, field
from models.utils import create_activation, create_norm

@dataclass
class MLPConfig:
    layer_dims: list #= field(default_factory=lambda: [512, 256, 128])
    in_dim: int
    out_dim: int
    dropout: float
    activation: str
    norm: str
    skip: bool


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropout = nn.Dropout(config.dropout)

        self.linear_in = nn.Linear(self.config.in_dim, config.layer_dims[0])
        self.linear_out = nn.Linear(config.layer_dims[-1], config.out_dim)

        if config.skip:
            self.residual = nn.linear(self.config.in_dim, config.out_dim)        

        for idx in range(1, len(config.layer_dims)):
            self.layers.append(nn.Linear(config.layer_dims[idx-1], config.layer_dims[idx]))
            self.norms.append(create_norm(self.config.norm)(config.layer_dims[idx]))
            self.activations.append(create_activation(self.config.activation))

    def forward(self, x):
        h = self.dropout(self.linear_in(x))
        for idx in range(len(self.config.layer_dims) - 1):
            h = self.norms[idx](self.layers[idx](h))
            h = self.activations[idx](h)
            h = self.dropout(h)
        h = self.linear_out(h)
        if self.config.skip:
            h = h + self.residual(x)
        return h

if __name__ == '__main__':  
    pass