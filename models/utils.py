import math
import torch

def create_activation(name):
    if name == "relu":
        return torch.nn.ReLU()
    elif name == "gelu":
        return torch.nn.GELU()
    elif name == "prelu":
        return torch.nn.PReLU()
    elif name == "elu":
        return torch.nn.ELU()
    elif name == "silu":
        return torch.nn.SiLU()
    elif name is None:
        return torch.nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return torch.nn.LayerNorm
    elif name == "batchnorm":
        return torch.nn.BatchNorm1d
    else:
        return torch.nn.Identity
    
def compute_positional_encoding(d_model, max_len):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe.squeeze()