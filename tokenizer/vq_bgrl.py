import os
import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from data import get_data
from args import parse_args
from torch.optim import AdamW
from dataclasses import dataclass
from utils import set_random_seeds
from loader import get_tokenizer_loader
from torch.nn.functional import cosine_similarity
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from utils import get_graph_drop_transform, CosineDecayScheduler
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, Sequential

@dataclass
class RQVAEConfig:
    quantization: str
    decay: float
    num_codebook: int 
    codebook_size: int
    codebook_dim: int
    commitment_weight: float

class GCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, batchnorm_mm=0.99, layernorm=False, weight_standardization=False):
        super().__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'),)

            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))

            layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index', layers)

    def forward(self, data):
        if self.weight_standardization:
            self.standardize_weights()
        return self.model(data.x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor, config):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor
        # self.code_projector = nn.Linear(config.codebook_dim, )
        self.config = config

        if config.quantization == 'vq':
            self.quantizer = VectorQuantize(
                dim = config.codebook_dim,
                codebook_size = config.codebook_size,  # codebook size
                decay = config.decay,                  # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = config.commitment_weight,     # the weight on the commitment loss
                kmeans_init = True,
                kmeans_iters = 10
            )
        elif config.quantization == 'rq':
            self.quantizer = ResidualVQ(
                dim = config.codebook_dim,
                num_quantizers = config.num_codebook,     # specify number of quantizers
                codebook_size = config.codebook_size,  # codebook size
                kmeans_init = True,
                kmeans_iters = 10,
                shared_codebook = False,
            )

        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.reset_parameters()
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters()) + list(self.quantizer.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x)
        quantized_online, indices, commit_loss, __ = self.quantizer(online_y, return_all_codes=True)
        
        if self.config.quantization == 'rq':
            commit_loss = torch.sum(commit_loss)

        # prediction
        online_q = self.predictor(quantized_online)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
            quantized_targt, __, __, __ = self.quantizer(target_y, return_all_codes=True)

        return online_q, quantized_targt.detach(), commit_loss
    
    @torch.no_grad()
    def encode(self, data):
        x = self.online_encoder(data)
        quantized, indices, __, all_codes = self.quantizer(x, return_all_codes=True)
        codebooks = torch.cat([layer.codebook for layer in self.quantizer.layers])
        return x, quantized, indices, all_codes, codebooks

class Tokenizer:
    def __init__(self, config):
        self.config = config
        tokenizer_config = RQVAEConfig(quantization=config['vq_method'], decay=config['vq_decay'], num_codebook=config['vq_num_codebook'], 
                                       codebook_size=config['vq_codebook_size'], codebook_dim=config['vq_dim'], commitment_weight=config['vq_weight'])

        input_size, representation_size = config['num_features'], config['vq_dim']
        encoder = GCN([input_size] + config['gnn_num_layers']*[config['gnn_hidden_dim']] + [config['vq_dim']], batchnorm=config['gnn_norm']=='batchnorm', layernorm=config['gnn_norm']=='layernorm')   # 512, 256, 128
        predictor = MLP_Predictor(representation_size, representation_size, hidden_size=config['gnn_hidden_dim'])
        self.model = BGRL(encoder, predictor, tokenizer_config).to(config['device'])
        if config['compile']:
            self.model = torch.compile(self.model)
        if config['verbose']:
            print(f'Total params of tokenizer: {sum(p.numel() for p in self.model.parameters())}')
        
        self.transform_1 = get_graph_drop_transform(drop_edge_p=0.2, drop_feat_p=0.2)
        self.transform_2 = get_graph_drop_transform(drop_edge_p=0.3, drop_feat_p=0.1)

        self.optimizer = AdamW(self.model.trainable_parameters(), lr=config['lr_tokenizer'], weight_decay=config['weight_decay_tokenizer'])
        self.lr_scheduler = CosineDecayScheduler(config['lr_tokenizer'], config['warmup_tokenizer'], config['epochs_tokenizer'])
        self.mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, config['epochs_tokenizer'])
    
    def train(self, loader, epoch):
        self.model.train()

        # update learning rate
        lr = self.lr_scheduler.get(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - self.mm_scheduler.get(epoch)

        total_loss, total_commit_loss, total_sim_loss = [], [], []
        for batch in tqdm(loader, disable=True):
            batch = batch.to(self.config['device'])
        # forward
            self.optimizer.zero_grad(set_to_none=True)

            x1, x2 = self.transform_1(batch), self.transform_2(batch)
            q1, y2, commit_loss1 = self.model(x1, x2)
            q2, y1, commit_loss2 = self.model(x2, x1)

            loss_sim = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            loss_commit = (commit_loss1 + commit_loss2) / 2
            loss = loss_sim + loss_commit
            loss.backward()

            # update online network
            self.optimizer.step()
            # update target network
            self.model.update_target_network(mm)
            
            total_loss.append(loss.item())
            total_commit_loss.append(loss_commit.item())
            total_sim_loss.append(loss_sim.item())

        return {'loss': np.mean(total_loss), 'commit_loss': np.mean(total_commit_loss), 'sim_loss': np.mean(total_sim_loss)}

    @torch.no_grad()
    def encode(self, loader):
        x_all, quantized_all, indices_all = [], [], []
        self.model.eval()
        for batch in tqdm(loader, disable=True):
            batch = batch.to(self.config['device'])
            x, quantized, indices, all_codes, codebooks = self.model.encode(batch)
            x_all.append(x)
            quantized_all.append(quantized)
            indices_all.append(indices)
        return torch.cat(x_all, dim=0), torch.cat(quantized_all, dim=0), torch.cat(indices_all, dim=0), all_codes, codebooks
    


def main(config):
    import warnings
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings("ignore")

    if config['verbose']:
        print('================= print config =================')
        for k, v in config.items():
            print(k, v)
        print('================================================')
    
    tokenizer_config = RQVAEConfig(quantization=config['vq_method'], decay=config['vq_decay'], num_codebook=config['vq_num_codebook'], 
                                codebook_size=config['vq_codebook_size'], codebook_dim=config['vq_dim'], commitment_weight=config['vq_weight'])

    data, pred_train_loader, pred_val_loader, pred_test_loader, train_idx, val_idx, test_idx = get_data(config)

    input_size, representation_size = data.x.size(1), config['vq_dim']
    encoder = GCN([input_size] + [512, 256, config['vq_dim']], batchnorm=True)   # 512, 256, 128
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=512)
    tokenizer = BGRL(encoder, predictor, tokenizer_config).to(config['device'])
    if config['compile']:
        tokenizer = torch.compile(tokenizer)
    if config['verbose']:
        print(f'Total params of tokenizer: {sum(p.numel() for p in tokenizer.parameters())}')
    
    tokenizer_train_loader, tokenizer_test_loader = get_tokenizer_loader(config, data)

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=0.2, drop_feat_p=0.2)
    transform_2 = get_graph_drop_transform(drop_edge_p=0.3, drop_feat_p=0.1)

    optimizer = AdamW(tokenizer.trainable_parameters(), lr=5e-4, weight_decay=1e-5)
    lr_scheduler = CosineDecayScheduler(5e-4, 1000, 10000)
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, 10000)


    def train(step):
        tokenizer.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad(set_to_none=True)

        x1, x2 = transform_1(data), transform_2(data)
        q1, y2, commit_loss1 = tokenizer(x1, x2)
        q2, y1, commit_loss2 = tokenizer(x2, x1)

        loss_sim = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        loss_commit = (commit_loss1 + commit_loss2) / 2
        loss = loss_sim + loss_commit
        loss.backward()

        # update online network
        optimizer.step()
        # update target network
        tokenizer.update_target_network(mm)
        return loss_sim, loss_commit, loss
        
    def eval():
        tokenizer.eval()
        x, quantized, indices, all_codes = tokenizer.encode(data)
        return x, quantized, indices, all_codes
                
    epoch_iter = tqdm(range(1, config['epochs_tokenizer'] + 1))
    for epoch in epoch_iter:
        loss_sim, loss_commit, loss = train(epoch-1)
        __, __, indices, __ = eval(epoch)
        epoch_iter.set_description(f"Tokenizer | Epoch: {epoch} | Loss: {loss:.4f}")
        if config['verbose']:
            print(f'Epoch: {epoch} | Loss: {loss:.4f} | Commitment loss: {loss_commit:.4f} | Similarity loss: {loss_sim:.4f} | Tokens: {len(set(map(tuple, indices.numpy())))}')

    # # save encoder weights
    # torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, 'bgrl-wikics.pt'))


if __name__ == '__main__':
    set_random_seeds(random_seed=123)
    config = parse_args().__dict__
    config['rec_feature'] = True
    # config['rec_edge'] = True
    # # config['llm_embedding'] = True
    # # config['tokenizer_input'] = 'llm'
    config['dataset'] = 'wikics'
    config['epochs_tokenizer'] = 10000
    # config['epochs_masked_transformer'] = 0
    # config['degree_encoding'] = True
    # config['position_encoding'] = True
    # config['hierarchy_encoding'] = True
    # config['gnn_skip'] = True
    config['sequence_type'] = 'dfs'
    # config['cluster_loss'] = True
    config['linear_probe'] = True
    # config['scheduler_tokenizer'] = True
    # config['scheduler_transformer'] = True
    # config['scheduler_masked_transformer'] = True
    # config['epochs_tokenizer'] = 2
    config['verbose'] = True

    config['root'] = os.path.abspath(config['root'])
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if config['search']:
    #     search(config)
    # else:
    main(config)