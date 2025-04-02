import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from utils import CosineDecayScheduler
from vector_quantize_pytorch import VectorQuantize


class Quantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.levels = self.config['num_sign_hops'] + 1

        # self.encoder = nn.Sequential(
        #         nn.Dropout(self.config['mlp_dropout']),
        #         nn.Linear(config['num_features'], 512),
        #         nn.ReLU(),
        #         nn.Linear(512, config['vq_dim'])
        # )

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(self.config['mlp_dropout']),
                nn.Linear(config['num_features'], 512),
                nn.ReLU(),
                nn.Linear(512, config['vq_dim'])
        )
            for __ in range(self.levels)
        ])

        # self.decoder = nn.ModuleList([
        #     nn.Linear(config['vq_dim'], config['num_features'])
        #     for __ in range(self.levels)
        # ])

        # self.decoder = nn.Linear(config['vq_dim'], config['num_features'])

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(self.config['mlp_dropout']),
                nn.Linear(config['vq_dim'], 512),
                nn.ReLU(),
                nn.Linear(512, config['num_features'])
        )
            for __ in range(self.levels)
        ])

        self.quantizer = nn.ModuleList([
            VectorQuantize(
                dim = config['vq_dim'],
                codebook_size = config['vq_codebook_size'],
                decay = config['vq_decay'],
                commitment_weight = config['vq_weight'],
                kmeans_init = True,
                kmeans_iters = 10
            )
            for __ in range(self.levels) 
        ])

    def forward(self, x):
        total_mse, total_commit = 0, 0 
        x = x.view(-1, self.levels, self.config['num_features'])
        # x = x + 0.1 * torch.randn_like(x)
        for idx in range(self.levels):
            h = self.encoder[idx](x[:, idx, :])
            quantized, __, commit_loss = self.quantizer[idx](h)
            x_hat = self.decoder[idx](quantized)
            total_mse += F.mse_loss(x_hat, x[:, idx, :], reduce=True)
            total_commit += commit_loss
        return total_mse/self.levels, commit_loss/self.levels
    
    @torch.no_grad()
    def encode(self, x):
        indices_all = []
        x = x.view(-1, self.levels, self.config['num_features'])
        for idx in range(self.levels):
            h = self.encoder[idx](x[:, idx, :])
            __, indices, __ = self.quantizer[idx](h)
            indices_all.append(indices)
        indices_all = torch.stack(indices_all).t()
        codebooks = torch.cat([self.quantizer[idx].codebook for idx in range(self.levels)])
        return indices_all, codebooks

class Tokenizer:
    def __init__(self, config):
        self.config = config
        self.model = Quantizer(config).to(config['device'])
        if config['compile']:
            self.model = torch.compile(self.model)
        self.optimizer = AdamW(self.model.parameters(), lr=config['lr_tokenizer'], weight_decay=config['weight_decay_tokenizer'])
        self.lr_scheduler = CosineDecayScheduler(config['lr_tokenizer'], config['warmup_tokenizer'], config['epochs_tokenizer'])
    
    def train(self, loader, epoch):
        self.model.train()

        # update learning rate
        lr = self.lr_scheduler.get(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        total_loss, total_commit_loss, total_sim_loss = [], [], []
        for batch in tqdm(loader, disable=True):
            x = batch.to(self.config['device'])
            sim_loss, commit_loss = self.model(x)
            loss = sim_loss * self.config['feature_rec_loss_weight'] + commit_loss 
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss.append(loss.item())
            total_commit_loss.append(commit_loss.item())
            total_sim_loss.append(sim_loss.item())

        return {'loss': np.mean(total_loss), 'commit_loss': np.mean(total_commit_loss), 'sim_loss': np.mean(total_sim_loss)}

    @torch.no_grad()
    def encode(self, loader):
        indices_all = []
        self.model.eval()
        for batch in tqdm(loader, disable=True):
            x = batch.to(self.config['device'])
            indices, codebooks = self.model.encode(x)
            indices_all.append(indices)
        return torch.cat(indices_all, dim=0), codebooks
        # return torch.cat(x_all, dim=0), torch.cat(quantized_all, dim=0), torch.cat(indices_all, dim=0), all_codes, codebooks