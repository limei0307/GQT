import os
import torch
import numpy as np
from tqdm import tqdm
from data import get_data
from args import parse_args
import torch.nn.functional as F
from models.transformer import TransformerEncoder 
# from tokenizer.tokenizer import pretrain_tokenizer
from loader import get_tokenizer_loader, get_transformer_loader, get_tokenizer_loader_sign
from utils import setup_config, CosineDecayScheduler, set_random_seeds
from sklearn.metrics import accuracy_score
from vq_gmae2_large import main_vq_gmae2_large
from vq_gmae2_full_batch import main_vq_gmae2_full_batch
    
def train_masked_transformer(model, loader, data, optimizer, device):
    model.train()
    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, disable=True)):
        optimizer.zero_grad(set_to_none=True)
        sequence, degree, distance, weight, transformer_input, _, _ = batch
        degree = degree.to(device)
        distance = distance.to(device)
        weight = weight.to(device)

        mask = (torch.rand(sequence.shape, device=device) < config['mask_probability'])
        input_id = data.indices[sequence].to(device).reshape(len(sequence), -1) # (batch, max_sequence_length * num_codebook)

        # mask = torch.rand(input_id.shape, device=device) > 0.85
        # input_id[mask] = config['vq_codebook_size'] * config['vq_num_codebook']

        if config['transformer_input'] == 'token_id':
            batch_data = (input_id, None, mask, degree, distance, weight)
        elif 'token_id' in config['transformer_input']:
            batch_data = (input_id, transformer_input.to(device), mask, degree, distance, weight)
        else:
            batch_data = (None, transformer_input.to(device), mask, degree, distance, weight)
        
        token_pred, __, mask = model(batch_data)
        y = data.indices[sequence].reshape(len(sequence), -1).to(device).to(device)[mask]
        
        loss = F.cross_entropy(token_pred[mask].reshape(-1, token_pred.shape[-1]), y.reshape(-1))
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()

    return loss_accum

def train_transformer(model, loader, data, optimizer, config):
    device = config['device']
    model.train()
    loss_accum, correct_accum, count_accum = 0, 0, 0
    for step, batch in enumerate(tqdm(loader, disable=True)):
        sequence, degree, distance, weight, transformer_input, label, mask_label = batch
        degree = degree.to(device)

        if config['transformer_input'] == 'token_id':
            input_id = data.indices[sequence].reshape(len(sequence), -1).to(device) # (batch, max_sequence_length * num_codebook)
            batch_data = (input_id, None, None, degree, distance, weight)
        elif 'token_id' in config['transformer_input']:
            input_id = data.indices[sequence].reshape(len(sequence), -1).to(device) # (batch, max_sequence_length * num_codebook)
            batch_data = (input_id, transformer_input.to(device), None, degree, distance, weight) # .repeat_interleave(config['vq_num_codebook'], dim=1)
        else:
            batch_data = (None, transformer_input.to(device), None, degree, distance, weight)
        __, out, __ = model(batch_data)
        # if config['first_node_only']:
        #     out = out[:, 0] # first node is the root node
        #     y = label[:, 0].to(device).squeeze()
        # else:
        #     out = out[mask_label]
        #     y = label[mask_label].to(device)

        y = label[:, 0].to(device).squeeze()
        
        loss = F.cross_entropy(out, y)
        loss_accum += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        y_pred = out.argmax(dim=-1)
        correct_accum += y.eq(y_pred).float().sum().item()
        count_accum += len(y)

    return correct_accum/count_accum, loss_accum/len(loader)

@torch.no_grad()
def test_transformer(model, loader, data, config):
    device = config['device']
    yp, yt = [], []
    model.eval()
    loss_accum, correct_accum, count_accum = 0, 0, 0
    for step, batch in enumerate(tqdm(loader, disable=True)):
        sequence, degree, distance, weight, transformer_input, label, mask_label = batch
        degree = degree.to(device)
        
        if config['transformer_input'] == 'token_id':
            input_id = data.indices[sequence].reshape(len(sequence), -1).to(device)
            batch_data = (input_id, None, None, degree, distance, weight)
        elif 'token_id' in config['transformer_input']:
            input_id = data.indices[sequence].reshape(len(sequence), -1).to(device)
            batch_data = (input_id, transformer_input.to(device), None, degree, distance, weight)
        else:
            batch_data = (None, transformer_input.to(device), None, degree, distance, weight)
        __, out, __ = model(batch_data)
        y = label[:, 0].to(device).squeeze()
        loss = F.cross_entropy(out, y)
        loss_accum += loss.item()
        y_pred = out.argmax(dim=-1)
        # correct_accum += y.eq(y_pred).float().sum().item()
        # count_accum += len(y)
        yp.extend(y_pred.cpu().numpy().tolist())
        yt.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(np.array(yt), np.array(yp))
    # correct_accum/count_accum
    return acc, loss_accum/len(loader)

def main(config):
    import warnings
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings("ignore")

    config['warmup_tokenizer'] = config['epochs_tokenizer'] / 10
    config['warmup_transformer'] = config['epochs_transformer'] / 10

    if config['verbose']:
        print('================= print config =================')
        for k, v in config.items():
            print(k, v)
        print('================================================')

    data, pred_train_loader, pred_val_loader, pred_test_loader, train_idx, val_idx, test_idx = get_data(config)
    

    best_test_accs = []
    test_best_valid_accs = []
    for run in range(config['runs']):
        if config['verbose']:
            print(f'Run {run:02d}:')

        set_random_seeds()
        if config['ssl_method'] != 'gmae2':
            pass
            # tokenizer_train_loader, tokenizer_test_loader = get_tokenizer_loader_sign(config, data)
            # indices, codebooks = pretrain_tokenizer(config, data, tokenizer_train_loader, tokenizer_test_loader, train_idx, val_idx, test_idx)
        elif 'ogbn' in config['dataset']:
            embeddings_train_val_test, quantized_train_val_test, indices_train_val_test, codebooks = main_vq_gmae2_large() # need to reorder
            encoder_out = torch.zeros_like(embeddings_train_val_test)
            encoder_out[data.train_mask] = embeddings_train_val_test[:data.train_mask.sum()]
            encoder_out[data.val_mask] = embeddings_train_val_test[data.train_mask.sum():data.train_mask.sum() + data.val_mask.sum()]
            encoder_out[data.test_mask] = embeddings_train_val_test[data.train_mask.sum() + data.val_mask.sum():]

            if quantized_train_val_test:
                quantized = torch.zeros_like(quantized_train_val_test)
                quantized[data.train_mask] = quantized_train_val_test[:data.train_mask.sum()]
                quantized[data.val_mask] = quantized_train_val_test[data.train_mask.sum():data.train_mask.sum() + data.val_mask.sum()]
                quantized[data.test_mask] = quantized_train_val_test[data.train_mask.sum() + data.val_mask.sum():]
            else:
                quantized = None
            
            if indices_train_val_test:
                indices = torch.zeros_like(indices_train_val_test)
                indices[data.train_mask] = indices_train_val_test[:data.train_mask.sum()]
                indices[data.val_mask] = indices_train_val_test[data.train_mask.sum():data.train_mask.sum() + data.val_mask.sum()]
                indices[data.test_mask] = indices_train_val_test[data.train_mask.sum() + data.val_mask.sum():]
            else:
                indices = None
        else:
            main_vq_gmae2_full_batch(config, data, train_idx, val_idx, test_idx)
            encoder_out, quantized, indices = None, None, None
        
    #     if indices:
    #         for codebook in range(1, indices.shape[-1]):
    #             indices[:,codebook:] += config['vq_codebook_size']

    #     # Setup Transformer
    #     transformer_config = setup_config(config)
    #     transformer = TransformerEncoder(transformer_config, codebooks)
    #     transformer = transformer.to(config['device'])
    #     if config['compile']:
    #         transformer = torch.compile(transformer)
    #     if config['verbose']:
    #         print(f'Total params of transformer: {sum(p.numel() for p in transformer.parameters())}')

    #     masked_transformer_optimizer = torch.optim.AdamW(transformer.parameters(), lr=config['lr_masked_transformer'], weight_decay=config['weight_decay_masked_transformer'])
    #     masked_transformer_scheduler = CosineDecayScheduler(config['lr_masked_transformer'], config['warmup_masked_transformer'], config['epochs_masked_transformer'])

    #     data.indices = indices
    #     # pred_input: N*D
    #     if config['llm_embedding'] and 'llm_emb' in config['transformer_input']:
    #         data.transformer_input = data.llm_embedding
    #     elif 'quantized_emb' in config['transformer_input']:
    #         data.transformer_input = quantized
    #     elif 'encoder_emb' in config['transformer_input']:
    #         data.transformer_input = encoder_out
    #     elif 'input_emb' in config['transformer_input']:
    #         data.transformer_input = data.x
    #     elif 'sign_emb' in config['transformer_input']:
    #         data.transformer_input = data.sign_feat
    #     else:
    #         data.transformer_input = data.x
        
    #     pred_train_loader, pred_val_loader, pred_test_loader = get_transformer_loader(config, data, train_idx, val_idx, test_idx)
    #     epoch_iter = tqdm(range(1, config['epochs_masked_transformer'] + 1))
    #     for epoch in epoch_iter:
    #         if config['scheduler_masked_transformer']:
    #             lr = masked_transformer_scheduler.get(epoch)
    #             for param_group in masked_transformer_optimizer.param_groups:
    #                 param_group['lr'] = lr

    #         loss = train_masked_transformer(transformer, pred_train_loader, data, masked_transformer_optimizer, config['device'])
    #         epoch_iter.set_description(f"Masked Transformer | Epoch: {epoch} | Loss: {loss:.4f}")
    #         if config['verbose'] and config['scheduler_masked_transformer']:
    #             print(f'Epoch {epoch:03d} | loss: {loss:.4f} | LR: {lr:.6f}')
    #         elif config['verbose']:
    #             print(f'Epoch {epoch:03d} | loss: {loss:.4f}')

    #     transformer_optimizer = torch.optim.AdamW(transformer.parameters(), lr=config['lr_transformer'], weight_decay=config['weight_decay_transformer'])
    #     transformer_scheduler = CosineDecayScheduler(config['lr_transformer'], config['warmup_transformer'], config['epochs_transformer'])
    #     best_val_acc, test_best_val, best_test = .0, .0, .0
    #     early_stop, pre_loss = 0, 1000
    #     epoch_iter = tqdm(range(1, config['epochs_transformer'] + 1))

    #     for epoch in epoch_iter:
    #         if config['scheduler_transformer']:
    #             lr = transformer_scheduler.get(epoch)
    #             for param_group in transformer_optimizer.param_groups:
    #                 param_group['lr'] = lr

    #         train_acc, train_loss = train_transformer(transformer, pred_train_loader, data, transformer_optimizer, config['device'])
    #         val_acc, val_loss = test_transformer(transformer, pred_val_loader, data, config['device'])
    #         test_acc, test_loss = test_transformer(transformer, pred_test_loader, data, config['device'])

    #         # print(test_acc, test_loss)

    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             test_best_val = test_acc
    #         if test_acc > best_test:
    #             best_test = test_acc

    #         if val_loss > pre_loss:
    #             early_stop += 1
    #         else:
    #             early_stop = 0
    #         pre_loss = val_loss
    #         if early_stop == config['early_stop_epochs']:
    #             if config['verbose']:
    #                 print('early stopping')
    #             break
            
    #         epoch_iter.set_description(f"Transformer | Epoch: {epoch} | Train Loss: {train_loss:0.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Best Test: {test_best_val:.4f}")
    #         if config['verbose']:
    #             print(f'Epoch {epoch:03d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | Best val: {test_best_val:.4f} | Best Test: {best_test:.4f}')
    #     best_test_accs.append(best_test)
    #     test_best_valid_accs.append(test_best_val)

    # best_test_acc = torch.tensor(best_test_accs)
    # test_best_valid_acc = torch.tensor(test_best_valid_accs)

    # if config['verbose']:
    #     print('============================')
    #     print(best_test_acc)
    #     print(f'Final Best Test: {best_test_acc.mean():.4f} ± {best_test_acc.std():.4f}')
    #     print(test_best_valid_acc)
    #     print(f'Final Test Best Valid: {test_best_valid_acc.mean():.4f} ± {test_best_valid_acc.std():.4f}')
    # return test_best_val

def search(config):
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    # config['bfs_hops'] = tune.choice([2,3,4,5])
    # config['dfs_walk_length'] = tune.choice([10, 25, 50, 75])

    config['num_sign_hops'] = tune.choice(list(range(2, 5)))
    config['ppr_num_neighbors'] = tune.choice(list(range(20, 100, 10)))
    # config['scale_feature'] = tune.choice([True, False])
    # config['normalize_feature'] = tune.choice([True, False])
    # config['to_undirected'] = tune.choice([True, False])

    config['transformer_num_layers'] = tune.choice([1, 2, 3, 4])
    config['transformer_hidden_dim'] = tune.choice([128, 256])
    config['transformer_num_heads'] = tune.choice([4, 8, 16])
    config['transformer_dropout'] = tune.uniform(0.1, 0.7)
    config['transformer_ffn'] = tune.choice([128, 256, 512])

    config['mlp_dropout'] = tune.uniform(0.1, 0.7)
    # config['gnn_norm'] = tune.choice(['layernorm', 'batchnorm'])
    # config['gnn_hidden_dim'] = tune.choice([128, 256, 512, 1024])
    # config['gnn_num_layers'] = tune.choice([2, 3, 4, 5])
    # config['gnn_skip'] = tune.choice([True, False])

    config['vq_dim'] = tune.choice([32, 64, 128, 256, 512])
    config['vq_codebook_size'] = tune.choice([64, 128, 256, 512, 1024])
    # config['vq_num_codebook'] = tune.choice([2,3,4, 5])
    # config['rec_edge'] = tune.choice([True, False])
    # config['cluster_loss'] = tune.choice([True, False])
    # config['label_loss'] = tune.choice([True, False])

    config['degree_encoding'] = tune.choice([True, False])
    config['position_encoding'] = tune.choice([True, False])
    config['hierarchy_encoding'] = tune.choice([True, False])

    config['lr_tokenizer'] = tune.loguniform(1e-5, 1e-2)
    config['lr_transformer'] = tune.loguniform(1e-5, 1e-2)
    # config['lr_masked_transformer'] = tune.loguniform(1e-5, 1e-2)
    config['epochs_tokenizer'] = tune.choice(list(range(500, 1000, 100)))
    config['epochs_transformer'] = tune.choice(list(range(500, 1000, 100)))
    # config['epochs_masked_transformer'] = tune.choice(list(range(200, 1000, 100)))
    
    config['scheduler_tokenizer'] = tune.choice([True, False])
    config['scheduler_transformer'] = tune.choice([True, False])
    # config['scheduler_masked_transformer'] = tune.choice([True, False])

    # config['lr_masked_transformer'] = tune.loguniform(1e-10, 1e-5)
    config['weight_decay_transformer'] = tune.loguniform(1e-10, 1e-5)
    # config['weight_decay_masked_transformer'] = tune.loguniform(1e-10, 1e-5)

    analysis = tune.run(
        main,
        num_samples=100,
        resources_per_trial={"cpu": config['cpu'], "gpu": config['gpu']},
        search_alg=OptunaSearch(mode='max'),
        config=config,)
    print(analysis)

if __name__ == '__main__':
    config = parse_args().__dict__
    # config['rec_feature'] = True
    # config['rec_edge'] = True
    # # config['llm_embedding'] = True
    # config['tokenizer_input'] = 'sign'
    # config['dataset'] = 'wikics'
    # config['num_sign_hops'] = 2
    # config['epochs_tokenizer'] = 1
    # config['epochs_masked_transformer'] = 0
    # config['degree_encoding'] = True
    # config['position_encoding'] = True
    # config['hierarchy_encoding'] = True
    # config['gnn_skip'] = True
    # config['sequence_type'] = 'dfs'
    # config['cluster_loss'] = True
    # config['linear_probe'] = True
    # config['ssl_method'] = 'bgrl'
    # config['use_codebook'] = True
    # config['scheduler_tokenizer'] = True
    # config['scheduler_transformer'] = True
    # config['scheduler_masked_transformer'] = True
    # config['epochs_tokenizer'] = 2
    # config['verbose'] = True
    # config['compile'] = True

    # config['normalize_feature'] = True     
    # config['scale_feature'] = True
    # config['to_undirected'] = True
    # config['gnn_norm'] = 'layernorm'
    # config['gnn_dropout'] = 0.448423
    # config['gnn_num_layers'] = 4
    # config['gnn_hidden_dim'] = 128
    # config['gnn_skip'] = True
    # config['vq_dim'] = 32
    # config['vq_codebook_size'] = 64   
    # config['vq_num_codebook'] = 2  
    # config['rec_edge'] = True
    # config['cluster_loss'] = False    
    # config['epochs_tokenizer'] = 1600
    # config['lr_tokenizer'] = 2.2979e-05
    # config['scheduler_tokenizer'] = False


    config['root'] = os.path.abspath(config['root'])
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config['search']:
        search(config)
    else:
        main(config)
