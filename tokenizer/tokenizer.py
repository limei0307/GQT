import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from tokenizer.vq_bgrl import Tokenizer as BGRL_Tokenizer
from tokenizer.vq_dgi import Tokenizer as DGI_Tokenizer
from tokenizer.vq_gae import Tokenizer as GAE_Tokenizer
from tokenizer.vq_h import Tokenizer 


def save_tokens(config, epoch, encoder_out, quantized, indices):
    path = f"{osp.join(config['root'], config['tokenizer_input'])}_"
    path+= f"{config['vq_encoder']}_{config['vq_decoder']}_{config['vq_decoder']}_{config['gnn_type']}_"
    path+= f"{config['vq_dim']}_{config['vq_codebook_size']}_{config['vq_num_codebook']}_"
    path+= f"{config['rec_feature']}_{config['rec_edge']}_{config['cluster_loss']}_{epoch}"
    torch.save(encoder_out.detach().cpu(), f'{path}_encoder_emb.pt')
    torch.save(quantized.detach().cpu(), f'{path}__token_emb.pt')
    torch.save(indices.detach().cpu(), f'{path}__id.pt')

def load_tokens(config):
    encoder_out = torch.load(f"{config['save_path']}_encoder_emb.pt")
    quantized = torch.load(f"{config['save_path']}_token_emb.pt")
    indices = torch.load(f"{config['save_path']}_id.pt")
    return encoder_out, quantized, indices

def fit_logistic_regression(X, y, train_masks, val_masks, test_mask):
    from sklearn import metrics
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, normalize

    y = y.detach().cpu().numpy()
    X = X.detach().cpu().numpy()

    one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)
    X = normalize(X, norm='l2')

    accuracies = []
    splits = train_masks.shape[1] if train_masks.dim() == 2 else 1
    for split_id in range(splits):
        if train_masks.dim() == 2:
            train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]
        else:
            train_mask, val_mask = train_masks, val_masks
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(best_test_acc)
    return accuracies

def pretrain_tokenizer(config, data, train_loader, test_loader, train_idx, val_idx, test_idx):
    if config['ssl_method'] == 'ae':
        tokenizer = GAE_Tokenizer(config)
    elif config['ssl_method'] == 'dgi':
        tokenizer = DGI_Tokenizer(config)
    elif config['ssl_method'] == 'bgrl':
        tokenizer = BGRL_Tokenizer(config)
    
    tokenizer = Tokenizer(config)
    
    if config['save_path'] == 'none' and ('token_id' in config['transformer_input'] or 'quantized_emb' in config['transformer_input'] or 'encoder_emb' in config['transformer_input']):
        early_stop = 0
        pre_loss = 1000
        epoch_iter = tqdm(range(1, config['epochs_tokenizer'] + 1))
        for epoch in epoch_iter:
            tokenizer.model.train()
            losses = tokenizer.train(train_loader, epoch)
            loss = losses['loss']
            tokenizer.model.eval()
            indices, __ = tokenizer.encode(test_loader)
            epoch_iter.set_description(f"Tokenizer | Epoch: {epoch} | Loss: {losses['loss']:.4f}")
            if config['verbose']: 
                print(f'Epoch: {epoch} | ' + ' | '.join([f'{k} : {v:.4f}' for k, v in losses.items()]) + f' | Tokens: {len(set(map(tuple, indices.cpu().numpy())))}')
            if loss > pre_loss:
                early_stop += 1
            else:
                early_stop = 0
            pre_loss = loss
            if early_stop == config['early_stop_epochs']:
                if config['verbose']:
                    print('early stopping')
                break
            # if epoch % config['save_epochs'] == 0: 
            #     tokenizer.model.eval()
            #     encoder_out, quantized, indices, all_codes, codebooks = tokenizer.encode(test_loader)
                # save_tokens(config, epoch, encoder_out, quantized, indices)
        tokenizer.model.eval()
        indices, codebooks = tokenizer.encode(test_loader)
        # save_tokens(config, epoch, encoder_out, quantized, indices)
    else:
        encoder_out, quantized, indices = load_tokens(config)
    
    # accuracy = 0.
    # if config['linear_probe']:
    #     accuracy = fit_logistic_regression(quantized if quantized.dim()==2 else quantized.view(-1, (config['num_sign_hops'] + 1) * config['vq_dim']), 
    #                                        data.y, train_idx, val_idx, test_idx)
    #     print(f'Linear accuracy: {accuracy[0]*100:.2f}%')
    
    return indices, codebooks