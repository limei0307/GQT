import copy
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gmae2_datasets.lc_sampler import setup_eval_dataloder, setup_finetune_dataloder, LinearProbingDataLoader
from gmae2_utils import accuracy, rocauc, set_random_seed, show_occupied_memory, get_current_lr

import wandb

eval_function = accuracy # rocauc

def transformer_predictor_func(
    config, data, train_idx, val_idx, test_idx, 
    model, graph, x, 
    feats, ego_graph_nodes, labels, 
    device, seeds=[0]
):
    logging.info("-- Transformer predictor in downstream tasks ---")
    if 'ogbn' in config['dataset']:
        train_ego_graph_nodes, val_ego_graph_nodes, test_ego_graph_nodes = ego_graph_nodes
        num_train, num_val = len(train_ego_graph_nodes), len(val_ego_graph_nodes)
        train_lbls, val_lbls, test_lbls = labels
        # if dataset_name in ["ogbn-papers100M", "mag-scholar-f", "mag-scholar-c","ogbn-arxiv","ogbn-products"]:
        # if dataset_name in ["ogbn-papers100M", "mag-scholar-f", "mag-scholar-c", "ogbn-arxiv", "ogbn-products"]:
        eval_loader = setup_eval_dataloder("lc", graph, feats, train_ego_graph_nodes+val_ego_graph_nodes+test_ego_graph_nodes, 512)

        with torch.no_grad():
            model.eval()
            embeddings = []
            quantized = []
            indices = []

            for batch in tqdm(eval_loader, desc="Infering...", disable=True):
                batch_g, targets, _, node_idx = batch
                batch_g = batch_g.to(device)
                x = batch_g.ndata.pop("feat")
                targets = targets.to(device)
                
                batch_emb, batch_emb_quantized, batch_indices = model.embed(batch_g, x)
                embeddings.append(batch_emb[targets].cpu())
                if batch_emb_quantized != None:
                    quantized.append(batch_emb_quantized[targets].cpu())
                if batch_indices != None:
                    indices.append(batch_indices[targets].cpu())
        
        embeddings_train_val_test = torch.cat(embeddings, dim=0)
        if batch_emb_quantized != None:
            quantized_train_val_test = torch.cat(quantized, dim=0)
        else:
            quantized_train_val_test = None
        if batch_indices != None:
            indices_train_val_test = torch.cat(indices, dim=0)
            codebooks = []
            for i in range(config['vq_num_codebook']):
                codebooks.append(model.vq.layers[i]._codebook.embed[0])
            codebooks = torch.cat(codebooks)
        else:
            indices_train_val_test = None
            codebooks = None
        # import pdb; pdb.set_trace()
        # torch.save(indices, 'arxiv_vq_ep60_200_indices_90941_29799.pt')
        # torch.save(quantized, 'arxiv_vq_ep60_200_quantized_90941_29799.pt')
        # torch.save(embeddings, 'arxiv_vq_ep60_200_embeddings_90941_29799.pt')

        # need to reorder
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
        model.eval()
        with torch.no_grad():
            encoder_out, quantized, indices = model.embed(graph.to(device), x.to(device))
            if indices != None:
                codebooks = []
                for i in range(config['vq_num_codebook']):
                    codebooks.append(model.vq.layers[i]._codebook.embed[0])
                codebooks = torch.cat(codebooks)
            else:
                codebooks = None
    if indices != None:
        for codebook in range(1, indices.shape[-1]):
            indices[:,codebook:] += config['vq_codebook_size']

    # Setup Transformer
    from models.transformer import TransformerEncoder
    from utils import setup_config, CosineDecayScheduler, set_random_seeds
    from loader import get_transformer_loader
    from main import train_transformer, test_transformer
    
    data.indices = indices
    # pred_input: N*D
    if config['llm_embedding'] and 'llm_emb' in config['transformer_input']:
        data.transformer_input = data.llm_embedding
    elif 'quantized_emb' in config['transformer_input']:
        data.transformer_input = quantized
    elif 'encoder_emb' in config['transformer_input']:
        data.transformer_input = encoder_out
    elif 'input_emb' in config['transformer_input']:
        data.transformer_input = data.x
    elif 'sign_emb' in config['transformer_input']:
        data.transformer_input = data.sign_feat
    else:
        data.transformer_input = data.x
    
    pred_train_loader, pred_val_loader, pred_test_loader = get_transformer_loader(config, data, train_idx, val_idx, test_idx)

    best_test_accs = []
    test_best_valid_accs = []
    for i,_ in enumerate(seeds):
        print(f"####### Run seed {seeds[i]} for Transformer predictor...")
        set_random_seed(seeds[i])
        transformer_config = setup_config(config)
        transformer = TransformerEncoder(transformer_config, codebooks)
        transformer = transformer.to(config['device'])
        transformer_optimizer = torch.optim.AdamW(transformer.parameters(), lr=config['lr_transformer'], weight_decay=config['weight_decay_transformer'])
        transformer_scheduler = CosineDecayScheduler(config['lr_transformer'], config['warmup_transformer'], config['epochs_transformer'])
        best_val_acc, test_best_val, best_test = .0, .0, .0
        early_stop, pre_loss = 0, 1000
        epoch_iter = tqdm(range(1, config['epochs_transformer'] + 1), disable=True)

        for epoch in epoch_iter:
            if config['scheduler_transformer']:
                lr = transformer_scheduler.get(epoch)
                for param_group in transformer_optimizer.param_groups:
                    param_group['lr'] = lr

            train_acc, train_loss = train_transformer(transformer, pred_train_loader, data, transformer_optimizer, config)
            val_acc, val_loss = test_transformer(transformer, pred_val_loader, data, config)
            test_acc, test_loss = test_transformer(transformer, pred_test_loader, data, config)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_best_val = test_acc
            if test_acc > best_test:
                best_test = test_acc

            if val_loss > pre_loss:
                early_stop += 1
            else:
                early_stop = 0
            pre_loss = val_loss
            if early_stop == config['early_stop_epochs']:
                if config['verbose']:
                    print('early stopping')
                break
            
            epoch_iter.set_description(f"Transformer | Epoch: {epoch} | Train Loss: {train_loss:0.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Best Test: {test_best_val:.4f}")
            if config['verbose']:
                print(f'Epoch {epoch:03d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | Best val: {test_best_val:.4f} | Best Test: {best_test:.4f}')
        best_test_accs.append(best_test)
        test_best_valid_accs.append(test_best_val)
    
    print(f"# final_best_test_acc (transformer predictor): {np.mean(best_test_accs):.4f}, std: {np.std(best_test_accs):.4f}")
    print(f"# final_test_best_val_acc (transformer predictor): {np.mean(test_best_valid_accs):.4f}, std: {np.std(test_best_valid_accs):.4f}")


def linear_probing_minibatch(
    model, graph,
    feats, ego_graph_nodes, labels, 
    lr_f, weight_decay_f, max_epoch_f, 
    device, batch_size=-1, shuffle=True, seeds=[0]):
    logging.info("-- Linear Probing in downstream tasks ---")
    train_ego_graph_nodes, val_ego_graph_nodes, test_ego_graph_nodes = ego_graph_nodes
    num_train, num_val = len(train_ego_graph_nodes), len(val_ego_graph_nodes)
    train_lbls, val_lbls, test_lbls = labels
    # if dataset_name in ["ogbn-papers100M", "mag-scholar-f", "mag-scholar-c","ogbn-arxiv","ogbn-products"]:
    # if dataset_name in ["ogbn-papers100M", "mag-scholar-f", "mag-scholar-c", "ogbn-arxiv", "ogbn-products"]:
    eval_loader = setup_eval_dataloder("lc", graph, feats, train_ego_graph_nodes+val_ego_graph_nodes+test_ego_graph_nodes, 512)

    with torch.no_grad():
        model.eval()
        embeddings = []
        quantized = []
        indices = []

        for batch in tqdm(eval_loader, desc="Infering...", disable=True):
            batch_g, targets, _, node_idx = batch
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            targets = targets.to(device)
            
            batch_emb, batch_emb_quantized, batch_indices = model.embed(batch_g, x)
            embeddings.append(batch_emb[targets].cpu())
            if batch_emb_quantized != None:
                quantized.append(batch_emb_quantized[targets].cpu())
            if batch_indices != None:
                indices.append(batch_indices[targets].cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    if batch_emb_quantized != None:
        quantized = torch.cat(quantized, dim=0)
    else:
        quantized = None
    if batch_indices != None:
        indices = torch.cat(indices, dim=0)
    else:
        indices = None
    # import pdb; pdb.set_trace()
    # torch.save(indices, 'arxiv_vq_ep60_200_indices_90941_29799.pt')
    # torch.save(quantized, 'arxiv_vq_ep60_200_quantized_90941_29799.pt')
    # torch.save(embeddings, 'arxiv_vq_ep60_200_embeddings_90941_29799.pt')
    
    if quantized != None:
        train_emb, val_emb, test_emb = quantized[:num_train], quantized[num_train:num_train+num_val], quantized[num_train+num_val:]

        batch_size = 5120
        acc = []
        for i,_ in enumerate(seeds):
            print(f"####### Run seed {seeds[i]} for LinearProbing...")
            set_random_seed(seeds[i])
            print(f"training sample:{len(train_emb)}")
            test_acc = node_classification_linear_probing(
                (train_emb, val_emb, test_emb), 
                (train_lbls, val_lbls, test_lbls), 
                lr_f, weight_decay_f, max_epoch_f, device, batch_size=batch_size, shuffle=shuffle, mute=True)
            acc.append(test_acc)

        print(f"# final_acc (linear prob on quantized emb): {np.mean(acc):.4f}, std: {np.std(acc):.4f}")
    
    if indices != None:
        indices = indices.float()
        train_emb, val_emb, test_emb = indices[:num_train], indices[num_train:num_train+num_val], indices[num_train+num_val:]

        batch_size = 5120
        acc = []
        for i,_ in enumerate(seeds):
            print(f"####### Run seed {seeds[i]} for LinearProbing...")
            set_random_seed(seeds[i])
            print(f"training sample:{len(train_emb)}")
            test_acc = node_classification_linear_probing(
                (train_emb, val_emb, test_emb), 
                (train_lbls, val_lbls, test_lbls), 
                lr_f, weight_decay_f, max_epoch_f, device, batch_size=batch_size, shuffle=shuffle, mute=True)
            acc.append(test_acc)

        print(f"# final_acc (linear prob on quantized emb): {np.mean(acc):.4f}, std: {np.std(acc):.4f}")
    
    if True:
        train_emb, val_emb, test_emb = embeddings[:num_train], embeddings[num_train:num_train+num_val], embeddings[num_train+num_val:]

        batch_size = 5120
        acc = []
        for i,_ in enumerate(seeds):
            print(f"####### Run seed {seeds[i]} for LinearProbing...")
            set_random_seed(seeds[i])
            print(f"training sample:{len(train_emb)}")
            test_acc = node_classification_linear_probing(
                (train_emb, val_emb, test_emb), 
                (train_lbls, val_lbls, test_lbls), 
                lr_f, weight_decay_f, max_epoch_f, device, batch_size=batch_size, shuffle=shuffle, mute=True)
            acc.append(test_acc)

        print(f"# final_acc (linear prob on encoded emb): {np.mean(acc):.4f}, std: {np.std(acc):.4f}")

    # return np.mean(acc)
   

class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits
        

def node_classification_linear_probing(embeddings, labels, lr, weight_decay, max_epoch, device, mute=False, batch_size=-1, shuffle=True):
    criterion = torch.nn.CrossEntropyLoss()

    train_emb, val_emb, test_emb = embeddings
    train_label, val_label, test_label = labels
    train_label = train_label.to(torch.long)
    val_label = val_label.to(torch.long)
    test_label = test_label.to(torch.long)
    
    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch), disbale=True)
    else:
        epoch_iter = range(max_epoch)

    encoder = LogisticRegression(train_emb.shape[1], int(train_label.max().item() + 1))
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    
    if batch_size > 0:
        train_loader = LinearProbingDataLoader(np.arange(len(train_emb)), train_emb, train_label, batch_size=batch_size, num_workers=4, persistent_workers=True, shuffle=shuffle)
        val_loader = LinearProbingDataLoader(np.arange(len(val_emb)), val_emb, val_label, batch_size=batch_size, num_workers=4, persistent_workers=True,shuffle=False)
        test_loader = LinearProbingDataLoader(np.arange(len(test_emb)), test_emb, test_label, batch_size=batch_size, num_workers=4, persistent_workers=True,shuffle=False)
    else:
        train_loader = [np.arange(len(train_emb))]
        val_loader = [np.arange(len(val_emb))]
        test_loader = [np.arange(len(test_emb))]

    def eval_forward(loader, _label):
        pred_all = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            pred = encoder(None, batch_x)
            pred_all.append(pred.cpu())
        pred = torch.cat(pred_all, dim=0)
        acc = eval_function(pred, _label)
        return acc

    for epoch in epoch_iter:
        encoder.train()

        for batch_x, batch_label in train_loader:
            batch_x = batch_x.to(device)
            batch_label = batch_label.to(device)
            pred = encoder(None, batch_x)
            loss = criterion(pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            encoder.eval()
            val_acc = eval_forward(val_loader, val_label)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(encoder)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc:.4f}")

    best_model.eval()
    encoder = best_model
    with torch.no_grad():
        test_acc = eval_forward(test_loader, test_label)
    # if mute:
    #     print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    # else:
    print(f"--- TestAcc: {test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    return test_acc


def finetune(
    model, 
    graph, 
    feats, 
    ego_graph_nodes, 
    labels, 
    split_idx, 
    lr_f, weight_decay_f, max_epoch_f, 
    use_scheduler, batch_size, 
    device, 
    logger=None, 
    full_graph_forward=False,
):
    logging.info("-- Finetuning in downstream tasks ---")
    train_egs, val_egs, test_egs = ego_graph_nodes
    print(f"num of egos:{len(train_egs)},{len(val_egs)},{len(test_egs)}")

    print(graph.num_nodes())

    train_nid = split_idx["train"].numpy()
    val_nid = split_idx["valid"].numpy()
    test_nid = split_idx["test"].numpy()

    train_lbls, val_lbls, test_lbls = [x.long() for x in labels]
    print(f"num of labels:{len(train_lbls)},{len(val_lbls)},{len(test_lbls)}")

    num_classes = max(max(train_lbls.max().item(), val_lbls.max().item()), test_lbls.max().item()) + 1
    
    model = model.get_encoder()
    model.reset_classifier(int(num_classes))
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = setup_finetune_dataloder("lc", graph, feats, train_egs, train_lbls, batch_size=batch_size, shuffle=True)
    val_loader = setup_finetune_dataloder("lc", graph, feats, val_egs, val_lbls, batch_size=batch_size, shuffle=False)
    test_loader = setup_finetune_dataloder("lc", graph, feats, test_egs, test_lbls, batch_size=batch_size, shuffle=False)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr_f, weight_decay=weight_decay_f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_f, weight_decay=weight_decay_f)

    if use_scheduler and max_epoch_f > 0:
        logging.info("Use schedular")
        warmup_epochs = int(max_epoch_f * 0.1)
        # scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch_f) ) * 0.5
        scheduler = lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else ( 1 + np.cos((epoch - warmup_epochs) * np.pi / (max_epoch_f - warmup_epochs))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    def eval_with_lc(model, loader):
        pred_counts = []
        model.eval()
        epoch_iter = tqdm(loader, disable=True)
        with torch.no_grad():
            for batch in epoch_iter:
                batch_g, targets, batch_lbls, node_idx = batch
                batch_g = batch_g.to(device)
                batch_lbls = batch_lbls.to(device)
                x = batch_g.ndata.pop("feat")

                prediction = model(batch_g, x)
                prediction = prediction[targets]
                pred_counts.append((prediction.argmax(1) == batch_lbls))
        pred_counts = torch.cat(pred_counts)
        acc = pred_counts.float().sum() / pred_counts.shape[0]
        return acc
    
    def eval_full_prop(model, g, nfeat, val_nid, test_nid, batch_size, device):
        model.eval()

        with torch.no_grad():
            pred = model.inference(g, nfeat, batch_size, device)
        model.train()

        return eval_function(pred[val_nid], val_lbls.cpu()), eval_function(pred[test_nid], test_lbls.cpu())

    best_val_acc = 0
    best_model = None
    best_epoch = 0
    test_acc = 0
    early_stop_cnt = 0

    for epoch in range(max_epoch_f):
        if epoch == 0:
            scheduler.step()
            continue
        if early_stop_cnt >= 10:
            break
        epoch_iter = tqdm(train_loader, disable=True)
        losses = []
        model.train()

        for batch_g, targets, batch_lbls, node_idx in epoch_iter:
            batch_g = batch_g.to(device)
            targets = targets.to(device)
            batch_lbls = batch_lbls.to(device)
            x = batch_g.ndata.pop("feat")

            prediction = model(batch_g, x)
            prediction = prediction[targets]
            loss = criterion(prediction, batch_lbls)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            metrics = {"finetune_loss": loss}
            wandb.log(metrics)

            if logger is not None:
                logger.log(metrics)

            epoch_iter.set_description(f"Finetuning | train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        if not full_graph_forward:
            if epoch > 0:
                val_acc = eval_with_lc(model, val_loader)
                _test_acc = 0
        else:
            if epoch > 0 and epoch % 1 == 0:
                val_acc, _test_acc = eval_full_prop(model, graph, feats, val_nid, test_nid, 10000, device)
                model = model.to(device)
        
        print('val Acc {:.4f}'.format(val_acc))
        if val_acc > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = val_acc
            test_acc = _test_acc
            best_epoch = epoch
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if not full_graph_forward:
            print("val Acc {:.4f}, Best Val Acc {:.4f}".format(val_acc, best_val_acc))
        else:
            print("Val Acc {:.4f}, Best Val Acc {:.4f} Test Acc {:.4f}".format(val_acc, best_val_acc, test_acc))

        metrics = {"epoch_val_acc": val_acc,
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "lr_f": get_current_lr(optimizer)}

        wandb.log(metrics)
        if logger is not None:
            logger.log(metrics)
        print(f"# Finetuning - Epoch {epoch} | train_loss: {np.mean(losses):.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}, Memory: {show_occupied_memory():.2f} MB")

    model = best_model
    if not full_graph_forward:
        test_acc = eval_with_lc(test_loader)

    print(f"Finetune | TestAcc: {test_acc:.4f} from Epoch {best_epoch}")
    return test_acc


def linear_probing_full_batch(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    with torch.no_grad():
        x, _, _ = model.embed(graph.to(device), x.to(device))
        in_feat = x.shape[1]
    encoder = LogisticRegression(in_feat, num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    encoder.to(device)
    optimizer_f = torch.optim.Adam(encoder.parameters(), lr=lr_f, weight_decay=weight_decay_f)
    final_acc, estp_acc = _linear_probing_full_batch(encoder, graph, x, optimizer_f, max_epoch_f, device, in_feat, num_classes, lr_f, weight_decay_f, mute)
    return final_acc, estp_acc


def _linear_probing_full_batch(model, graph, feat, optimizer, max_epoch, device, in_feat, num_classes, lr, weight_decay, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    if len(graph.ndata["train_mask"].shape) > 1:
        num_split = graph.ndata["train_mask"].shape[-1]
        total_result = torch.zeros(num_split, dtype=torch.float)
        for i in range(num_split):
            model = LogisticRegression(in_feat, num_classes)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            train_mask = graph.ndata["train_mask"][:,i]
            val_mask = graph.ndata["val_mask"][:,i]
            test_mask = graph.ndata["test_mask"][:,i]

            best_val_acc = 0
            best_val_epoch = 0
            best_model = None

            if not mute:
                epoch_iter = tqdm(range(max_epoch), disable=True)
            else:
                epoch_iter = range(max_epoch)

            for epoch in epoch_iter:
                model.train()
                out = model(graph, x)
                loss = criterion(out[train_mask], labels[train_mask])
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    pred = model(graph, x)
                    val_acc = eval_function(pred[val_mask], labels[val_mask])
                    val_loss = criterion(pred[val_mask], labels[val_mask])
                    test_acc = eval_function(pred[test_mask], labels[test_mask])
                    test_loss = criterion(pred[test_mask], labels[test_mask])

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    best_model = copy.deepcopy(model)

                if not mute:
                    epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

            best_model.eval()
            with torch.no_grad():
                pred = best_model(graph, x)
                estp_test_acc = eval_function(pred[test_mask], labels[test_mask])
            # if mute:
            #     print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
            # else:
            print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
            total_result[i] = estp_test_acc
        print(f"--- TestAcc over all splits: mean {torch.std_mean(total_result)[1]:.4f}, std {torch.std_mean(total_result)[0]:.4f} ---", )

    else:
        model = LogisticRegression(in_feat, num_classes)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val_acc = 0
        best_val_epoch = 0
        best_model = None

        if not mute:
            epoch_iter = tqdm(range(max_epoch), disable=True)
        else:
            epoch_iter = range(max_epoch)

        for epoch in epoch_iter:
            model.train()
            out = model(graph, x)
            loss = criterion(out[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pred = model(graph, x)
                val_acc = eval_function(pred[val_mask], labels[val_mask])
                val_loss = criterion(pred[val_mask], labels[val_mask])
                test_acc = eval_function(pred[test_mask], labels[test_mask])
                test_loss = criterion(pred[test_mask], labels[test_mask])

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                best_model = copy.deepcopy(model)

            if not mute:
                epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

        best_model.eval()
        with torch.no_grad():
            pred = best_model(graph, x)
            estp_test_acc = eval_function(pred[test_mask], labels[test_mask])
        # if mute:
        #     print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
        # else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    return test_acc, estp_test_acc
