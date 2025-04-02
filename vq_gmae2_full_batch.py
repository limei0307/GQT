import logging
import numpy as np
from tqdm import tqdm
import torch
from args import parse_args

from gmae2_utils import create_optimizer, set_random_seed, TBLogger, get_current_lr
from gmae2_datasets.data_proc import load_small_dataset
from gmae2_models.finetune import linear_probing_full_batch, transformer_predictor_func
from gmae2_models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(config, data, train_idx, val_idx, test_idx, model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, eval_epoch, lr_f, weight_decay_f, max_epoch_f, linear_prob, transformer_predictor, logger=None):
    logging.info("start pre-training..")
    graph = graph.to(device)
    x = feat.to(device)

    target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    epoch_iter = tqdm(range(max_epoch), disable=True)

    for epoch in epoch_iter:
        model.train()

        loss = model(graph, x, targets=target_nodes)

        loss_dict = {"loss": loss.item()}
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if epoch == 0 or (epoch + 1) % eval_epoch == 0:
            if linear_prob:
                logging.info(f"linear probing at epoch {epoch}")
                linear_probing_full_batch(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=True)
            if transformer_predictor:
                logging.info(f"transformer probing at epoch {epoch}")
                transformer_predictor_func(config, data, train_idx, val_idx, test_idx, 
                                            model, graph, x, 
                                            None, None, None, 
                                            device, seeds=[0])

    return model


def main_vq_gmae2_full_batch(config, data, train_idx, val_idx, test_idx):
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [0]
    dataset_name = args.dataset
    max_epoch = args.epochs_tokenizer
    eval_epoch = int(args.epochs_tokenizer/10)
    max_epoch_f = args.epochs_transformer * 20
    num_hidden = args.vq_dim
    num_layers = args.gnn_num_layers
    encoder_type = args.gnn_type
    decoder_type = args.gnn_type
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr_tokenizer
    weight_decay = args.weight_decay_tokenizer
    lr_f = args.lr_transformer
    weight_decay_f = args.weight_decay_transformer
    linear_prob = args.linear_probe
    load_model = args.load_model
    logs = args.verbose
    use_scheduler = args.scheduler_tokenizer

    transformer_predictor = args.transformer_predictor

    graph, (num_features, num_classes) = load_small_dataset(dataset_name, split_seed=42, to_undirected=args.to_undirected)
    args.num_features = num_features

    if linear_prob:
        acc_list_linear_prob = []
        estp_acc_list_linear_prob = []
    if transformer_predictor:
        acc_list_transformer_predictor = []
        estp_acc_list_transformer_predictor = []

    # if len(graph.ndata["train_mask"].shape) > 1:
    #     seeds = [i for i in range(graph.ndata["train_mask"].shape[-1])]
    for i, seed in enumerate(seeds):
        # if len(graph.ndata["train_mask"].shape) > 1:
        #     print(f"####### Run split {i} with seed 0")
        #     graph.ndata["train_mask"] = graph.ndata["train_mask"][:,i]
        #     graph.ndata["val_mask"] = graph.ndata["val_mask"][:,i]
        #     graph.ndata["test_mask"] = graph.ndata["test_mask"][:,i]
        #     set_random_seed(0)
        # else:
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(config, data, train_idx, val_idx, test_idx, 
                             model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, eval_epoch, lr_f, weight_decay_f, max_epoch_f, linear_prob, transformer_predictor, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        
        # model = model.to(device)

        # if linear_prob:
        #     model.eval()
        #     final_acc, estp_acc = linear_probing_full_batch(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device)
        #     acc_list_linear_prob.append(final_acc)
        #     estp_acc_list_linear_prob.append(estp_acc)

        # if transformer_predictor:
        #     model.eval()
        #     final_acc, estp_acc = transformer_predictor_func(config, data, train_idx, val_idx, test_idx, 
        #                                     model, graph, x,
        #                                     None, None, None, 
        #                                     device, seeds=[0])
        #     acc_list_transformer_predictor.append(final_acc)
        #     estp_acc_list_transformer_predictor.append(estp_acc)

        if logger is not None:
            logger.finish()

    # if linear_prob:
    #     final_acc, final_acc_std = np.mean(acc_list_linear_prob), np.std(acc_list_linear_prob)
    #     estp_acc, estp_acc_std = np.mean(estp_acc_list_linear_prob), np.std(estp_acc_list_linear_prob)
    #     print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    #     print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")

    # if transformer_predictor:
    #     final_acc, final_acc_std = np.mean(acc_list_transformer_predictor), np.std(acc_list_transformer_predictor)
    #     estp_acc, estp_acc_std = np.mean(estp_acc_list_transformer_predictor), np.std(estp_acc_list_transformer_predictor)
    #     print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    #     print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
