import argparse

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--first_node_only', default=False, action='store_true')
    
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--root', type=str, default='dataset/', help='')
    parser.add_argument('--tokenizer_input', type=str, default='feature', choices=['feature', 'llm', 'sign'])
    parser.add_argument('--transformer_input', type=str, default='token_id_quantized_emb', help='[token_id / ] + [input_emb / sign_emb / quantized_emb / encoder_emb / llm_emb]')
    parser.add_argument('--sequence_type', type=str, default='ppr', choices=['ppr', 'bfs', 'dfs', 'hybrid'])
    parser.add_argument('--max_sequence_length', type=int, default=15)
    parser.add_argument('--max_degree', type=int, default=512)
    parser.add_argument('--max_distance', type=int, default=256)
    parser.add_argument('--ppr_num_neighbors', type=int, default=30, help='Number of neighbors for PPR')
    parser.add_argument('--bfs_num_hops', type=int, default=4, help='Hop of neighbors for BFS')
    parser.add_argument('--dfs_walk_length', type=int, default=50, help='Walk length for DFS')
    parser.add_argument('--dfs_num_walks', type=int, default=50, help='Number of RW repeats for DFS')
    parser.add_argument('--distance_encoding', default=False, action='store_true')
    parser.add_argument('--degree_encoding', default=False, action='store_true')
    parser.add_argument('--position_encoding', default=False, action='store_true')
    parser.add_argument('--hierarchy_encoding', default=False, action='store_true')
    parser.add_argument('--use_gating', default=False, action='store_true')
    parser.add_argument('--num_sign_hops', type=int, default=-1, help='Number of hops for SIGN embedding')
    parser.add_argument('--normalize_feature', default=False, action='store_true')
    parser.add_argument('--scale_feature', default=False, action='store_true')
    parser.add_argument('--to_undirected', default=False, action='store_true')
    parser.add_argument('--llm_embedding', default=False, action='store_true')
    parser.add_argument('--mask_probability', type=float, default=0.15)

    # Transformer config
    parser.add_argument('--transformer_num_layers', type=int, default=3, help='')
    parser.add_argument('--transformer_hidden_dim', type=int, default=128, help='')
    parser.add_argument('--transformer_num_heads', type=int, default=8, help='')
    parser.add_argument('--transformer_dropout', type=float, default=0.5, help='')
    parser.add_argument('--transformer_activation', type=str, default='gelu')
    parser.add_argument('--transformer_ffn', type=int, default=256, help='')
    parser.add_argument('--transformer_aggr_tokens', type=str, default='sum')
    parser.add_argument('--transformer_aggr_neighbors', type=str, default='attn')
    parser.add_argument('--use_codebook', default=False, action='store_true')

    # GNN config
    parser.add_argument('--gnn_type', type=str, default='gat') # GraphMAE2 only support gat
    parser.add_argument('--gnn_activation', type=str, default='prelu')
    parser.add_argument('--gnn_norm', type=str, default='layernorm')
    parser.add_argument('--gnn_dropout', type=float, default=0.2)
    parser.add_argument('--gnn_num_layers', type=int, default=4)
    parser.add_argument('--gnn_hidden_dim', type=int, default=1024)
    parser.add_argument('--gnn_num_heads', type=int, default=8)
    parser.add_argument('--gnn_skip', default=True)

    # MLP config
    parser.add_argument('--mlp_layers', type=int, default=[512, 256], nargs='+')
    parser.add_argument('--mlp_activation', type=str, default='gelu')
    parser.add_argument('--mlp_norm', type=str, default='layernorm')
    parser.add_argument('--mlp_dropout', type=float, default=0.5)
    parser.add_argument('--mlp_skip', default=False, action='store_true')
    
    # vq parameters
    # https://github.com/lucidrains/vector-quantize-pytorch
    parser.add_argument('--ssl_method', type=str, default='gmae2', help='ae, dgi, bgrl, gmae2')
    parser.add_argument("--use_vq", default=False, action='store_true')
    parser.add_argument("--kmeans_init", action="store_true")
    parser.add_argument("--cosine_sim", action="store_true")
    parser.add_argument("--implicit_neural_codebook", action="store_true")
    parser.add_argument('--vq_method', type=str, default='rq', help='vq, rq')
    parser.add_argument('--vq_dim', type=int, default=1024, help='')
    parser.add_argument('--vq_codebook_size', type=int, default=4096, help='')
    parser.add_argument('--vq_num_codebook', type=int, default=3, help='')
    parser.add_argument('--vq_decay', type=float, default=0.8, help='')
    parser.add_argument('--vq_weight', type=float, default=0.25, help='')
    parser.add_argument('--vq_encoder', type=str, default='gnn')
    parser.add_argument('--vq_decoder', type=str, default='gnn')
    parser.add_argument('--rec_feature', default=True, action='store_true')
    parser.add_argument('--rec_edge', default=False, action='store_true')
    parser.add_argument('--cluster_loss', default=False, action='store_true')
    parser.add_argument('--num_cluster', type=int, default=1024, help='')
    parser.add_argument('--label_loss', default=False, action='store_true')
    parser.add_argument('--feature_loss_type', type=str, default='cosine', choices=['l1', 'l2', 'cosine'])
    parser.add_argument('--edge_loss_type', type=str, default='ce', choices=['ce'])
    parser.add_argument('--feature_rec_loss_weight', type=float, default=10, help='')
    parser.add_argument('--edge_rec_loss_weight', type=float, default=1, help='')
    parser.add_argument('--commit_loss_weight', type=float, default=1, help='')
    parser.add_argument('--linear_probe', default=False, action='store_true')
    parser.add_argument('--transformer_predictor', default=False, action='store_true')
    
    # training parameters
    parser.add_argument('--runs', type=int, default=1, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--n_parts', type=int, default=1, help='')

    parser.add_argument('--epochs_tokenizer', type=int, default=200, help='')
    parser.add_argument('--lr_tokenizer', type=float, default=0.001, help='')
    parser.add_argument('--scheduler_tokenizer', default=False, action='store_true')
    parser.add_argument('--weight_decay_tokenizer', type=float, default=1e-6, help='')
    parser.add_argument('--warmup_tokenizer', type=int, default=100, help='')

    parser.add_argument('--epochs_transformer', type=int, default=100, help='')
    parser.add_argument('--lr_transformer', type=float, default=0.0001, help='')
    parser.add_argument('--scheduler_transformer', default=False, action='store_true')
    parser.add_argument('--weight_decay_transformer', type=float, default=1e-6, help='')
    parser.add_argument('--warmup_transformer', type=int, default=100, help='')

    parser.add_argument('--epochs_masked_transformer', type=int, default=0, help='')
    parser.add_argument('--lr_masked_transformer', type=float, default=0.001, help='')
    parser.add_argument('--scheduler_masked_transformer', default=False, action='store_true')
    parser.add_argument('--weight_decay_masked_transformer', type=float, default=1e-6, help='')
    parser.add_argument('--warmup_masked_transformer', type=int, default=100, help='')
    
    parser.add_argument('--early_stop_epochs', type=int, default=20, help='')
    parser.add_argument("--post_norm", default=False, action='store_true')
    parser.add_argument("--pre_norm", default=False, action='store_true')

    parser.add_argument('--save_path', type=str, default='none', help='if given, use the saved tokens; otherwise, train the tokenizer')
    parser.add_argument('--save_epochs', type=int, default=200, help='')
    parser.add_argument("--verbose", default=False, action='store_true')
    parser.add_argument("--compile", default=False, action='store_true')

    parser.add_argument("--search", default=False, action='store_true')
    parser.add_argument('--gpu', type=float, default=0.2)
    parser.add_argument('--cpu', type=int, default=10)

    # additional configs for GraphMAE2
    parser.add_argument('--drop_edge_rate', type=float, default=0.5)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--ego_graph_file_path", type=str, default='GraphMAE2/lc_ego_graphs/ogbn-arxiv-lc-ego-graphs-256.pt')
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default='GraphMAE2/checkpoints/gat_gat_1024_4_ogbn-arxiv_vq.pt')
    # parser.add_argument("--load_emb", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--mask_method", type=str, default="random")
    parser.add_argument("--remask_rate", type=float, default=0.5)
    parser.add_argument("--remask_method", type=str, default="random")
    parser.add_argument("--num_out_heads", type=int, default=1)
    parser.add_argument("--in_drop", type=float, default=.2, help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1, help="attention dropout")
    parser.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu")
    parser.add_argument("--alpha_l", type=float, default=6)
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--num_remasking", type=int, default=3) 
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--no_pretrain", action="store_true")

    return parser.parse_args()