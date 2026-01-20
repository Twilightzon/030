import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *
from models.meta_fairness import ParetoMetaAlignment, MetaSGD_Fairness, ParetoOptimizer
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.utils import dropout_adj, convert


class Data:
    def __init__(self, edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx):
        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx

        
        
def train_sens_estimator(model, optimizer, criterion, epochs, data, save_name):
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _,s_output= model(data.features, data.edge_index)
        loss_train = criterion(s_output[data.idx_train], data.sens[data.idx_train].unsqueeze(1).float())
        loss_train.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _,s_output_val= model(data.features, data.edge_index)
        try:
            auc_val = roc_auc_score(data.sens[data.idx_val].cpu().numpy(), 
                                    torch.sigmoid(s_output_val[data.idx_val]).cpu().numpy())
        except ValueError:
            auc_val = -1
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}, Estimator Train Loss: {loss_train.item():.4f}, Estimator Val AUC: {auc_val:.4f}")
        if auc_val > best_auc:
            best_auc = auc_val
            torch.save(model.state_dict(), save_name)

def run(args):
    if args.dataset == 'bail':
        sens_attr = "WHITE"
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = "./datasets/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(args.dataset, sens_attr,
                                                                              predict_attr, path=path_bail,
                                                                              label_number=label_number,
                                                                              )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    elif args.dataset == 'pokec_z':
        dataset = 'region_job'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 4000
        sens_number = 200
        sens_idx = 3
        seed = 20
        path = "./datasets/pokec/"
        test_idx = False
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                               sens_attr,
                                                                                               predict_attr,
                                                                                               path=path,
                                                                                               label_number=label_number,
                                                                                               sens_number=sens_number,
                                                                                               seed=seed,
                                                                                               test_idx=test_idx)
        labels[labels > 1] = 1
    elif args.dataset == 'pokec_n':
        dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 3500
        sens_number = 200
        sens_idx = 3
        seed = 20
        path = "./datasets/pokec/"
        test_idx = False
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                               sens_attr,
                                                                                               predict_attr,
                                                                                               path=path,
                                                                                               label_number=label_number,
                                                                                               sens_number=sens_number,
                                                                                               seed=seed,
                                                                                               test_idx=test_idx)
        labels[labels > 1] = 1
    else:
        print('Invalid dataset name!!')
        exit(0)

        
        
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sens = sens.to(device)
    
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    features = features.to(args.device)
    num_nodes = features.shape[0]
    edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), ).to(args.device)
    labels = labels.to(args.device)
    data = Data(edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)

    num_class = 1
    
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_cont = ContLoss(tem=args.tem)

    estimator_save_name = f'estimator_{args.dataset}.pt'
    sens_estimator = GCN(nfeat=features.shape[1],
                         nhid=args.hidden,
                         nclass=1,
                         dropout=args.dropout).to(args.device)
    optimizer_estimator = optim.Adam(sens_estimator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_sens_estimator(sens_estimator, optimizer_estimator, criterion_bce, 500, data, estimator_save_name) # 训练200轮作为示例
    if args.model == 'gcn':
        stu_model = S_GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=num_class,
                    dropout=args.dropout,
                    beta_ib=args.beta_ib)
        optimizer = optim.Adam(stu_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        stu_model = stu_model.to(args.device)
        trained_gnn = GCN(nfeat=features.shape[1],
                          nhid=args.hidden,
                          nclass=num_class,
                          dropout=args.dropout)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trained_gnn = trained_gnn.to(args.device)

    elif args.model == 'gin':
        stu_model = GIN(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=num_class,
                        dropout=args.dropout)
        optimizer = optim.Adam(stu_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        stu_model = stu_model.to(args.device)
        trained_gnn = GIN(nfeat=features.shape[1],
                          nhid=args.hidden,
                          nclass=num_class,
                          dropout=args.dropout)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trained_gnn = trained_gnn.to(args.device)
    train_vanilla(trained_gnn, optimizer_van, criterion_bce, args.epochs, data, save_name=f'{args.model}_vanilla.pt')
    trained_gnn.load_state_dict(torch.load(f'{args.model}_vanilla.pt'))
    trained_gnn.eval()
    with torch.no_grad():
        h_van, output_van = trained_gnn(features, edge_index.to(args.device))
    syn_t = SynTeacher(in_dim=features.shape[1],
                   hid_dim=args.hidden,
                   out_dim=num_class,
                   lambda_cl=args.lambda_cl,
                   temp=args.temp,
                   lambda_adv=args.lambda_adv,
                   disc_epochs=args.disc_epochs,
                   disc_lr=args.disc_lr,
                   disc_wd=args.disc_wd,
                   use_feature_masker=args.use_feature_masker, 
                   masker_ratio=args.masker_ratio, 
                   bce_projected_loss_weight=args.bce_projected_loss_weight,
                   gnn2_noise_std=args.gnn2_noise_std,
                   gnn2_edge_drop_rate=args.gnn2_edge_drop_rate,
                   lambda_cov=args.lambda_cov,use_meta_learning=True,
    meta_lr=args.meta_lr,   
    num_meta_steps=args.num_meta_steps
                      )
    syn_t.sens_estimator.load_state_dict(torch.load(estimator_save_name))
    syn_t = syn_t.to(args.device)

    params_gnn1_masker = list(syn_t.expert_gnn1.parameters())
    if args.use_feature_masker:
        params_gnn1_masker.extend(list(syn_t.feature_masker.parameters()))
    optimizer_gnn1_masker = optim.Adam(params_gnn1_masker, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_d = syn_t.optimizer_d
    optimizer_gnn2 = optim.Adam(syn_t.expert_gnn2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params_experts = list(syn_t.expert_gnn1.parameters()) + list(syn_t.expert_gnn2.parameters())
    if args.use_feature_masker:
        params_experts.extend(list(syn_t.feature_masker.parameters()))
    optimizer_experts_contrastive = optim.Adam(params_experts, lr=args.lr * 0.1, weight_decay=args.weight_decay) # 使用较小的学习率进行微调

    params_proj = list(syn_t.projector.parameters()) + list(syn_t.final_classifier_on_projected.parameters())
    optimizer_proj = optim.Adam(params_proj, lr=args.lr, weight_decay=args.weight_decay)
    epochs_stage1 = args.epochs // 2 
    epochs_stage2 = args.epochs // 4 
    epochs_stage3 = args.epochs // 4 
    syn_t.train_fair_expert1(optimizer_gnn1_masker, optimizer_d, criterion_bce, data, epochs_stage1)
    syn_t.train_robust_expert2(optimizer_gnn2, criterion_bce, data, epochs_stage1)

    optimizer_meta = torch.optim.Adam(
        syn_t.meta_alignment.parameters(),
        lr=args.meta_lr,
        weight_decay=1e-5
    )
    pareto_preference = syn_t.train_meta_alignment(
        optimizer_meta, criterion_bce, data, epochs_stage2
    )
    syn_t.train_projector_module(optimizer_proj, criterion_bce, data, epochs_stage3)
    syn_t.eval()
    with torch.no_grad():
        current_features = data.features
        if args.use_feature_masker:
            mask_one_hot = F.gumbel_softmax(syn_t.feature_masker(), tau=1.0, hard=True, dim=-1)
            feature_mask = mask_one_hot[:, 0]
            current_features = data.features * feature_mask
        h1, _ = syn_t.expert_gnn1(current_features, data.edge_index)
        noise = torch.randn_like(data.features) * args.gnn2_noise_std
        x_perturbed = data.features + noise
        row, col, _ = data.edge_index.coo()
        edge_index_tensor = torch.stack([row, col], dim=0)
        perturbed_edge_index, _ = dropout_adj(edge_index_tensor, p=args.gnn2_edge_drop_rate, training=False)
        h2, _ = syn_t.expert_gnn2(x_perturbed, perturbed_edge_index)
        h_fair = syn_t.projector(torch.cat((h1, h2), dim=1)).detach()
    train_student(stu_model, optimizer, criterion_bce, criterion_cont, args, data, save_name=f'{args.model}_student.pt',
                  soft_target=h_fair)

    auc, f1, acc, dp, eo = evaluation(stu_model, f'{args.model}_student.pt', data)

    return auc, f1, acc, dp, eo


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--proj_hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_out_heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--tem', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--lr_w', type=float, default=1)
    parser.add_argument('--beta_ib', type=float, default=1)
    parser.add_argument('--lambda_cl', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--disc_epochs', type=int, default=50)
    parser.add_argument('--disc_lr', type=float, default=0.001)
    parser.add_argument('--disc_wd', type=float, default=1e-5)
    parser.add_argument('--use_feature_masker', action='store_true', default=True)
    parser.add_argument('--masker_lr', type=float, default=0.001)
    parser.add_argument('--masker_wd', type=float, default=1e-5)
    parser.add_argument('--masker_ratio', type=float, default=0.1)
    parser.add_argument('--bce_projected_loss_weight', type=float, default=1.0)
    parser.add_argument('--gnn2_noise_std', type=float, default=0.2)
    parser.add_argument('--gnn2_edge_drop_rate', type=float, default=0.7)
    parser.add_argument('--lambda_cov', type=float, default=1.0)
    parser.add_argument('--use_meta_learning', action='store_true', default=True)
    parser.add_argument('--meta_lr', type=float, default=0.001)
    parser.add_argument('--num_meta_steps', type=int, default=5)
    parser.add_argument('--pareto_preference_lr', type=float, default=0.1)
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    auc, f1, acc, dp, eo = np.zeros(shape=(args.seed_num, 2)), np.zeros(shape=(args.seed_num, 2)), \
                                     np.zeros(shape=(args.seed_num, 2)), np.zeros(shape=(args.seed_num, 2)), \
                                     np.zeros(shape=(args.seed_num, 2))

    for seed in range(args.seed_num):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.allow_tf32 = False
        auc[seed, :], f1[seed, :], acc[seed, :], dp[seed, :], eo[seed, :] = run(args)

        print(f"========seed {seed}========")
        print("=================START=================")
        print(f"Parameter:τ={args.tem}, γ={args.gamma}, lr_w={args.lr_w}")
        print(f"AUCROC: {np.around(np.mean(auc[:, 0]) * 100, 2)} ± {np.around(np.std(auc[:, 0]) * 100, 2)}")
        print(f'F1-score: {np.around(np.mean(f1[:, 0]) * 100, 2)} ± {np.around(np.std(f1[:, 0]) * 100, 2)}')
        print(f'ACC: {np.around(np.mean(acc[:, 0]) * 100, 2)} ± {np.around(np.std(acc[:, 0]) * 100, 2)}')
        print(f'parity: {np.around(np.mean(dp[:, 0]) * 100, 2)} ± {np.around(np.std(dp[:, 0]) * 100, 2)}')
        print(f'Equality: {np.around(np.mean(eo[:, 0]) * 100, 2)} ± {np.around(np.std(eo[:, 0]) * 100, 2)}')
        print("=================END=================")

