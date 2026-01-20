import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .gcn import GCN  
from torch_geometric.utils import dropout_adj
from .meta_fairness import ParetoMetaAlignment, MetaSGD_Fairness, ParetoOptimizer



class ChannelMasker(nn.Module):
    def __init__(self, num_features): 
        super(ChannelMasker, self).__init__()
        initial_weights = torch.distributions.Uniform(0, 1).sample((num_features, 2))
        self.weights = nn.Parameter(initial_weights)
    def reset_parameters(self):
        if hasattr(self, 'weights'):
            nn.init.xavier_uniform_(self.weights.data)
    def forward(self):
        return self.weights




class MLP_Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim=1): 
        super(MLP_Discriminator, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim) 
    def reset_parameters(self):
        self.lin.reset_parameters()
    def forward(self, h):
        return torch.sigmoid(self.lin(h))
    
    
class CrossExpertContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, mode='cross'):
        super().__init__()
        self.temperature = temperature
        self.mode = mode 

    def forward(self, h1, h2):
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        sim_matrix = torch.matmul(h1, h2.T) / self.temperature
        labels = torch.arange(sim_matrix.size(0)).to(h1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        if self.mode == 'intra':
            sim_matrix_intra = torch.matmul(h1, h1.T) / self.temperature
            loss_intra = F.cross_entropy(sim_matrix_intra, labels)
            loss += loss_intra
        return loss



class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Projector, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.lin3 = nn.Linear(out_dim, out_dim)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h):
        y = self.lin1(h)
        y = self.lin2(y)
        y = self.lin3(y)
        return y


class SynTeacher(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=1, lambda_cl=0.1, temp=0.5,
                 lambda_adv=0.1, disc_hid_dim=None, disc_epochs=2, disc_lr=0.001, disc_wd=1e-5,
                 use_feature_masker=False, masker_lr=0.001, masker_wd=1e-5, masker_ratio=0.1,bce_projected_loss_weight=0.5,gnn2_noise_std=0.1,gnn2_edge_drop_rate=0.2,lambda_cov=1.0,use_meta_learning=True, meta_lr=0.01, num_meta_steps=5): 
        super(SynTeacher, self).__init__()
        self.expert_gnn1 = GCN(nfeat=in_dim, nhid=hid_dim, nclass=out_dim, dropout=0.5)
        self.expert_gnn2 = GCN(nfeat=in_dim, nhid=hid_dim, nclass=out_dim, dropout=0.5)
        self.projector_output_dim = hid_dim 
        self.projector = Projector(2*hid_dim, self.projector_output_dim)    
        self.gnn2_noise_std = gnn2_noise_std 
        self.gnn2_edge_drop_rate = gnn2_edge_drop_rate
        self.lambda_cov = lambda_cov
        self.final_classifier_on_projected = nn.Linear(self.projector_output_dim, out_dim)
        self.bce_projected_loss_weight = bce_projected_loss_weight
        
        self.lambda_cl = lambda_cl
        self.use_meta_learning = use_meta_learning
        self.meta_alignment = ParetoMetaAlignment(
            hidden_dim=hid_dim,
            num_objectives=2,
            meta_lr=meta_lr,
            num_meta_steps=num_meta_steps
        )
        self.pareto_optimizer = ParetoOptimizer(num_objectives=2)

        
        
        self.lambda_adv = lambda_adv  
        self.disc_epochs = disc_epochs
        if disc_hid_dim is None:
            disc_hid_dim = hid_dim 
        self.discriminator = MLP_Discriminator(disc_hid_dim, 1) 
        self.use_feature_masker = use_feature_masker
        self.sens_estimator = GCN(nfeat=in_dim, nhid=hid_dim, nclass=1, dropout=0.5)
        
        
        if self.use_feature_masker:
            self.feature_masker = ChannelMasker(in_dim)
            self.optimizer_masker = torch.optim.Adam(self.feature_masker.parameters(), lr=masker_lr, weight_decay=masker_wd)
            self.masker_ratio = masker_ratio 
        
            

        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            self.weights_init(m)
        self.para_joint = list(self.expert_gnn1.parameters()) + \
                          list(self.expert_gnn2.parameters()) + \
                          list(self.projector.parameters()) + \
                          list(self.final_classifier_on_projected.parameters()) 

        if self.use_feature_masker and hasattr(self, 'feature_masker'): 
            self.para_joint.extend(list(self.feature_masker.parameters()))
        self.para_gnn1 = list(self.expert_gnn1.parameters())
        self.para_gnn2 = list(self.expert_gnn2.parameters())
        self.para_proj = list(self.projector.parameters())
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=disc_lr, weight_decay=disc_wd)
    
    
    

    def train_fair_expert1(self, optimizer_gnn1_masker, optimizer_d, criterion_bce, data, epochs):
        idx_train = data.idx_train 
        for epoch in range(epochs):
            self.expert_gnn1.eval()
            self.feature_masker.eval() if self.use_feature_masker else None
            self.discriminator.train()
            for _ in range(self.disc_epochs):
                optimizer_d.zero_grad()
                with torch.no_grad():
                    h1_for_disc, _ = self.expert_gnn1(data.features, data.edge_index)
                sens_pred_d = self.discriminator(h1_for_disc[data.idx_train].detach())
                loss_d = criterion_bce(sens_pred_d, data.sens[data.idx_train].unsqueeze(1).float())
                loss_d.backward()
                optimizer_d.step()

            self.expert_gnn1.train()
            self.feature_masker.train() if self.use_feature_masker else None
            self.discriminator.eval()
            optimizer_gnn1_masker.zero_grad()

            current_features = data.features
            loss_mask_reg = torch.tensor(0.0).to(data.features.device)  
            if self.use_feature_masker:
                feature_mask_logits = self.feature_masker() 
                mask_one_hot = F.gumbel_softmax(feature_mask_logits, tau=1.0, hard=True, dim=-1) 
                feature_mask = mask_one_hot[:, 0]  
                current_features = data.features * feature_mask 
                loss_mask_reg = self.masker_ratio * F.mse_loss(feature_mask, torch.ones_like(feature_mask))
            h1, output1_task = self.expert_gnn1(current_features, data.edge_index)
            h1_train = h1[idx_train]
            output1_task_train = output1_task[idx_train]
            loss_bce1_task = criterion_bce(output1_task_train, data.labels[idx_train].unsqueeze(1).float())
            sens_pred_adv1 = self.discriminator(h1_train)  
            loss_adv1 = -F.binary_cross_entropy(sens_pred_adv1, data.sens[idx_train].unsqueeze(1).float())
            with torch.no_grad():
                s_est_output, _ = self.sens_estimator(data.features, data.edge_index)
                s_est_score = torch.sigmoid(s_est_output[idx_train])
            y1_score = torch.sigmoid(output1_task_train)
            y1_score_centered = y1_score - y1_score.mean()
            s_est_score_centered = s_est_score - s_est_score.mean()
            loss_cov = torch.abs(torch.mean(y1_score_centered * s_est_score_centered))
            total_loss_expert1 = loss_bce1_task + self.lambda_adv * loss_adv1 + self.lambda_cov * loss_cov + loss_mask_reg
            total_loss_expert1.backward()
            optimizer_gnn1_masker.step()

            if epoch % 50 == 0:
                 print(f"Epoch {epoch:03d}, Expert1 Loss: {total_loss_expert1.item():.4f}, BCE: {loss_bce1_task.item():.4f}, Adv: {loss_adv1.item():.4f}, Cov: {loss_cov.item():.4f}")

    def train_robust_expert2(self, optimizer_gnn2, criterion_bce, data, epochs):
        idx_train = data.idx_train 
        for epoch in range(epochs):
            self.train()
            optimizer_gnn2.zero_grad()
            noise = torch.randn_like(data.features) * self.gnn2_noise_std
            x_perturbed = data.features + noise 
            row, col, _ = data.edge_index.coo()
            edge_index_tensor = torch.stack([row, col], dim=0)
            perturbed_edge_index, _ = dropout_adj(edge_index_tensor, p=self.gnn2_edge_drop_rate,
                                                  force_undirected=True,  
                                                  training=self.training)
            h2, output2_task = self.expert_gnn2(x_perturbed, perturbed_edge_index)
            h2_train = h2[idx_train]
            output2_task_train = output2_task[idx_train] 
            loss_bce2_task = criterion_bce(output2_task_train, data.labels[idx_train].unsqueeze(1).float())
            loss_bce2_task.backward()
            optimizer_gnn2.step()
            
            if epoch % 50 == 0:
                 print(f"Epoch {epoch:03d}, Expert2 Loss: {loss_bce2_task.item():.4f}")
    
    
    
    def train_projector_module(self, optimizer_proj, criterion_bce, data, epochs):
        self.expert_gnn1.eval()
        self.expert_gnn2.eval()
        self.feature_masker.eval() if self.use_feature_masker else None
        self.projector.train()
        self.final_classifier_on_projected.train()

        for epoch in range(epochs):
            optimizer_proj.zero_grad()
            
            with torch.no_grad():
                current_features = data.features
                if self.use_feature_masker:
                    mask_one_hot = F.gumbel_softmax(self.feature_masker(), tau=1.0, hard=True, dim=-1)
                    feature_mask = mask_one_hot[:, 0]
                    current_features = data.features * feature_mask
                h1, _ = self.expert_gnn1(current_features, data.edge_index)

                noise = torch.randn_like(data.features) * self.gnn2_noise_std
                x_perturbed = data.features + noise
                
                
                row, col, _ = data.edge_index.coo()
                edge_index_tensor_for_h2 = torch.stack([row, col], dim=0)
                perturbed_edge_index, _ = dropout_adj(edge_index_tensor_for_h2, p=self.gnn2_edge_drop_rate, training=False) # training=False因为专家已冻结，但仍需扰动
                
                
                
                h2, _ = self.expert_gnn2(x_perturbed, perturbed_edge_index)

            h_combined = torch.cat((h1, h2), dim=1)
            h_projected = self.projector(h_combined)
            output_projected = self.final_classifier_on_projected(h_projected)

            loss = criterion_bce(output_projected[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
            loss.backward()
            optimizer_proj.step()

            if epoch % 50 == 0:
                print(f"Epoch {epoch:03d}, Projector Loss: {loss.item():.4f}")
                
                
                
    def train_meta_alignment(self, optimizer_meta, criterion_bce, data, epochs):
        
        idx_train = data.idx_train
        best_pareto_loss = float('inf')
        
        optimizer_meta_alignment = torch.optim.Adam(
            self.meta_alignment.parameters(), 
            lr=0.001, 
            weight_decay=1e-5
        )
        
        for epoch in range(epochs):
            self.meta_alignment.train()
            self.expert_gnn1.eval()  
            self.expert_gnn2.eval()
            
            optimizer_meta_alignment.zero_grad()

            with torch.no_grad():
                current_features = data.features
                if self.use_feature_masker:
                    mask_one_hot = F.gumbel_softmax(
                        self.feature_masker(), 
                        tau=1.0, 
                        hard=True, 
                        dim=-1
                    )
                    feature_mask = mask_one_hot[:, 0]
                    current_features = data.features * feature_mask
                h1, _ = self.expert_gnn1(current_features, data.edge_index)
                noise = torch.randn_like(data.features) * self.gnn2_noise_std
                x_perturbed = data.features + noise
                h2, _ = self.expert_gnn2(x_perturbed, data.edge_index)
            h1_train = h1[idx_train]
            h2_train = h2[idx_train]
            alignment_loss, pareto_weights, h1_aligned, h2_aligned = \
                self.meta_alignment(h1_train, h2_train, 
                                   sens=data.sens[idx_train], 
                                   labels=data.labels[idx_train])
            self.discriminator.eval()
            with torch.no_grad():
                sens_pred_h1 = self.discriminator(h1_aligned)
                fairness_objective = F.binary_cross_entropy(
                    sens_pred_h1, 
                    data.sens[idx_train].unsqueeze(1).float()
                )
            
            robustness_objective = F.mse_loss(h1_aligned, h2_aligned)
            objectives = torch.stack([fairness_objective, robustness_objective])
            pareto_loss = self.pareto_optimizer.compute_pareto_loss(objectives)
            total_loss = alignment_loss + pareto_loss
            total_loss.backward()
            optimizer_meta_alignment.step()
            with torch.no_grad():
                self.pareto_optimizer.update_preference(objectives)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: "
                      f"Alignment Loss: {alignment_loss.item():.4f}, "
                      f"Fairness Obj: {fairness_objective.item():.4f}, "
                      f"Robustness Obj: {robustness_objective.item():.4f}, "
                      f"Pareto Loss: {pareto_loss.item():.4f}, "
                      f"Preference: {self.pareto_optimizer.preference}")
            if pareto_loss.item() < best_pareto_loss:
                best_pareto_loss = pareto_loss.item()
        return self.pareto_optimizer.preference
    
    
    def finetune_contrastive(self, optimizer_experts, criterion_bce, data, epochs):
        self.expert_gnn1.train()
        self.expert_gnn2.train()
        self.discriminator.eval()
        self.sens_estimator.eval()
        self.projector.eval()
        self.final_classifier_on_projected.eval()


        for epoch in range(epochs):
            optimizer_experts.zero_grad()
            current_features = data.features
            if self.use_feature_masker:
                mask_one_hot = F.gumbel_softmax(self.feature_masker(), tau=1.0, hard=True, dim=-1)
                feature_mask = mask_one_hot[:, 0]
                current_features = data.features * feature_mask
            h1, output1 = self.expert_gnn1(current_features, data.edge_index)

            noise = torch.randn_like(data.features) * self.gnn2_noise_std
            x_perturbed = data.features + noise
            row, col, _ = data.edge_index.coo()
            edge_index_tensor_for_h2 = torch.stack([row, col], dim=0)
            perturbed_edge_index, _ = dropout_adj(edge_index_tensor_for_h2, p=self.gnn2_edge_drop_rate, training=True)
            
            
            h2, output2 = self.expert_gnn2(x_perturbed, perturbed_edge_index)

            idx_train = data.idx_train

            loss_cont = self.contrast_loss(h1[idx_train], h2[idx_train])

            loss_bce1 = criterion_bce(output1[idx_train], data.labels[idx_train].unsqueeze(1).float())
            loss_bce2 = criterion_bce(output2[idx_train], data.labels[idx_train].unsqueeze(1).float())
            aux_bce_loss = (loss_bce1 + loss_bce2) / 2
            total_loss = loss_cont + 0.1 * aux_bce_loss
            
            total_loss.backward()
            optimizer_experts.step()

            if epoch % 50 == 0:
                print(f"Epoch {epoch:03d}, Contrastive Finetune Loss: {total_loss.item():.4f}, Contrastive: {loss_cont.item():.4f}, Aux BCE: {aux_bce_loss.item():.4f}")


    def weights_init(self, m): 
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, GCNConv): 
            pass 

    def forward(self, x, edge_index):
        h1 = self.expert_gnn1(x,edge_index)
        x_perturbed = x + torch.randn_like(x) * 0.1
        h2 = self.expert_gnn2(x_perturbed, edge_index)
        h = self.projector(torch.cat((h1, h2), 1))
        y = self.classifier(h)
        return h, y

    def distill(self, x, edge_index, x_ones):
        h1 = self.forward_mlp(x)
        h2 = self.expert_gnn(x_ones, edge_index)
        h = self.projector(torch.cat((h1, h2), 1))
        return h

    def forward_mlp(self, x):
        h = x
        for l, layer in enumerate(self.expert_mlp):
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)
        y = self.c1(h)
        return h, y

    def forward_gnn(self, x_ones, edge_index):
        h = self.expert_gnn(x_ones, edge_index)
        y = self.c2(h)
        return h, y


    def train_expert_mlp(self, optimizer, criterion, epochs, data):
        total_loss = 0.0
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            h_mlp, output_mlp = self.forward_mlp(data.features)
            with torch.no_grad():
                features_one = torch.ones_like(data.features).to(data.features.device)
                h_gnn,_=self.forward_gnn(features_one,data.edge_index)
            
            cl_loss = self.contrast_loss(h_mlp[data.idx_train],h_gnn[data.idx_train])
            
            loss = criterion(output_mlp[data.idx_train],data.labels[data.idx_train].unsqueeze(1).float())
            total_loss = loss + self.lambda_cl * cl_loss
            
            total_loss.backward()
            optimizer.step()
        return h_mlp.detach()

    def train_expert_gnn(self, optimizer, criterion, epochs, data):
        features_one = torch.ones_like(data.features).to(data.features.device)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            h_gnn, output_gnn = self.forward_gnn(features_one, data.edge_index)
            
            with torch.no_grad():
                h_mlp, _ = self.forward_mlp(data.features)
            
            cl_loss = self.contrast_loss(h_gnn[data.idx_train],h_mlp[data.idx_train])
            
            
            loss = criterion(output_gnn[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
            total_loss = loss + self.lambda_cl * cl_loss
            
            total_loss.backward()
            optimizer.step()

        return h_gnn.detach()

    def train_projector(self, optimizer, criterion, epochs, data, input, label):
        for epoch in range(epochs):
            # train projector
            self.train()
            optimizer.zero_grad()

            output_proj = self.projector(input)
            loss_train = criterion(output_proj[data.idx_train], label[data.idx_train])
            loss_train.backward()
            optimizer.step()

        self.eval()
        with torch.no_grad():
            h_proj = self.projector(input)
        return h_proj