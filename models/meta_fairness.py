import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable


class ParetoMetaAlignment(nn.Module):
    def __init__(self, hidden_dim, num_objectives=2, meta_lr=0.01, num_meta_steps=5):
        super(ParetoMetaAlignment, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_objectives = num_objectives 
        self.meta_lr = meta_lr
        self.num_meta_steps = num_meta_steps
        self.pareto_weight_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objectives),
            nn.Softmax(dim=-1) 
        )
        
        self.align_net_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.align_net_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def compute_fairness_metric(self, h, sens, labels):
        sens_pred = torch.sigmoid(h.mean(dim=1)) 
        fairness_loss = F.binary_cross_entropy(sens_pred, sens)
        return fairness_loss
    
    def compute_robustness_metric(self, h1, h2):
        robustness_loss = F.mse_loss(h1, h2)
        return robustness_loss
    
    def pareto_forward(self, h1, h2):
        h_concat = torch.cat([h1, h2], dim=1) 
        weights = self.pareto_weight_net(h_concat)
        return weights
    
    def align_representations(self, h1, h2):
        h1_aligned = self.align_net_1(h1)
        h2_aligned = self.align_net_2(h2)
        return h1_aligned, h2_aligned
    
    def compute_alignment_loss(self, h1_aligned, h2_aligned, weights):
        h1_norm = F.normalize(h1_aligned, dim=1)
        h2_norm = F.normalize(h2_aligned, dim=1)
        similarity = (h1_norm * h2_norm).sum(dim=1) 
        weighted_similarity = similarity * (weights[:, 0] * weights[:, 1]) 
        alignment_loss = -weighted_similarity.mean()
        return alignment_loss
    
    def forward(self, h1, h2, sens=None, labels=None):
        pareto_weights = self.pareto_forward(h1, h2)
        h1_aligned, h2_aligned = self.align_representations(h1, h2)
        alignment_loss = self.compute_alignment_loss(h1_aligned, h2_aligned, pareto_weights)
        return alignment_loss, pareto_weights, h1_aligned, h2_aligned


class MetaSGD_Fairness(torch.optim.SGD):
    def __init__(self, params, modules, lr=0.01, momentum=0, weight_decay=0, 
                 nesterov=False, rollback=False):
        super(MetaSGD_Fairness, self).__init__(
            params, lr, momentum=momentum, 
            weight_decay=weight_decay, nesterov=nesterov
        )
        self.prev_states = []
        self.modules = modules + [self]
        self.rollback = rollback
    
    def parameters(self):
        for pg in self.param_groups:
            for p in pg['params']:
                yield p
    
    def get_state(self):
        return copy.deepcopy([m.state_dict() for m in self.modules])
    
    def set_state(self, state):
        for m, s in zip(self.modules, state):
            m.load_state_dict(s)
    
    def step(self, objective=None, *args, **kwargs):
        if objective is not None:
            self.prev_states.append((self.get_state(), objective, args, kwargs))
            loss = objective(*args, **kwargs)
            loss.backward()
        super(MetaSGD_Fairness, self).step()
    
    def meta_backward(self):
        alpha_groups = []
        for pg in self.param_groups:
            alpha_groups.append([])
            for p in pg['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                grad = p.grad.data.clone()
                alpha_groups[-1].append((grad, torch.zeros_like(grad)))
        curr_state = self.get_state()
        for prev_state in reversed(self.prev_states):
            state, objective, args, kwargs = prev_state
            self.set_state(state)
            loss = objective(*args, **kwargs)
            grad = torch.autograd.grad(
                loss, list(self.parameters()),
                create_graph=True, allow_unused=True
            )
            grad = {p: g for p, g in zip(self.parameters(), grad)}
            X = 0.0
            for pg, ag in zip(self.param_groups, alpha_groups):
                lr = pg['lr']
                wd = pg['weight_decay']
                momentum = pg['momentum']
                for p, a in zip(pg['params'], ag):
                    g = grad[p]
                    if g is not None:
                        X = X + g.mul(a[0].mul(-lr) + a[1]).sum()
            self.zero_grad()
            X.backward()
            for pg, ag in zip(self.param_groups, alpha_groups):
                lr = pg['lr']
                wd = pg['weight_decay']
                momentum = pg['momentum']
                for p, a in zip(pg['params'], ag):
                    a_new = (
                        a[0].mul(1 - lr * wd).add_(wd, a[1]).add_(p.grad.data),
                        a[1].mul(momentum).add_(-lr * momentum, a[0])
                    )
                    a[0].copy_(a_new[0])
                    a[1].copy_(a_new[1])
        self.prev_states = []
        if not self.rollback:
            self.set_state(curr_state)


class ParetoOptimizer:
    def __init__(self, num_objectives=2, preference_update_rate=0.1):
        self.num_objectives = num_objectives
        self.preference_update_rate = preference_update_rate
        self.preference = torch.ones(num_objectives) / num_objectives
        
    def update_preference(self, objectives_values):
        self.preference = self.preference.to(objectives_values.device)
        normalized_objectives = objectives_values / (objectives_values.sum() + 1e-8)
        gradient = normalized_objectives
        self.preference = self.preference + self.preference_update_rate * gradient
        self.preference = self.preference / self.preference.sum()
        return self.preference
    
    def compute_pareto_loss(self, objectives_values, weights=None):
        if weights is None:
            weights = self.preference
        weights = weights.to(objectives_values.device)
        pareto_loss = (weights * objectives_values).sum()
        return pareto_loss
    
    def is_pareto_optimal(self, current_objectives, reference_objectives):
        not_worse = (current_objectives <= reference_objectives).all()
        better = (current_objectives < reference_objectives).any()
        
        return not_worse and better