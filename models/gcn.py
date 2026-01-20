import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class InformationBottleneck(nn.Module):
    def __init__(self, input_dim, output_dim, beta=0.1):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Linear(input_dim, output_dim)
        self.mu = nn.Linear(output_dim, output_dim)
        self.logvar = nn.Linear(output_dim, output_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, self.beta * kl_div



class S_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,beta_ib=0.1):
        super(S_GCN, self).__init__()
        self.body = S_GCN_Body(nfeat, nhid, dropout,beta_ib=beta_ib)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h,ib_loss = self.body(x, edge_index)
        x = self.fc(h)
        return h, x,ib_loss


class S_GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout,beta_ib=0.1):
        super(S_GCN_Body, self).__init__()
        
        self.ib_layer = InformationBottleneck(nfeat,nfeat,beta=beta_ib)
        
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        
        x_ib, ib_loss = self.ib_layer(x)
        
        x = self.gc1(x_ib, edge_index)
        return x,ib_loss
    
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.body(x, edge_index)
        x = self.fc(h)
        return h, x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return x