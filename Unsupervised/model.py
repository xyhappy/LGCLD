import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool, GINConv, GCNConv, GATConv, SAGEConv
import numpy as np

class MLP(nn.Module):
    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        return self.net(x)
    

class Encoder(nn.Module):
    def __init__(self, num_features, hidden_dim, num_gc_layers, flag="GIN"):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.flag = flag
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if self.flag == "GIN":
                if i:
                    nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
                else:
                    nn = Sequential(Linear(num_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
                conv = GINConv(nn)
            elif self.flag == "GCN":
                if i:
                    conv = GCNConv(hidden_dim, hidden_dim)
                else:
                    conv = GCNConv(num_features, hidden_dim)
            elif self.flag == "GAT":
                if i:
                    conv = GATConv(hidden_dim, hidden_dim)
                else:
                    conv = GATConv(num_features, hidden_dim)
            elif self.flag == "SAGE":
                if i:
                    conv = SAGEConv(hidden_dim, hidden_dim)
                else:
                    conv = SAGEConv(num_features, hidden_dim)
            bn = torch.nn.BatchNorm1d(hidden_dim)
            self.convs.append(conv)
            self.bns.append(bn)
        self.projection = MLP(hidden_dim*num_gc_layers, hidden_dim*num_gc_layers, hidden_dim*num_gc_layers)

    def forward(self, x, edge_index, batch, device, Noise=False, mark=1):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        xs = []
        for i in range(self.num_gc_layers):
            if Noise == True:
                split_x = torch.split(x, torch.bincount(batch).tolist())
                stds = [chunk.std(dim=0).view(1,-1) for chunk in split_x]
                x = x + F.normalize(torch.normal(0, torch.ones_like(x) * torch.cat(stds,0)[batch]), dim=-1)
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, dim=1)
        if mark == 1:
            x = self.projection(x)
        return x
    
    def get_embeddings(self, loader, device):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x = self.forward(x, edge_index, batch, device, False, 0) 
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class EMA():
    def __init__(self, m):
        super().__init__()
        self.m = m

    def update_average(self, pretext, main):
        if pretext is None:
            return main
        return pretext * self.m + (1 - self.m) * main


def update_moving_average(ema_updater, pretext_model, main_model):
    for main_params, pretext_params in zip(main_model.parameters(), pretext_model.parameters()):
        pretext_weight, main_weight = pretext_params.data, main_params.data
        pretext_params.data = ema_updater.update_average(pretext_weight, main_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Net(nn.Module):
    def __init__(self, num_features, hidden_dim, num_gc_layers, alpha, flag, momentum):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.embedding_dim = hidden_dim * num_gc_layers
        self.flag = flag
        self.main_encoder = Encoder(num_features, hidden_dim, num_gc_layers)
        if self.flag == "GIN":
            self.pretext_encoder = copy.deepcopy(self.main_encoder)
            set_requires_grad(self.pretext_encoder, False)
            self.pretext_ema_updater = EMA(momentum)
        else:
            self.pretext_encoder = Encoder(num_features, hidden_dim, num_gc_layers, self.flag)

    def reset_moving_average(self):
        del self.pretext_encoder
        self.pretext_encoder = None

    def update_ma(self):
        assert self.pretext_encoder is not None, 'pretext encoder has not been created yet'
        update_moving_average(self.pretext_ema_updater, self.pretext_encoder, self.main_encoder)

    def loss_infomax(self, x, x_cl):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_cl_abs = x_cl.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_cl) / torch.einsum('i,j->ij', x_abs, x_cl_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_infomin(self, x, device):
        x = F.normalize(x, dim=0)
        in_matrix = torch.mm(x.T, x)
        iden_matrix = torch.eye(in_matrix.shape[0],in_matrix.shape[0]).to(device)
        loss = (iden_matrix - in_matrix).pow(2).mean()
        return loss
     
    def forward(self, x, edge_index, batch, device):
        m_xg = self.main_encoder(x, edge_index, batch, device)
        if self.flag == "GIN":
            with torch.no_grad():
                p_xg = self.pretext_encoder(x, edge_index, batch, device, True, 1)
        else:
            p_xg = self.pretext_encoder(x, edge_index, batch, device, True, 1)
        loss1 = self.loss_infomax(m_xg, p_xg)
        loss2 = self.loss_infomin(m_xg, device) + self.loss_infomin(p_xg, device)
        loss = loss1 + self.alpha * loss2          
        return loss