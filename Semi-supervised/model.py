import copy
import torch
import torch.nn.functional as F
from functools import partial
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv, GINConv, GATConv, SAGEConv


class Encoder(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden, num_gc_layers, residual, num_fc_layers, global_pool, dropout, flag="GCN"):
        super(Encoder, self).__init__()
        self.conv_residual = residual
        self.global_pool = global_pool
        self.dropout = dropout
        self.flag = flag

        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        # BN
        for i in range(num_gc_layers):
            bn = torch.nn.BatchNorm1d(hidden) if i else torch.nn.BatchNorm1d(num_features)
            self.bns_conv.append(bn)

        # GNN
        for i in range(num_gc_layers):
            if self.flag == "GCN":
                conv = GCNConv(hidden, hidden) if i else GCNConv(num_features, hidden)
            
            elif self.flag == "GIN":
                if i:
                    nn = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden))
                else:
                    nn = Sequential(Linear(num_features, hidden), ReLU(), Linear(hidden, hidden))
                conv = GINConv(nn)

            elif self.flag == "GAT":
                conv = GATConv(hidden, hidden) if i else GCNConv(num_features, hidden)
            
            elif self.flag == "SAGE":
                conv = SAGEConv(hidden, hidden) if i else SAGEConv(num_features, hidden)

            self.convs.append(conv)

        # NN
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.lin_class = Linear(hidden, num_classes)

        # projection head
        self.proj_head = Sequential(Linear(hidden, hidden), ReLU(inplace=True), Linear(hidden, hidden))
            
    
    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)
    
    
    def forward_cl(self, x, edge_index, batch, noise=False):
        for i, conv in enumerate(self.convs):
            if noise == True:
                split_x = torch.split(x, torch.bincount(batch).tolist())
                stds = [torch.std(chunk, dim=0).unsqueeze(0) for chunk in split_x]
                x = x + F.normalize(torch.normal(0, torch.ones_like(x) * torch.cat(stds,0)[batch]), dim=-1)

            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x


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


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden, num_gc_layers,
                 num_fc_layers, residual, global_pool, dropout, alpha, flag, momentum):
        super(Net, self).__init__()
        self.conv_residual = residual 
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        self.alpha = alpha
        self.main_encoder = Encoder(num_features, num_classes, hidden, num_gc_layers, residual, num_fc_layers, self.global_pool, dropout)
        if flag == "GCN":
            self.pretext_encoder = copy.deepcopy(self.main_encoder)
            set_requires_grad(self.pretext_encoder, False)
            self.pretext_ema_updater = EMA(momentum)
        else:
            self.pretext_encoder = Encoder(num_features, num_classes, hidden, num_gc_layers, residual, num_gc_layers, self.global_pool, dropout, flag)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def reset_moving_average(self):
        del self.pretext_encoder
        self.pretext_encoder = None

    def update_ma(self):
        assert self.pretext_encoder is not None, 'pretext encoder has not been created yet'
        update_moving_average(self.pretext_ema_updater, self.pretext_encoder, self.main_encoder)

    def loss_infomax(self, x, x_cl):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)  #|x|
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
    
    def forward_cl(self, x, edge_index, batch, device):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        m_xg = self.main_encoder.forward_cl(x, edge_index, batch)
        p_xg = self.pretext_encoder.forward_cl(x, edge_index, batch, True)
        loss1 = self.loss_infomax(m_xg, p_xg)
        loss2 = self.loss_infomin(m_xg, device) + self.loss_infomin(p_xg, device)
        loss = loss1 + self.alpha * loss2
        return loss
    
    def forward(self, x, edge_index, batch, device):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        xg = self.main_encoder(x, edge_index, batch)
        return xg
    
    def __repr__(self):
        return self.__class__.__name__

