''' NRI with PyTorch Geometric using Spike-Slab Prior '''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from SourceModified.message_passing import MessagePassing2

import numpy as np

#MLP Block with XavierInit & BatchNorm1D (n_in, n_hid, n_out)
class MLPBlock(nn.Module):
    ''' MLP Block with XavierInit & BatchNorm1D (n_in, n_hid, n_out) '''
    def __init__(self, n_in, n_hid, n_out, prob_drop=0.):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.prob_drop = prob_drop
        self.init_weights()
    
    def init_weights(self):
        ''' Init weight with Xavier Nornalization '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        sizes = inputs.size()
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(sizes)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.prob_drop, training=self.training) #training
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class gts_adj_inf(MessagePassing2):
    """ GTS style structure inference model using Pytorch Geometric """
    def __init__(self, n_nodes, n_hid, out_channel, ks_x, ks_h, st_x, st_h, total_steps):
        super(gts_adj_inf, self).__init__(aggr='add') # Eq 7 aggr part
        self.conv_x = nn.Conv1d(1, out_channel, ks_x, stride=st_x)
        self.conv_h = nn.Conv1d(out_channel, 2*out_channel, ks_h, stride=st_h)
        self.convx_dim = int(((total_steps - ks_x) / st_x) + 1)
        self.convh_dim = int(((self.convx_dim - ks_h) / st_h) + 1) * out_channel * 2

        self.bn_conv_x = nn.BatchNorm1d(out_channel)
        self.bn_conv_h = nn.BatchNorm1d(2*out_channel)
        self.bn_h = nn.BatchNorm1d(n_hid)

        self.fc_conv_h = nn.Linear(self.convh_dim, n_hid)
        self.fc_cat = nn.Linear(2*n_hid, n_hid)
        self.fc_out_adj = nn.Linear(n_hid, 2)
        self.fc_out_corr = nn.Linear(n_hid, 1)
        
        self.fc_e2n = nn.Linear(n_hid, n_hid)

        self.nodes = n_nodes
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, edge_index):
        # Eq 5
        x = inputs.reshape(self.nodes, 1, -1)
        x = self.conv_x(x) # [sims * nodes, hid_dim]
        x = F.relu(x)
        x = self.bn_conv_x(x)

        x = self.conv_h(x)
        x = F.relu(x)
        x = self.bn_conv_h(x)
        x = x.reshape(self.nodes, -1)

        x = self.fc_conv_h(x)
        x = F.relu(x)
        x = self.bn_h(x)

        _, x_e = self.propagate(edge_index, x=x)
        x_adj = self.fc_out_adj(x_e)
        x_corr = self.fc_out_corr(x_e)
        return x_adj, x_corr
            
    def message(self, x_i, x_j):
        x_edge = torch.cat([x_i, x_j], dim=-1)
        x_edge = self.fc_cat(x_edge)
        x_edge = F.relu(x_edge)
        return x_edge

    def update(self, aggr_out):
        new_embedding = self.fc_e2n(aggr_out)
        return new_embedding

class gts_adj_inf_cs(MessagePassing2):
    """ GTS style structure inference model using Pytorch Geometric """
    def __init__(self, n_nodes, n_hid, out_channel, ks_x, ks_h, st_x, st_h, total_steps):
        super(gts_adj_inf_cs, self).__init__(aggr='add') # Eq 7 aggr part
        self.conv_x = nn.Conv1d(1, out_channel, ks_x, stride=st_x)
        self.conv_h = nn.Conv1d(out_channel, 2*out_channel, ks_h, stride=st_h)
        self.convx_dim = int(((total_steps - ks_x) / st_x) + 1)
        self.convh_dim = int(((self.convx_dim - ks_h) / st_h) + 1) * out_channel * 2

        self.bn_conv_x = nn.BatchNorm1d(out_channel)
        self.bn_conv_h = nn.BatchNorm1d(2*out_channel)
        self.bn_h = nn.BatchNorm1d(n_hid)

        self.fc_conv_h = nn.Linear(self.convh_dim, n_hid)
        self.fc_cat = nn.Linear(2*n_hid, n_hid)
        self.fc_out_adj = nn.Linear(n_hid, 2)
        self.fc_out_corr = nn.Linear(n_hid, 1)
        
        self.fc_e2n = nn.Linear(n_hid, n_hid)

        self.nodes = n_nodes
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, edge_index):
        # Eq 5
        x = inputs.reshape(self.nodes, 1, -1)
        x = self.conv_x(x) # [sims * nodes, hid_dim]
        x = F.relu(x)
        x = self.bn_conv_x(x)

        x = self.conv_h(x)
        x = F.relu(x)
        x = self.bn_conv_h(x)
        x = x.reshape(self.nodes, -1)

        x = self.fc_conv_h(x)
        x = F.relu(x)
        x = self.bn_h(x)

        x_n, _ = self.propagate(edge_index, x=x)
        x_adj = self.fc_out_adj(x_n)
        x_corr = self.fc_out_corr(x_n)
        return x_adj, x_corr
            
    def message(self, x_i, x_j):
        x_edge = torch.cat([x_i, x_j], dim=-1)
        x_edge = self.fc_cat(x_edge)
        x_edge = F.relu(x_edge)
        return x_edge

    def update(self, aggr_out):
        new_embedding = self.fc_e2n(aggr_out)
        return new_embedding



'''device = 'cpu'
fully_connected = np.ones((10, 10)) - np.eye(10)
circshift_edge = np.zeros((10, 10))
circshift_edge[:, 0] = 1
circshift_edge[0, :] = 1
circshift_edge[0, 0] = 0

edge_fc = np.where(fully_connected)
edge_fc = np.array([edge_fc[0], edge_fc[1]], dtype=np.int64)
edge_fc = torch.LongTensor(edge_fc)

edge_cs = np.where(circshift_edge)
edge_cs = np.array([edge_cs[0], edge_cs[1]], dtype=np.int64)
edge_cs = torch.LongTensor(edge_cs)
edge_cs, edge_fc = edge_cs.to(device), edge_fc.to(device)

inputs = torch.randn((10, 4800))
model = gts_adj_inf(10, 32, 2, 20, 5, 20, 5, 4800)
x_adj, x_corr = model(inputs, edge_fc)
x_adj_cs, x_corr_cs = model(inputs, edge_cs)
adj = F.gumbel_softmax(x_adj, tau=1, hard=True) #[edges, 2]
adj_cs = F.gumbel_softmax(x_adj_cs, tau=1, hard=True) #[edges, 2]

from utils import *
adj_cs_all = circshift_z(adj_cs[:, 0].clone(), 10, 1, ave_z=True)
corr_cs_all = circshift_z(x_corr_cs, 10, 1, ave_z=True)

adj_sym = make_z_sym_gts(adj[:, 0].clone(), 10) #[edges, 1] > [edges]
adj_sym_cs = make_z_sym_gts(adj_cs_all, 10) #[edges, 1] > [edges]
corr_sym = make_z_sym_gts(x_corr, 10) #[edges, 1] > [edges]
corr_sym_cs = make_z_sym_gts(corr_cs_all, 10) #[edges, 1] > [edges]

corr_cs_all.shape

z_idx = torch.where(adj_sym) 
z_idx_cs = torch.where(adj_sym_cs) 
z_corr = x_corr[z_idx]
z_corr = corr_sym[z_idx_cs]
adj_sym = z_to_adj(adj_sym, 10, 'cpu')
adj_sym_cs = z_to_adj(adj_sym_cs, 10, 'cpu')

adj = adj_to_edge_idx_pyg(adj_sym)
adj.shape
torch.where(adj_sym)
adj_all = get_edgeidx_by_batchsize(adj, 10, 10)'''
