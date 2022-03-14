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

class VSCEncoder_NRI(MessagePassing2):
    """ MLP Encoder from https://github.com/ethanfetaya/NRI using Pytorch Geometric """
    def __init__(self, n_in, n_hid, n_nodes, skip=True):
        super(VSCEncoder_NRI, self).__init__(aggr='add') # Eq 7 aggr part
        self.mlp_eq5_embedding = MLPBlock(n_in, n_hid, n_hid)
        self.mlp_eq6 = MLPBlock(n_hid*2, n_hid, n_hid)
        self.mlp_eq7 = MLPBlock(n_hid, n_hid, n_hid)
        self.mlp_eq8_skipcon = MLPBlock(n_hid*3, n_hid, n_hid)
        
        self.fc_out = nn.Linear(n_hid, 2)
        self.fc_log_spike = nn.Linear(n_hid, 1)
        self.fc_out_lam = nn.Linear(n_hid, 1)
        self.fc_log_spike_lam = nn.Linear(n_hid, 1)
        self.skip = skip

        self.n_nodes = n_nodes
        self.n_in = n_in # [dim_in_feature * tsteps]
        self.n_edges = n_nodes * (n_nodes-1)

        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, edge_index, one_hop=True):
        # Eq 5
        x = self.mlp_eq5_embedding(inputs) # [sims * nodes, hid_dim]

        # x: aggr_msg, x_edge: message output for skip-con
        x_n1, x_e1 = self.propagate(edge_index, x=x, skip=not self.skip) #[sims * nodes, hid_dim], [sims * edges, hid_dim] 
        x_n2, x_e2 = self.propagate(edge_index, x=x_n1, skip=self.skip, x_skip=x_e1) #[batch_size * edges, hid_dim]

        if one_hop:
            mu_sig = self.fc_out(x_e1) #[batch_size * nodes, node_features]
            log_spike = -F.relu(self.fc_log_spike(x_e1))
            return mu_sig, log_spike

        else:
            mu_sig = self.fc_out(x_e2) #[batch_size * edges, edge_features]
            log_spike = -F.relu(self.fc_log_spike(x_e2))
            return mu_sig, log_spike
            

    def message(self, x_i, x_j, skip, x_skip=None):
        # Eq 6
        if not skip:
            x_edge = torch.cat([x_i, x_j], dim=-1)
            x_edge = self.mlp_eq6(x_edge)
        # Eq 8
        else:
            x_edge = torch.cat([x_i, x_j, x_skip], dim=-1)
            x_edge = self.mlp_eq8_skipcon(x_edge)
        return x_edge

    def update(self, aggr_out):
        # Eq 7 MLP part
        new_embedding = self.mlp_eq7(aggr_out)
        return new_embedding

class MLPDecoder_NRI(MessagePassing):
    """ MLP Encoder from https://github.com/ethanfetaya/NRI using Pytorch Geometric.\n
    Edge types are not explictly hard-coded by default.\n
    To explicitly hard code first edge-type as 'non-edge', set 'edge_type_explicit = True' 
    in config.yml.\n """
    def __init__(self, features_dims, msg_hid, msg_out, 
                 n_hid, config, do_prob=0.):
        super(MLPDecoder_NRI, self).__init__(aggr='add')
        self.fc1_eq10 = nn.Linear(2*features_dims, msg_hid)
        self.fc2_eq10 = nn.Linear(msg_hid, msg_out)
        self.eq10_out_dim = msg_out

        self.fc1_eq11 = nn.Linear(features_dims + msg_out, n_hid)
        self.fc2_eq11 = nn.Linear(n_hid, n_hid)
        self.fc3_eq11 = nn.Linear(n_hid, features_dims)
        self.dropout_prob = do_prob

        #config (num_nodes, node_feature_dim, num_timesteps, pred_step, num_chunk, device)
        self.num_nodes = config.get('num_nodes')
        self.node_feature_dim = config.get('num_node_features')
        self.num_timesteps = config.get('num_timesteps')
        self.edge_types = config.get('edge_types')
        self.pred_step = config.get('pred_step')
        self.num_chunk = np.ceil(config.get('num_timesteps') / 
                                 config.get('pred_step')).astype(int)

    def single_step_pred(self, idxed_input, edge_index, z):
        # idxed_input [sims * nodes, idxed_tsteps, features] | z [edges, edge_types] 
        idxed_input = idxed_input.transpose(0,1).contiguous()
        x = self.propagate(edge_index, x=idxed_input, z=z, indexed_input=idxed_input)
        x = x.transpose(0,1).contiguous()

        return x #[sims * nodes, tsteps, features]

    def forward(self, inputs, edge_index, z, pred_steps):
        ''' '''
        # inputs [nodes, tsteps * features] / z [edges, edge_types]
        inputs = inputs.view(-1, self.num_timesteps, self.node_feature_dim)

        assert(pred_steps <= self.num_timesteps)
        preds = []

        # inputs [sims * nodes, tsteps, features]
        idxed_inputs = inputs[:, 0::pred_steps, :]

        for _ in range(0, pred_steps):
            idxed_inputs = self.single_step_pred(idxed_inputs, edge_index, z)
            preds.append(idxed_inputs)
            
        pred_all = torch.stack(preds, dim=2) # sims*nodes, num_chunks, pred_step, feature_dim
        pred_all = pred_all.view(-1, self.num_chunk*pred_steps, self.node_feature_dim)
        return pred_all[:, :self.num_timesteps - 1, :] #[sims * nodes, tsteps, features]

    # Eq 10
    def message(self, x_i, x_j, z):        
        x_edge = torch.cat([x_i, x_j], dim=-1) #[sims * edges, features]
        msg = F.relu(self.fc1_eq10(x_edge))
        msg = F.dropout(msg, p=self.dropout_prob)
        msg = F.relu(self.fc2_eq10(msg))
        msg = msg * z #element-wise product with broadcast
        return msg

    # Eq 11
    def update(self, aggr_out, indexed_input):
        x_cat = torch.cat([indexed_input, aggr_out], dim=-1)
        feature_diff = F.dropout(F.relu(self.fc1_eq11(x_cat)), p=self.dropout_prob)
        feature_diff = F.dropout(F.relu(self.fc2_eq11(feature_diff)), p=self.dropout_prob)
        feature_diff = self.fc3_eq11(feature_diff)
        return indexed_input + feature_diff

class MLPDecoder_NRI_Poisson(MessagePassing):
    """ MLP Encoder from https://github.com/ethanfetaya/NRI using Pytorch Geometric.\n
    Edge types are not explictly hard-coded by default.\n
    To explicitly hard code first edge-type as 'non-edge', set 'edge_type_explicit = True' 
    in config.yml.\n """
    def __init__(self, features_dims, msg_hid, msg_out, 
                 n_hid, config, do_prob=0.):
        super(MLPDecoder_NRI_Poisson, self).__init__(aggr='add')
        self.fc1_eq10 = nn.Linear(2*features_dims, msg_hid)
        self.fc2_eq10 = nn.Linear(msg_hid, msg_out)
        self.eq10_out_dim = msg_out

        self.fc1_eq11 = nn.Linear(features_dims + msg_out, n_hid)
        self.fc2_eq11 = nn.Linear(n_hid, n_hid)
        self.fc3_eq11 = nn.Linear(n_hid, features_dims)
        self.dropout_prob = do_prob

        #config (num_nodes, node_feature_dim, num_timesteps, pred_step, num_chunk, device)
        self.num_nodes = config.get('num_nodes')
        self.node_feature_dim = config.get('num_node_features')
        self.num_timesteps = config.get('num_timesteps')
        self.edge_types = config.get('edge_types')
        self.pred_step = config.get('pred_step')
        self.num_chunk = np.ceil(config.get('num_timesteps') / 
                                 config.get('pred_step')).astype(int)

    def single_step_pred(self, idxed_input, edge_index, z):
        # idxed_input [sims * nodes, idxed_tsteps, features] | z [edges, edge_types] 
        idxed_input = idxed_input.transpose(1,0).contiguous()
        x = self.propagate(edge_index, x=idxed_input, z=z, indexed_input=idxed_input)
        x = x.transpose(1,0).contiguous()
        return x #[sims * nodes, tsteps, features]

    def forward(self, inputs, edge_index, z, pred_steps):
        ''' '''
        # inputs [nodes, tsteps *features] / z [edges, edge_types]
        inputs = inputs.view(-1, self.num_timesteps, self.node_feature_dim)

        assert(pred_steps <= self.num_timesteps)
        preds = []

        # inputs [sims * nodes, tsteps, features]
        idxed_inputs = inputs[:, 0::pred_steps, :]

        for _ in range(0, pred_steps):
            idxed_inputs = self.single_step_pred(idxed_inputs, edge_index, z)
            preds.append(idxed_inputs)
            
        pred_all = torch.stack(preds, dim=2) # sims*nodes, num_chunks, pred_step, feature_dim
        pred_all = pred_all.view(-1, self.num_chunk*pred_steps, self.node_feature_dim)
        #pred_all = F.relu(pred_all)
        return pred_all[:, :self.num_timesteps - 1, :] #[sims * nodes, tsteps, features]

    # Eq 10
    def message(self, x_i, x_j, z):        
        x_edge = torch.cat([x_i, x_j], dim=-1) #[sims * edges, features]
        msg = F.relu(self.fc1_eq10(x_edge))
        msg = F.dropout(msg, p=self.dropout_prob)
        msg = F.relu(self.fc2_eq10(msg))
        msg = msg * z #element-wise product with broadcast
        return msg

    # Eq 11
    def update(self, aggr_out, indexed_input):
        x_cat = torch.cat([indexed_input, aggr_out], dim=-1)
        feature_diff = F.dropout(F.relu(self.fc1_eq11(x_cat)), p=self.dropout_prob)
        feature_diff = F.dropout(F.relu(self.fc2_eq11(feature_diff)), p=self.dropout_prob)
        feature_diff = self.fc3_eq11(feature_diff)
        return feature_diff

class RNNDecoder_NRI(MessagePassing):
    def __init__(self, feature_dim, edge_types, n_hid, config, num_timesteps, device, do_prob=0.):
        super(RNNDecoder_NRI, self).__init__(aggr='add') # Eq 14
        self.fc1_eq13 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.fc2_eq13 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_dim = n_hid
        self.do_prob = do_prob
        # Eq 15
        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(feature_dim, n_hid, bias=True)
        self.input_i = nn.Linear(feature_dim, n_hid, bias=True)
        self.input_n = nn.Linear(feature_dim, n_hid, bias=True)
        # Eq 16
        self.fc1_eq16 = nn.Linear(n_hid, n_hid)
        self.fc2_eq16 = nn.Linear(n_hid, n_hid)
        self.fc3_eq16 = nn.Linear(n_hid, feature_dim)
        #config
        self.node_feature_dim = config.get('num_node_features')
        self.num_timesteps = num_timesteps
        self.device = device

    def forward(self, inputs, edge_index, z, pred_steps=1):
        #[num_graphs * num_nodes, tsteps, feature_dim]
        inputs = inputs.reshape(-1, self.num_timesteps, self.node_feature_dim)
        num_chunk = np.ceil(self.num_timesteps / pred_steps).astype(int)

        assert(pred_steps <= self.num_timesteps)
        pred_all = []
        #[num_graphs * num_nodes, num_chunks, hidden]
        hidden_node_init = nn.Parameter(torch.zeros(num_chunk, inputs.shape[0], self.msg_out_dim)) 
        hidden_node_init = hidden_node_init.to(self.device)

        #[num_graphs * num_nodes, num_chunks, d_f] > [num_chunks, num_graphs * num_nodes, d_f]
        ins_0 = inputs[:, 0::pred_steps, :]
        ins_0 = ins_0.transpose(0,1).contiguous()
        for step in range(pred_steps):
            '''
            if (step % pred_steps) == 0:
                ins = inputs[:, step, :] #[num_graphs * num_nodes * num_chunks, 1]
            else:
                ins = pred_all[-1]
            '''
            if step == 0:
                pred, hidden = self.propagate(edge_index, x=hidden_node_init, z=z,
                                              inputs=ins_0, hidden=hidden_node_init) #initial step: chunk
            else:
                pred, hidden = self.propagate(edge_index, x=hidden, z=z, inputs=pred_all[-1], hidden=hidden)
            pred_all.append(pred) #[num_chunks, num_graphs * num_nodes, d_f]
        preds = torch.stack(pred_all, dim=1) #[num_chunks, pred_steps, num_graphs * num_nodes, d_f]
        preds = preds.reshape(num_chunk * pred_steps, -1, self.node_feature_dim) #[tsteps, num_g * num_n, d_f]
        preds = preds.transpose(0,1).contiguous() #[num_g * num_n, tsteps, d_f]
        #preds = F.relu(preds)
        return preds[:, :self.num_timesteps - 1, :]

    def message(self, x_i, x_j, z):
        x_edge = torch.cat((x_i, x_j), dim=-1)
        norm = float(len(self.fc1_eq13))
        for i in range(len(self.fc1_eq13)):
            if i == 0:
                msg = torch.tanh(self.fc1_eq13[i](x_edge))
                msg = F.dropout(msg, p=self.do_prob)
                msg = torch.tanh(self.fc2_eq13[i](msg))
                msg = msg * z
                all_msgs = msg / norm
            else:
                msg = torch.tanh(self.fc1_eq13[i](x_edge))
                msg = F.dropout(msg, p=self.do_prob)
                msg = torch.tanh(self.fc2_eq13[i](msg))
                msg = msg * z
                all_msgs = all_msgs + (msg / norm)
        return all_msgs

    def update(self, aggr_msg, inputs, hidden):
        # Eq 15
        r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(aggr_msg))
        i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(aggr_msg))
        n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(aggr_msg))
        hidden = (1 - i) * n + i * hidden
        # Eq 16
        pred = F.dropout(F.relu(self.fc1_eq16(hidden)), p=self.do_prob)
        pred = F.dropout(F.relu(self.fc2_eq16(pred)), p=self.do_prob)
        pred = self.fc3_eq16(pred)
        #pred = inputs + pred
        # mu, hidden(t+1)
        return pred, hidden

class RNNDecoder_NRI_BurnIn(MessagePassing):
    def __init__(self, feature_dim, n_hid, config, pre_steps, pred_steps, device, do_prob=0.):
        super(RNNDecoder_NRI_BurnIn, self).__init__(aggr='add') # Eq 14
        self.fc1_eq13 = nn.Linear(2 * n_hid, n_hid)
        self.fc2_eq13 = nn.Linear(n_hid, n_hid)
        self.msg_out_dim = n_hid
        self.do_prob = do_prob
        # Eq 15
        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(pre_steps + feature_dim, n_hid, bias=True)
        self.input_i = nn.Linear(pre_steps + feature_dim, n_hid, bias=True)
        self.input_n = nn.Linear(pre_steps + feature_dim, n_hid, bias=True)
        # Eq 16
        self.fc1_eq16 = nn.Linear(n_hid, n_hid)
        self.fc2_eq16 = nn.Linear(n_hid, n_hid)
        self.fc3_eq16 = nn.Linear(n_hid, feature_dim)
        # config
        self.node_feature_dim = config.get('num_node_features')
        self.pre_steps = pre_steps
        self.pred_steps = pred_steps
        self.window_size = pre_steps + pred_steps - 1
        self.device = device

    def forward(self, inputs, edge_index, z, pred_steps=1):
        # [num_graphs * num_nodes, tsteps, feature_dim]
        # num_timesteps should be [window_size = pre_steps + pred_steps - 1]
        inputs = inputs.reshape(-1, self.window_size, self.node_feature_dim)
        batch_size = inputs.shape[0]
        pre_steps = self.pre_steps

        assert(pred_steps <= self.window_size)
        pred_all = []
        #[num_graphs * num_nodes, hidden]
        hidden_node_init = nn.Parameter(torch.zeros(batch_size, self.msg_out_dim))
        hidden_node_init = hidden_node_init.to(self.device)
        log_lam_init = nn.Parameter(torch.zeros(batch_size, self.node_feature_dim))
        log_lam_init = log_lam_init.to(self.device)

        for step in range(pred_steps):
            #ins = inputs[:, step:step+pre_steps, :] #[num_graphs * num_nodes, pre_steps, d_f]
            ins = inputs[:, step:pre_steps, :]
            ins = ins.reshape(batch_size, -1) #[num_graphs * num_nodes, pre_steps * d_f]
            '''
            if (step % pred_steps) == 0:
                ins = inputs[:, step, :] #[num_graphs * num_nodes * num_chunks, 1]
            else:
                ins = pred_all[-1]
            '''
            #initial step
            if step == 0:
                pred, hidden = self.propagate(edge_index, x=hidden_node_init, z=z, inputs=ins, 
                                              hidden=hidden_node_init, log_lam=log_lam_init)
                y_hat = torch.poisson(pred)
            else:
                ins_cat = torch.cat((ins, y_hat), dim=-1)
                pred, hidden = self.propagate(edge_index, x=hidden, z=z, inputs=ins_cat, 
                                              hidden=hidden, log_lam=pred_all[-1])
                y_hat = torch.cat((y_hat, torch.poisson(pred)), dim=-1)
            # pred: log(lambda_(t+1))
            pred_all.append(pred) #[num_graphs * num_nodes, d_f]
            
        preds = torch.stack(pred_all, dim=1) #[num_graphs * num_nodes, pred_steps, d_f]
        return preds

    def message(self, x_i, x_j, z):
        x_edge = torch.cat((x_i, x_j), dim=-1)

        msg = torch.tanh(self.fc1_eq13(x_edge))
        msg = F.dropout(msg, p=self.do_prob)
        msg = torch.tanh(self.fc2_eq13(msg))
        msg = msg * z
        all_msgs = msg

        return all_msgs

    def update(self, aggr_msg, inputs, hidden, log_lam):
        # The input to the message passing operation is the recurrent hidden state at the previous time step.
        # Eq 15 / Inputs [x_t, MSG_t, log_lam_(t+1)]
        inputs_cat = torch.cat((inputs, log_lam), dim=-1)
        r = torch.sigmoid(self.input_r(inputs_cat) + self.hidden_r(aggr_msg))
        i = torch.sigmoid(self.input_i(inputs_cat) + self.hidden_i(aggr_msg))
        n = torch.tanh(self.input_n(inputs_cat) + r * self.hidden_h(aggr_msg))
        hidden = (1 - i) * n + i * hidden
        # Eq 16
        pred = F.dropout(F.relu(self.fc1_eq16(hidden)), p=self.do_prob)
        pred = F.dropout(F.relu(self.fc2_eq16(pred)), p=self.do_prob)
        pred = self.fc3_eq16(pred)
        # mu, hidden(t+1)
        return pred, hidden

class RNNDecoder_NRI_many_to_one(MessagePassing):
    def __init__(self, feature_dim, n_hid, config, pre_steps, pred_steps, device, do_prob=0.):
        super(RNNDecoder_NRI_many_to_one, self).__init__(aggr='add') # Eq 14
        self.fc1_eq13 = nn.Linear(2 * n_hid, n_hid)
        self.fc2_eq13 = nn.Linear(n_hid, n_hid)
        self.msg_out_dim = n_hid
        self.do_prob = do_prob
        # Eq 15
        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(pre_steps + feature_dim, n_hid, bias=True)
        self.input_i = nn.Linear(pre_steps + feature_dim, n_hid, bias=True)
        self.input_n = nn.Linear(pre_steps + feature_dim, n_hid, bias=True)
        # Eq 16
        self.fc1_eq16 = nn.Linear(n_hid, n_hid)
        self.fc2_eq16 = nn.Linear(n_hid, n_hid)
        self.fc3_eq16 = nn.Linear(n_hid, feature_dim)
        # config
        self.node_feature_dim = config.get('num_node_features')
        self.pre_steps = pre_steps
        self.pred_steps = pred_steps
        self.window_size = pre_steps + pred_steps - 1
        self.device = device

    def forward(self, inputs, edge_index, z, pred_steps=1):
        # [num_graphs * num_nodes, tsteps, feature_dim]
        # num_timesteps should be [window_size = pre_steps + pred_steps - 1]
        inputs = inputs.reshape(-1, self.window_size, self.node_feature_dim)
        batch_size = inputs.shape[0]
        pre_steps = self.pre_steps

        assert(pred_steps <= self.window_size)
        pred_all = []
        #[num_graphs * num_nodes, hidden]
        hidden_node_init = nn.Parameter(torch.zeros(batch_size, self.msg_out_dim))
        hidden_node_init = hidden_node_init.to(self.device)
        log_lam_init = nn.Parameter(torch.zeros(batch_size, self.node_feature_dim))
        log_lam_init = log_lam_init.to(self.device)

        for step in range(pred_steps):
            ins = inputs[:, step:step+pre_steps, :] #[num_graphs * num_nodes, pre_steps, d_f]
            #ins = inputs[:, step:pre_steps, :]
            ins = ins.reshape(batch_size, -1) #[num_graphs * num_nodes, pre_steps * d_f]
            '''
            if (step % pred_steps) == 0:
                ins = inputs[:, step, :] #[num_graphs * num_nodes * num_chunks, 1]
            else:
                ins = pred_all[-1]
            '''
            #initial step
            if step == 0:
                pred, hidden = self.propagate(edge_index, x=hidden_node_init, z=z, inputs=ins, 
                                              hidden=hidden_node_init, log_lam=log_lam_init)
                #y_hat = torch.poisson(pred)
            else:
                #ins_cat = torch.cat((ins, y_hat), dim=-1)
                pred, hidden = self.propagate(edge_index, x=hidden, z=z, inputs=ins, 
                                              hidden=hidden, log_lam=pred_all[-1])
                #y_hat = torch.cat((y_hat, torch.poisson(pred)), dim=-1)
            # pred: log(lambda_(t+1))
            pred_all.append(pred) #[num_graphs * num_nodes, d_f]
            
        preds = torch.stack(pred_all, dim=1) #[num_graphs * num_nodes, pred_steps, d_f]
        return preds

    def message(self, x_i, x_j, z):
        x_edge = torch.cat((x_i, x_j), dim=-1)

        msg = torch.tanh(self.fc1_eq13(x_edge))
        msg = F.dropout(msg, p=self.do_prob)
        msg = torch.tanh(self.fc2_eq13(msg))
        msg = msg * z
        all_msgs = msg

        return all_msgs

    def update(self, aggr_msg, inputs, hidden, log_lam):
        # The input to the message passing operation is the recurrent hidden state at the previous time step.
        # Eq 15 / Inputs [x_t, MSG_t, log_lam_(t+1)]
        inputs_cat = torch.cat((inputs, log_lam), dim=-1)
        r = torch.sigmoid(self.input_r(inputs_cat) + self.hidden_r(aggr_msg))
        i = torch.sigmoid(self.input_i(inputs_cat) + self.hidden_i(aggr_msg))
        n = torch.tanh(self.input_n(inputs_cat) + r * self.hidden_h(aggr_msg))
        hidden = (1 - i) * n + i * hidden
        # Eq 16
        pred = F.dropout(F.relu(self.fc1_eq16(hidden)), p=self.do_prob)
        pred = F.dropout(F.relu(self.fc2_eq16(pred)), p=self.do_prob)
        pred = self.fc3_eq16(pred)
        # mu, hidden(t+1)
        return pred, hidden

class RNNDecoder_NRI_m2o_perturb(MessagePassing):
    def __init__(self, feature_dim, n_hid, config, p1_bs, history, nodes, device, do_prob=0.):
        super(RNNDecoder_NRI_m2o_perturb, self).__init__(aggr='add') # Eq 14
        self.fc1_eq13 = nn.Linear(2 * n_hid, n_hid)
        self.fc2_eq13 = nn.Linear(n_hid, n_hid)
        self.msg_out_dim = n_hid
        self.phase1_batchsize = p1_bs
        self.do_prob = do_prob
        # Eq 15
        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(history + feature_dim, n_hid, bias=True)
        self.input_i = nn.Linear(history + feature_dim, n_hid, bias=True)
        self.input_n = nn.Linear(history + feature_dim, n_hid, bias=True)
        # Eq 16
        self.fc1_eq16 = nn.Linear(n_hid, n_hid)
        self.fc2_eq16 = nn.Linear(n_hid, n_hid)
        self.fc3_eq16 = nn.Linear(n_hid, feature_dim)
        # config
        self.node_feature_dim = config.get('num_node_features')
        self.history = history
        self.nodes = nodes
        self.device = device

    def forward(self, inputs, edge_index, z, pred_steps=1):
        # [num_nodes, phase1_batch, tsteps, feature_dim]
        # num_timesteps should be [window_size = history + pred_steps - 1]
        inputs = inputs.reshape(self.nodes, self.phase1_batchsize, -1, self.node_feature_dim)
        num_nodes = inputs.shape[0]
        history = self.history
        batch_size = self.phase1_batchsize

        assert(pred_steps <= inputs.shape[2])
        pred_all = []
        #[num_graphs * num_nodes, hidden]
        hidden_node_init = nn.Parameter(torch.zeros(num_nodes, batch_size * self.msg_out_dim)) #Changed
        hidden_node_init = hidden_node_init.to(self.device)
        log_lam_init = nn.Parameter(torch.zeros(num_nodes, batch_size, self.node_feature_dim)) #Changed
        log_lam_init = log_lam_init.to(self.device)

        for step in range(pred_steps):
            ins = inputs[:, :, step:step+history, :] #[num_nodes, p1_bs, history, d_f]
            #ins = inputs[:, step:history, :]
            ins = ins.reshape(num_nodes, batch_size, -1) #[num_nodes, p1_bs, history * d_f]
            '''
            if (step % pred_steps) == 0:
                ins = inputs[:, step, :] #[num_graphs * num_nodes * num_chunks, 1]
            else:
                ins = pred_all[-1]
            '''
            #initial step
            if step == 0:
                pred, hidden = self.propagate(edge_index, x=hidden_node_init, z=z, inputs=ins, 
                                              hidden=hidden_node_init, log_lam=log_lam_init)
                #y_hat = torch.poisson(pred)
            else:
                #ins_cat = torch.cat((ins, y_hat), dim=-1)
                pred, hidden = self.propagate(edge_index, x=hidden, z=z, inputs=ins, 
                                              hidden=hidden, log_lam=pred_all[-1])
                #y_hat = torch.cat((y_hat, torch.poisson(pred)), dim=-1)
            # pred: log(lambda_(t+1))
            pred_all.append(pred) #[num_nodes, p1_bs, d_f]
            
        preds = torch.stack(pred_all, dim=2) #[num_nodes, p1_bs, pred_steps, d_f]
        return preds

    def message(self, x_i, x_j, z):
        x_i, x_j = (x_i.reshape(-1, self.phase1_batchsize, self.msg_out_dim), 
                    x_j.reshape(-1, self.phase1_batchsize, self.msg_out_dim))
        x_edge = torch.cat((x_i, x_j), dim=-1)
        # [edges, bs, hdim]
        msg = torch.tanh(self.fc1_eq13(x_edge))
        msg = F.dropout(msg, p=self.do_prob)
        msg = torch.tanh(self.fc2_eq13(msg))
        msg = msg * z.reshape(-1, 1, 1)
        all_msgs = msg 

        all_msgs = all_msgs.reshape(-1, self.phase1_batchsize * self.msg_out_dim)
        return all_msgs

    def update(self, aggr_msg, inputs, hidden, log_lam):
        # The input to the message passing operation is the recurrent hidden state at the previous time step.
        # Eq 15 / Inputs [x_t, MSG_t, log_lam_(t+1)] [num_nodes, p1_bs, history * d_f], [num_nodes, p1_bs, 1]
        inputs_cat = torch.cat((inputs, log_lam), dim=-1) # why log_lam? > different data type || why concat?
        aggr_msg = aggr_msg.reshape(-1, self.phase1_batchsize, self.msg_out_dim)
        hidden = hidden.reshape(-1, self.phase1_batchsize, self.msg_out_dim)

        r = torch.sigmoid(self.input_r(inputs_cat) + self.hidden_r(aggr_msg))
        i = torch.sigmoid(self.input_i(inputs_cat) + self.hidden_i(aggr_msg))
        n = torch.tanh(self.input_n(inputs_cat) + r * self.hidden_h(aggr_msg))
        
        hidden = (1 - i) * n + i * hidden
        # Eq 16
        pred = F.dropout(F.relu(self.fc1_eq16(hidden)), p=self.do_prob)
        pred = F.dropout(F.relu(self.fc2_eq16(pred)), p=self.do_prob)
        pred = self.fc3_eq16(pred)
        hidden = hidden.reshape(-1, self.phase1_batchsize * self.msg_out_dim)
        # mu, hidden(t+1)
        return pred, hidden

class RNNDecoder_NRI_circshift(MessagePassing):
    def __init__(self, feature_dim, n_hid, config, p1_bs, history, nodes, device, do_prob=0.):
        super(RNNDecoder_NRI_circshift, self).__init__(aggr='add') # Eq 14
        self.fc1_eq13 = nn.Linear(2 * n_hid, n_hid)
        self.fc2_eq13 = nn.Linear(n_hid, n_hid)
        self.msg_out_dim = n_hid
        self.phase1_batchsize = p1_bs
        self.do_prob = do_prob
        # Eq 15
        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(history + feature_dim, n_hid, bias=True)
        self.input_i = nn.Linear(history + feature_dim, n_hid, bias=True)
        self.input_n = nn.Linear(history + feature_dim, n_hid, bias=True)
        # Eq 16
        self.fc1_eq16 = nn.Linear(n_hid, n_hid)
        self.fc2_eq16 = nn.Linear(n_hid, n_hid)
        self.fc3_eq16 = nn.Linear(n_hid, feature_dim)
        # config
        self.node_feature_dim = config.get('num_node_features')
        self.history = history
        self.nodes = nodes
        self.device = device

    def forward(self, inputs, edge_index, z, pred_steps=1):
        # [num_nodes, phase1_batch, tsteps, feature_dim]
        # num_timesteps should be [window_size = history + pred_steps - 1]
        inputs = inputs.reshape(self.nodes, self.phase1_batchsize, -1, self.node_feature_dim)
        num_nodes = inputs.shape[0]
        history = self.history
        batch_size = self.phase1_batchsize

        assert(pred_steps <= inputs.shape[2])
        pred_all = []
        #[num_graphs * num_nodes, hidden]
        hidden_node_init = nn.Parameter(torch.zeros(num_nodes, batch_size * self.msg_out_dim)) #Changed
        hidden_node_init = hidden_node_init.to(self.device)
        log_lam_init = nn.Parameter(torch.zeros(num_nodes, batch_size, self.node_feature_dim)) #Changed
        log_lam_init = log_lam_init.to(self.device)

        for step in range(pred_steps):
            ins = inputs[:, :, step:step+history, :] #[num_nodes, p1_bs, history, d_f]
            #ins = inputs[:, step:history, :]
            ins = ins.reshape(num_nodes, batch_size, -1) #[num_nodes, p1_bs, history * d_f]
            '''
            if (step % pred_steps) == 0:
                ins = inputs[:, step, :] #[num_graphs * num_nodes * num_chunks, 1]
            else:
                ins = pred_all[-1]
            '''
            #initial step
            if step == 0:
                pred, hidden = self.propagate(edge_index, x=hidden_node_init, z=z, inputs=ins, 
                                              hidden=hidden_node_init, log_lam=log_lam_init)
                #y_hat = torch.poisson(pred)
            else:
                #ins_cat = torch.cat((ins, y_hat), dim=-1)
                pred, hidden = self.propagate(edge_index, x=hidden, z=z, inputs=ins, 
                                              hidden=hidden, log_lam=pred_all[-1])
                #y_hat = torch.cat((y_hat, torch.poisson(pred)), dim=-1)
            # pred: log(lambda_(t+1))
            pred_all.append(pred) #[num_nodes, p1_bs, d_f]
            
        preds = torch.stack(pred_all, dim=2) #[num_nodes, p1_bs, pred_steps, d_f]
        return preds

    def message(self, x_i, x_j):
        x_i, x_j = (x_i.reshape(-1, self.phase1_batchsize, self.msg_out_dim), 
                    x_j.reshape(-1, self.phase1_batchsize, self.msg_out_dim))
        x_edge = torch.cat((x_i, x_j), dim=-1)
        # [edges, bs, hdim]
        msg = torch.tanh(self.fc1_eq13(x_edge))
        msg = F.dropout(msg, p=self.do_prob)
        msg = torch.tanh(self.fc2_eq13(msg))
        all_msgs = msg 

        all_msgs = all_msgs.reshape(-1, self.phase1_batchsize * self.msg_out_dim)
        return all_msgs

    def update(self, aggr_msg, inputs, hidden, log_lam, z):
        # The input to the message passing operation is the recurrent hidden state at the previous time step.
        # Eq 15 / Inputs [x_t, MSG_t, log_lam_(t+1)] [num_nodes, p1_bs, history * d_f], [num_nodes, p1_bs, 1]
        inputs_cat = torch.cat((inputs, log_lam), dim=-1) # why log_lam? > different data type || why concat?
        aggr_msg = aggr_msg.reshape(-1, self.phase1_batchsize, self.msg_out_dim)
        inputs_cat = inputs_cat * z.reshape(-1, 1, 1)
        aggr_msg = aggr_msg * z.reshape(-1, 1, 1)
        hidden = hidden.reshape(-1, self.phase1_batchsize, self.msg_out_dim)

        r = torch.sigmoid(self.input_r(inputs_cat) + self.hidden_r(aggr_msg))
        i = torch.sigmoid(self.input_i(inputs_cat) + self.hidden_i(aggr_msg))
        n = torch.tanh(self.input_n(inputs_cat) + r * self.hidden_h(aggr_msg))
        
        hidden = (1 - i) * n + i * hidden
        # Eq 16
        pred = F.dropout(F.relu(self.fc1_eq16(hidden)), p=self.do_prob)
        pred = F.dropout(F.relu(self.fc2_eq16(pred)), p=self.do_prob)
        pred = self.fc3_eq16(pred)
        hidden = hidden.reshape(-1, self.phase1_batchsize * self.msg_out_dim)
        # mu, hidden(t+1)
        return pred, hidden

