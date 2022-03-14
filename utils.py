from ast import Assert
from numpy.core.fromnumeric import transpose
import torch
import numpy as np
import mat73
from torch.serialization import validate_cuda_device
from generate_data import *
from scipy.optimize import minimize
import generate_spike as gen_spike
import os

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))

def kl_categorical(preds, log_prior, num_nodes, eps=1e-16):
    num_edges = num_nodes * (num_nodes - 1)
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / preds.size(0) * num_edges

def kl_categorical_uniform(preds, num_nodes, num_edge_types, add_const=False, eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    num_edges = num_nodes * (num_nodes - 1)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / preds.size(0) * num_edges # sims * edges

def kl_gaussian(preds_mu, preds_log_var):
    kl_div = 0.5 * (preds_log_var.exp() + preds_mu**2 - 1 - preds_log_var)
    return kl_div.sum() / preds_mu.size(0)

def edge_accuracy(preds, target):
    #preds는 index를 return [#sims, #edges, #edge_types] -> [#sims, #edges] / element -1 dim에서의 max index
    preds_max = preds.argmax(-1) 
    correct = preds_max.float().eq(
        target.float().view_as(preds_max)).cpu().sum()
    
    return np.float(correct) / (target.size(0))

# Too slow...
def edge_accuracy_index(preds, target):
    edge_types = preds.size(-1)
    target_size = target.size(0)

    preds = preds.argmax(-1).float().cpu().detach()
    target = target.float().view_as(preds).cpu().detach().numpy()
    preds = preds.numpy()

    preds_per = np.zeros_like(preds)

    for i in range(edge_types):
        idx = np.where(preds == i)
        permute = np.mean(target[idx] - preds[idx])
        preds_per[idx] = preds[idx] + np.round(permute, 0)

    correct = np.sum(preds_per == target)

    return correct / target_size

def edge_accuracy_con(preds, target):
    edge_types = target.size(-1)
    target_size = target.size(0)

    preds = preds.float().cpu().detach() #continuous value
    target = target.float().view_as(preds).cpu().detach().numpy()
    preds = preds.numpy()

    preds_per = np.zeros_like(preds)

    for i in range(edge_types):
        idx = np.where(preds == i)
        permute = np.mean(target[idx] - preds[idx])
        preds_per[idx] = preds[idx] + np.round(permute, 0)

    correct = np.sum(preds_per == target)

    return correct / target_size

def kl_pseudo(u_star, mu_sig, log_spike, mu_sig_p, log_spike_p, n_edges):
    eps = 1e-10
    batch_size = u_star.shape[0]
    T_p = torch.cat((mu_sig_p, log_spike_p), dim=-1).reshape(-1, n_edges, 3)
    expand_size = [batch_size] + list(T_p.shape)
    T_p = T_p.unsqueeze(0).expand(expand_size)

    u_s = u_star.unsqueeze(-1).unsqueeze(-1).expand_as(T_p)
    x_u = (u_s * T_p).sum(dim=1).reshape(-1, 3)    

    spike = log_spike.exp()
    mu, logvar = mu_sig[:, 0:1], mu_sig[:, 1:2]
    mu_p, logvar_p, log_gamma_p = x_u[:, 0:1], x_u[:, 1:2], x_u[:, 2:3]
    spike_p = log_gamma_p.exp()
    KL_mu_sig = (-0.5 * spike.mul(1 + logvar - logvar_p - 
                 (logvar.exp().sqrt() + (mu - mu_p).pow(2))/(logvar_p.exp().sqrt())))
    KL_spike = (1 - spike) * (torch.log(1 - spike + eps) -  torch.log(1 - spike_p + eps))
    KL_slab = spike * (torch.log(spike + eps) - torch.log(spike_p + eps))
    KL = KL_mu_sig + KL_spike + KL_slab
    
    return KL.sum() / batch_size

def kl_spike_only(mu_sig, log_spike, alpha, device):
    eps = 1e-6
    mu, logvar = mu_sig[:, 0:1], mu_sig[:, 1:2]
    spike = torch.clamp(log_spike.exp(), eps, 1-eps) #[batch * edges, 1]
    alpha = alpha.to(device) #[num_edges] > [1, edges, 1] > [batch, edges, 1] > [batch * edges, 1]
    alpha_expand = alpha.unsqueeze(0)
    alpha_expand = alpha_expand.expand_as(spike).clone() 
    alpha_expand = torch.clamp(alpha_expand, eps, 1 - eps)

    KL_mu_sig = -0.5 * spike.mul(1 + logvar - logvar.exp().sqrt() - mu.pow(2))
    KL_spike = (1 - spike) * (torch.log(1 - spike) - torch.log(1 - alpha_expand))
    KL_slab = spike * (torch.log(spike) - torch.log(alpha_expand))
    KL = KL_mu_sig + KL_slab + KL_spike
    return KL.sum() / mu_sig.shape[0]

def poisson_nll_with_deltaT(log_lambda, tar_spike, delta_T):
    loss_poisson = delta_T * log_lambda.exp() - tar_spike * log_lambda
    return loss_poisson.mean()

def poisson_nll(log_lambda, tar_spike, eps):
    log_lambda = torch.clamp(log_lambda, min=eps, max=1e2)
    loss_poisson = log_lambda.exp() - tar_spike * log_lambda
    return loss_poisson.mean()

def penalty_term(log_spike_p, u_star, alpha, n_edges, device):
    eps = 1e-6
    batch_size = u_star.shape[0]
    expand_size = [batch_size] + list(log_spike_p.reshape(-1, n_edges, 1).shape)
    gamma = log_spike_p.exp().reshape(-1, n_edges, 1).unsqueeze(0).expand(expand_size) # 128 20 20 1

    u_s = u_star.unsqueeze(-1).unsqueeze(-1).expand_as(gamma)
    gamma_bar = (u_s * gamma).sum(1).squeeze() #128 20
    alpha = alpha.to(device) #[num_edges] > [1, edges, 1] > [batch, edges, 1] > [batch * edges, 1]
    alpha_expand = alpha.unsqueeze(0).unsqueeze(-1)
    alpha_expand = alpha_expand.expand(batch_size, n_edges, 1).reshape(-1, 1) 
    alpha_expand = torch.clamp(alpha_expand, eps, 1 - eps)
    KL = (gamma_bar.mul(torch.log(gamma_bar + eps) - torch.log(alpha + eps)).sum() + 
          (1 - gamma_bar).mul(torch.log((1 - gamma_bar) + eps) - torch.log((1 - alpha) + eps)).sum())
    return KL * n_edges / log_spike_p.shape[0]

def load_springs(nodes, edge_types, data_dir):
    if nodes == 5:
        if edge_types == 3:
            tr = spring5_edge3_train(data_dir)
            val = spring5_edge3_valid(data_dir)
            test = spring5_edge3_test(data_dir)
        elif edge_types == 10:
            tr = spr_e10n5_tr(data_dir)
            val = spr_e10n5_val(data_dir)
            test = spr_e10n5_test(data_dir)

    elif nodes == 10:
        if edge_types == 3:
            tr = spr_e3n10_tr(data_dir)
            val = spr_e3n10_val(data_dir)
            test = spr_e3n10_test(data_dir)
        elif edge_types == 10:
            tr = spr_e10n10_tr(data_dir)
            val = spr_e10n10_val(data_dir)
            test = spr_e10n10_test(data_dir)
    
    return tr, val, test

def load_springs_with_alphaD(nodes, edge_types, data_dir, alphaD):
    if nodes == 5:
        if edge_types == 10:
            if alphaD == 0.3:
                tr = spr_n5e10_a03_train(data_dir)
                val = spr_n5e10_a03_valid(data_dir)
                test = spr_n5e10_a03_test(data_dir)
            if alphaD == 0.6:
                tr = spr_n5e10_a06_train(data_dir)
                val = spr_n5e10_a06_valid(data_dir)
                test = spr_n5e10_a06_test(data_dir)
            if alphaD == 0.9:
                tr = spr_n5e10_a09_train(data_dir)
                val = spr_n5e10_a09_valid(data_dir)
                test = spr_n5e10_a09_test(data_dir)     
    if nodes == 10:
        if edge_types == 10:
            if alphaD == 0.3:
                tr = spr_n10e10_a03_train(data_dir)
                val = spr_n10e10_a03_valid(data_dir)
                test = spr_n10e10_a03_test(data_dir)
        if edge_types == 10:
            if alphaD == 0.6:
                tr = spr_n10e10_a06_train(data_dir)
                val = spr_n10e10_a06_valid(data_dir)
                test = spr_n10e10_a06_test(data_dir)  
        if edge_types == 10:
            if alphaD == 0.9:
                tr = spr_n10e10_a09_train(data_dir)
                val = spr_n10e10_a09_valid(data_dir)
                test = spr_n10e10_a09_test(data_dir)                  
    return tr, val, test

def make_z_sym(z, num_nodes, batch_num):
    z = z.reshape(batch_num, num_nodes, num_nodes-1)
    for i in range(num_nodes-1):
        z[:, i+1:num_nodes, i] = z[:, i, i:num_nodes-1]
    z = z.reshape(-1, 1)
    return z

def make_z_sym_nri(z, num_nodes, batch_num, edge_types):
    z = z.reshape(batch_num, num_nodes, num_nodes-1, edge_types)
    for i in range(num_nodes-1):
        z[:, i+1:num_nodes, i, :] = z[:, i, i:num_nodes-1, :]
    z = z.reshape(-1, edge_types)
    return z

def make_z_sym_gts(z, num_nodes, device, mode):
    if mode == 'transpose':
        z = z.reshape(-1)
        adj = torch.zeros(num_nodes, num_nodes).reshape(-1)
        adj = adj.to(device)
        for i in range(num_nodes - 1):
            adj[(num_nodes+1)*i + 1:(num_nodes+1)*i + num_nodes + 1] = z[num_nodes*i: num_nodes*i + num_nodes]
        adj = adj.reshape(num_nodes, num_nodes)
        adj = (adj + adj.t()) / 2
        
        mask = (1 - torch.eye(num_nodes)).bool()
        mask = mask.to(device)

        z = torch.masked_select(adj, mask)
    elif mode == 'ltmatonly':
        z = z.reshape(num_nodes, num_nodes-1)
        for i in range(num_nodes-1):
            z[i+1:num_nodes, i] = z[i, i:num_nodes-1]
        z = z.reshape(-1)
    else:
        assert False, 'Invalid Mode'
    return z

def circshift_z_nodemode(z, num_nodes, batch_num):
    z = z.reshape(batch_num, num_nodes)
    z_nodemode = z[:, 1:]
    
    expand_size = [batch_num, num_nodes-1, num_nodes]
    z_nodemode_expand = z_nodemode.unsqueeze(-1)
    z_edge = z_nodemode_expand.expand(expand_size).clone()

    for i in range(num_nodes):
        z_circshift = torch.roll(z_nodemode, shifts=i, dims=1)
        z_edge[:, :, i] = z_circshift
    z_edge = z_edge.permute(0, 2, 1)
    z_edge = z_edge.reshape(-1, 1)
    return z_edge

def circshift_z(z, num_nodes, batch_num, ave_z=False):
    if not(ave_z):
        z = z.reshape(batch_num, -1)
    else:
        z = z.reshape(batch_num, -1)
        z = z.mean(dim=0, keepdim=True)
        
    z = z[:, 0:num_nodes-1] # Use
    z_expand = z.unsqueeze(-1)
    expand_size = [batch_num, num_nodes-1, num_nodes]
    z_edge = z_expand.expand(expand_size).clone()

    for i in range(num_nodes):
        z_circshift = torch.roll(z, shifts=i, dims=1)
        z_edge[:, :, i] = z_circshift
    z_edge = z_edge.permute(0, 2, 1)
    if ave_z:
        z_edge = z_edge.mean(dim=0)
    z_edge = z_edge.reshape(-1, 1)
    return z_edge

def calc_l1_dev_scaler(x, w_hat, w):
    l1_dev = np.abs(x*w_hat - w - np.mean(x*w_hat - w))
    return np.mean(l1_dev)

def calc_l1_dev_bias(x, w_hat, w):
    l1_dev = np.square(w_hat + x - w)
    return np.mean(l1_dev)

def calc_l1_dev_scale_factor_l2_loss(w_hat, w_tar):
    num_nodes = w_hat.shape[0]
    w_hat_ori, w_tar_ori = np.empty((num_nodes, num_nodes)), np.empty((num_nodes, num_nodes))

    for i in range(num_nodes):
        w_hat_ori[i] = np.roll(w_hat[i], -i)
        w_tar_ori[i] = np.roll(w_tar[i], -i)

    w_hat_vec = np.mean(w_hat_ori, axis=0)
    w_tar_vec = np.mean(w_tar_ori, axis=0)
    w_hat_vec, w_tar_vec = w_hat_vec.reshape(-1), w_tar_vec.reshape(-1)
 
    k_argmin = None
    l1_dev_min = np.inf

    for i in range(num_nodes):        
        if (w_hat_vec[i] - np.mean(w_hat_vec)) != 0:
            k = (w_tar_vec[i] + np.mean(w_tar_vec)) / (w_hat_vec[i] - np.mean(w_hat_vec))
            l1_dev = np.linalg.norm(w_hat_vec*k - w_tar_vec, ord=1)
            if l1_dev < l1_dev_min:
                l1_dev_min = l1_dev
                k_argmin = k
    
    l2_loss = np.linalg.norm(w_hat_vec*k_argmin - w_tar_vec, ord=2)
    l2_tar = np.linalg.norm(w_tar_vec, ord=2)
    return k_argmin, l2_loss/l2_tar, w_hat_vec, w_tar_vec

def scale_w_hat_scipy(w_hat, w_tar):
    num_nodes = w_hat.shape[0]
    w_hat_ori, w_tar_ori = np.empty((num_nodes, num_nodes)), np.empty((num_nodes, num_nodes))

    for i in range(num_nodes):
        w_hat_ori[i] = np.roll(w_hat[i], -i)
        w_tar_ori[i] = np.roll(w_tar[i], -i)

    w_hat_vec = np.mean(w_hat_ori, axis=0)
    w_tar_vec = np.mean(w_tar_ori, axis=0)
    w_hat_vec, w_tar_vec = w_hat_vec.reshape(-1), w_tar_vec.reshape(-1)
    
    k_init = np.concatenate((np.random.rand(1), np.zeros(1)))
    k_scaler = minimize(calc_l1_dev_scaler, k_init[0], args=(w_hat_vec, w_tar_vec)).x
    k_bias = minimize(calc_l1_dev_bias, k_init[1], args=(w_hat_vec*k_scaler, w_tar_vec)).x
    k_argmin = np.concatenate((k_scaler, k_bias))
    
    w_hat_norm = w_hat_ori * k_scaler + k_bias
    w_hat_std = np.std(w_hat_norm, axis=0)

    l2_loss = np.linalg.norm(w_hat_vec*k_scaler + k_bias - w_tar_vec, ord=2)
    l2_tar = np.linalg.norm(w_tar_vec, ord=2)
    return k_argmin, l2_loss/l2_tar, w_hat_vec, w_tar_vec, w_hat_std

def load_neural_spike(data_dir, neuron, timesteps, data_ratio=1):
    if neuron == 'ring':
        if timesteps == 50:
            tr = gen_spike.spike_ring_binned_train_50(data_dir)
            val = gen_spike.spike_ring_binned_valid_50(data_dir)
            test = gen_spike.spike_ring_binned_test_50(data_dir)
        elif timesteps == 100:
            tr = gen_spike.spike_ring_binned_train_100(data_dir)
            val = gen_spike.spike_ring_binned_valid_100(data_dir)
            test = gen_spike.spike_ring_binned_test_100(data_dir)
        elif (timesteps == 200) and (data_ratio == 1):
            tr = gen_spike.spike_ring_raw_train(data_dir)
            val = gen_spike.spike_ring_raw_valid(data_dir)
            test = gen_spike.spike_ring_raw_test(data_dir)        
    elif neuron == 'LNP':
        if timesteps == 50:
            tr = gen_spike.spike_LNP_binned_train_50(data_dir)
            val = gen_spike.spike_LNP_binned_valid_50(data_dir)
            test = gen_spike.spike_LNP_binned_test_50(data_dir)
        elif timesteps == 100:
            tr = gen_spike.spike_LNP_binned_train_100(data_dir)
            val = gen_spike.spike_LNP_binned_valid_100(data_dir)
            test = gen_spike.spike_LNP_binned_test_100(data_dir)
        elif (timesteps == 200) and (data_ratio == 1):
            tr = gen_spike.spike_LNP_raw_train_1(data_dir)
            val = gen_spike.spike_LNP_raw_valid_1(data_dir)
            test = gen_spike.spike_LNP_raw_valid_1(data_dir)
        elif (timesteps == 200) and (data_ratio == 100):
            tr = gen_spike.spike_LNP_raw_train_whole(data_dir)
            val = gen_spike.spike_LNP_raw_valid_whole(data_dir)
            test = gen_spike.spike_LNP_raw_test_whole(data_dir)
    return tr, val, test

def load_neural_spike_bin(data_dir, pred_step, percentage):
    if pred_step == 20:
        if percentage == 100:
            tr = gen_spike.spike_LNP_bin_train_whole(data_dir)
            val = gen_spike.spike_LNP_bin_valid_whole(data_dir)
            test = gen_spike.spike_LNP_bin_test_whole(data_dir)
        elif percentage == 1:
            tr = gen_spike.spike_LNP_bin_train_1(data_dir)
            val = gen_spike.spike_LNP_bin_valid_1(data_dir)
            test = gen_spike.spike_LNP_bin_test_1(data_dir)
    elif pred_step == 40:
        tr = gen_spike.spike_LNP_bin_train_whole_40(data_dir)
        val = gen_spike.spike_LNP_bin_valid_whole_40(data_dir)
        test = gen_spike.spike_LNP_bin_test_whole_40(data_dir)
    return tr, val, test

def load_neural_spike_bin_phase2(data_dir, pred_step, percentage, mode):
    if pred_step == 20:
        if percentage == 100:
            if mode == 'zero':
                tr = gen_spike.LNP_bin_zeros_train(data_dir)
                val = gen_spike.LNP_bin_zeros_valid(data_dir)
                test = gen_spike.LNP_bin_zeros_test(data_dir)
            elif mode == 'rand':
                tr = gen_spike.LNP_bin_rand_train(data_dir)
                val = gen_spike.LNP_bin_rand_valid(data_dir)
                test = gen_spike.LNP_bin_rand_test(data_dir)            
            elif mode == 'init':
                tr = gen_spike.LNP_init_bin_train(data_dir)
                val = gen_spike.LNP_init_bin_valid(data_dir)
                test = gen_spike.LNP_init_bin_test(data_dir)
    return tr, val, test

def spec_decompose(mat, count):
    mat_spec = np.zeros_like(mat)
    
    mat_normed = (np.abs(mat) - np.abs(mat).min()) / (np.abs(mat).max() - np.abs(mat).min())
    lam, vec_all = np.linalg.eigh(mat_normed)
    arg_descending = np.abs(lam).argsort()[::-1]

    lam_sorted = lam[arg_descending]
    vec_all_sorted = vec_all[:, arg_descending]

    for i in range(count):
        idx = arg_descending[i]
        vec = vec_all[:, idx:idx+1]
        mat_spec = mat_spec + lam[idx]*np.matmul(vec, vec.T)
    
    return lam_sorted, vec_all_sorted, mat_spec

def get_edgeidx_by_batchsize(edge, batch_size, num_nodes, device):
    size = [batch_size] + list(edge.shape)
    edge_expand = torch.empty(size)
    edge_expand = edge_expand.to(device)
    
    for i in range(batch_size):
        edge_expand[i, :, :] = edge + num_nodes*i
    
    edge_expand = edge_expand.permute(1, 0, 2)
    edge_expand = edge_expand.reshape(edge.shape[0], -1)
    return edge_expand.long()

def z_to_adj(z, nodes, device):
    adj = torch.zeros(nodes * nodes)
    
    for i in range(nodes - 1):
        adj[i*(nodes+1) + 1:(i+1)*(nodes+1)] = z[i*nodes: (i+1)*nodes]
    
    adj = adj.reshape(nodes, nodes)
    adj = adj.to(device)
    return adj.long()

def adj_to_edge_idx_pyg(adj):
    adj = torch.where(adj)
    adj = torch.stack((adj[0], adj[1]), dim=0)
    return adj

def z_to_adj_plot(z, nodes):
    adj = np.zeros(nodes * nodes)
    
    for i in range(nodes - 1):
        adj[i*(nodes+1) + 1:(i+1)*(nodes+1)] = z[i*nodes: (i+1)*nodes]
    
    adj = adj.reshape(nodes, nodes)
    return adj

def to_dec_batch(spike_whole, nodes, history, pred_step, p2_bs):
    total_steps = spike_whole.shape[1]
    window_size = history + pred_step - 1
    iter_step = int((total_steps - history) / pred_step)

    spike_window = []
    spike_target = []

    for i in range(iter_step):
        step = i * pred_step
        spike_window.append(spike_whole[:, step:step+window_size])
        spike_target.append(spike_whole[:, step+history:step+history+pred_step])
    
    spike_window_whole = torch.stack(spike_window, dim=1) #[100, iter_step, 219]
    spike_target_whole = torch.stack(spike_target, dim=1) #[100, iter_step, 20]

    whole_cut = int(spike_window_whole.shape[1] / p2_bs) * p2_bs
    
    spike_window_whole = spike_window_whole[:, :whole_cut, :]
    spike_target_whole = spike_target_whole[:, :whole_cut, :]     
    spike_window_whole = spike_window_whole.reshape(nodes, -1, p2_bs, window_size) #[100, iters, bs, window]
    spike_target_whole = spike_target_whole.reshape(nodes, -1, p2_bs, pred_step)

    return spike_window_whole, spike_target_whole

def to_dec_batch_ptar(spike_whole, nodes, history, pred_step, p2_bs):
    total_steps = spike_whole.shape[1]
    window_size = history + pred_step - 1
    iter_step = int(total_steps / history)

    spike_window = []
    spike_target = []

    for i in range(iter_step):
        step = i * history
        spike_window.append(spike_whole[:, step:step+window_size])
        spike_target.append(spike_whole[:, step+history:step+history+pred_step])
    
    spike_window_whole = torch.stack(spike_window, dim=1) #[100, iter_step, 219]
    spike_target_whole = torch.stack(spike_target, dim=1) #[100, iter_step, 20]

    whole_cut = int(spike_window_whole.shape[1] / p2_bs) * p2_bs
    
    spike_window_whole = spike_window_whole[:, :whole_cut, :]
    spike_target_whole = spike_target_whole[:, :whole_cut, :]     
    spike_window_whole = spike_window_whole.reshape(nodes, -1, p2_bs, window_size) #[100, iters, bs, window]
    spike_target_whole = spike_target_whole.reshape(nodes, -1, p2_bs, pred_step)

    return spike_window_whole, spike_target_whole

def to_dec_batch_ptar_plot(spike_whole, nodes, history, pred_step, p2_bs):
    total_steps = spike_whole.shape[1]
    window_size = history + pred_step - 1
    iter_step = int((total_steps - history) / pred_step)

    spike_window = []
    spike_target = []

    for i in range(iter_step):
        step = i * pred_step
        spike_window.append(spike_whole[:, step:step+window_size])
        spike_target.append(spike_whole[:, step+history:step+history+pred_step])
    
    spike_window_whole = torch.stack(spike_window, dim=1) #[100, iter_step, 219]
    spike_target_whole = torch.stack(spike_target, dim=1) #[100, iter_step, 20]

    whole_cut = int(spike_window_whole.shape[1] / p2_bs) * p2_bs
    
    spike_window_whole = spike_window_whole[:, :whole_cut, :]
    spike_target_whole = spike_target_whole[:, :whole_cut, :]     
    spike_window_whole = spike_window_whole.reshape(nodes, -1, p2_bs, window_size) #[100, iters, bs, window]
    spike_target_whole = spike_target_whole.reshape(nodes, -1, p2_bs, pred_step)

    return spike_window_whole, spike_target_whole

