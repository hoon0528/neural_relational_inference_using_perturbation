import time
import os
import argparse

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import yaml
from generate_spike import LNP_perturb_bin, LNP_perturb_bin_tar
from model.nri_vsc_wo_pinputs import RNNDecoder_NRI_m2o_perturb
from model.gts_adj import gts_adj_inf
from utils import *

torch.manual_seed(42)

parser = argparse.ArgumentParser()
# Dataset Configuration
parser.add_argument('--config', type=str, default='_ss_wo_pin')
parser.add_argument('--dataset', type=str, default='neural_spike')
parser.add_argument('--neurons', type=str, default='LNP')
parser.add_argument('--bin', type=int, default=1)
parser.add_argument('--nodes', type=int, default=100)
# Experiments Configuration
parser.add_argument('--experiment', type=str, default='perturbation')
parser.add_argument('--loss_scaler', type=float, default=30)
parser.add_argument('--history', type=int, default=200)
parser.add_argument('--data_length_ratio', type=int, default=4)
parser.add_argument('--pred_step_p1', type=int, default=4)
parser.add_argument('--pred_step_p2', type=int, default=20)
parser.add_argument('--perturb_step', type=int, default=200)
parser.add_argument('--znorm', type=str, default='sigmoid')
parser.add_argument('--symmode', type=str, default='ltmatonly')
parser.add_argument('--decoder', type=str, default='RNN_many_to_one')
parser.add_argument('--adj_infmodel', type=str, default='gts')
parser.add_argument('--adj_infstyle', type=str, default='complete')
parser.add_argument('--add_similarity_z_term', type=int, default=1)
parser.add_argument('--similarity_z_term', type=str, default='cos')
parser.add_argument('--gts_complete', type=int, default=1)
parser.add_argument('--gts_totalstep', type=int, default=40000)
# Dimension of Layers
parser.add_argument('--enc_hid_dim', type=int, default=64)
parser.add_argument('--dec_hid_dim', type=int, default=64)
parser.add_argument('--dec_msg_dim', type=int, default=64)
parser.add_argument('--out_channel', type=int, default=8)
parser.add_argument('--kernal_x_1', type=int, default=200)
parser.add_argument('--kernal_x_2', type=int, default=200)
parser.add_argument('--stride_x_1', type=int, default=20)
parser.add_argument('--stride_x_2', type=int, default=20)
# Learning features
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--phase1_batchsize', type=int, default=80)
parser.add_argument('--phase2_batchsize', type=int, default=5)
parser.add_argument('--device', type=str, default=0)

# create save_folder ant e.t.c
args = parser.parse_args()
config_suffix = args.config + '_' + args.dataset + '200'
with open('config/config{}.yml'.format(config_suffix), encoding='UTF8') as f:
    setting = yaml.load(f, Loader=yaml.FullLoader)
settings = setting.get('train')

sim_setting = settings.get('sim_setting')
model_params = settings.get('model_params')

num_nodes = args.nodes
edge_types = sim_setting.get('edge_types')
num_node_features = sim_setting.get('num_node_features') # Fixed(1) in Neural Spike data
history = args.history
pred_step_p1 = args.pred_step_p1
pred_step_p2 = args.pred_step_p2

data_suffix = sim_setting.get('data') + '/' + args.neurons + str(args.history)
model_params = settings.get('model_params')

phase2_length = args.perturb_step * args.phase1_batchsize
if args.add_similarity_z_term:
    exp_config_suffix = 'add_zloss_' + args.similarity_z_term + '_' +  str(phase2_length)
else:
    exp_config_suffix = 'no_zloss_' + args.similarity_z_term + '_' + str(phase2_length)

save_folder = model_params.get('model').get('save_folder') + data_suffix + '/' + 'perturbation'
plot_folder = 'plot/' + data_suffix

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print(exp_config_suffix)

lr = model_params.get('training').get('lr')
lr_decay = model_params.get('training').get('lr_decay')
gamma = model_params.get('training').get('gamma')
p1_bs = args.phase1_batchsize
p2_bs = args.phase2_batchsize
epochs = args.epochs
perturb_step = args.perturb_step
data_length_ratio = args.data_length_ratio

enc_hid = args.enc_hid_dim
dec_msg_hid = args.dec_hid_dim
dec_msg_out = args.dec_msg_dim
dec_hid = args.dec_hid_dim
dec_drop = model_params.get('decoder').get('do_prob')

device = sim_setting.get('device') + str(args.device)

if args.bin:
    bin_str = 'binned'
else:
    bin_str = 'raw'

loss_scaler_str = str(args.loss_scaler).replace('.', '')

if args.experiment == 'perturbation':
    encoder_file = os.path.join(save_folder, 'enc_h{}_{}_{}.pt'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    decoder_file = os.path.join(save_folder, 'dec_h{}_{}_{}.pt'.format(dec_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    loss_file = os.path.join(save_folder, 'loss_h{}_{}_{}.npy'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    log_file = os.path.join(save_folder, 'log_h{}_{}_{}.txt'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))

elif args.experiment == 'equilibrium':
    encoder_file = os.path.join(save_folder, 'enc_h{}_{}_eq.pt'.format(enc_hid, exp_config_suffix))
    decoder_file = os.path.join(save_folder, 'dec_h{}_{}_eq.pt'.format(dec_hid, exp_config_suffix))
    loss_file = os.path.join(save_folder, 'loss_h{}_{}_eq.npy'.format(enc_hid, exp_config_suffix))
    log_file = os.path.join(save_folder, 'log_h{}_{}_eq.txt'.format(enc_hid, exp_config_suffix))

elif args.experiment == 'perturb_target':
    if args.add_similarity_z_term:
        exp_config_suffix = 'add_zloss_' + args.similarity_z_term
    else:
        exp_config_suffix = 'no_zloss_' + args.similarity_z_term
    encoder1_file = os.path.join(save_folder, 'enc_h{}_{}_{}_ptar_1.pt'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    encoder2_file = os.path.join(save_folder, 'enc_h{}_{}_{}_ptar_2.pt'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    decoder1_file = os.path.join(save_folder, 'dec_h{}_{}_{}_ptar_1.pt'.format(dec_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    decoder2_file = os.path.join(save_folder, 'dec_h{}_{}_{}_ptar_2.pt'.format(dec_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    loss_file = os.path.join(save_folder, 'loss_h{}_{}_{}_ptar.npy'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    log_file = os.path.join(save_folder, 'log_h{}_{}_{}_ptar.txt'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))

elif args.experiment == 'perturb_target_only':
    if args.add_similarity_z_term:
        exp_config_suffix = 'add_zloss_' + args.similarity_z_term
    else:
        exp_config_suffix = 'no_zloss_' + args.similarity_z_term
    encoder_file = os.path.join(save_folder, 'enc_h{}_{}_{}_pto.pt'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    decoder_file = os.path.join(save_folder, 'dec_h{}_{}_{}_pto.pt'.format(dec_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    loss_file = os.path.join(save_folder, 'loss_h{}_{}_{}_pto.npy'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))
    log_file = os.path.join(save_folder, 'log_h{}_{}_{}_pto.txt'.format(enc_hid, exp_config_suffix,
                                                                       loss_scaler_str))

else:
    assert False, 'Invalid Experiment'

if os.path.isfile(log_file):
    log = open(log_file, 'a')
else:
    log = open(log_file, 'w')
device = torch.device(device if torch.cuda.is_available() else 'cpu')

# Define models
adj_inf_model = gts_adj_inf(num_nodes, enc_hid, args.out_channel, args.kernal_x_1, args.kernal_x_2, 
                            args.stride_x_1, args.stride_x_2, args.gts_totalstep)
decoder = RNNDecoder_NRI_m2o_perturb(num_node_features, dec_hid, sim_setting, 
                                     p1_bs, history, num_nodes, device)                            
gts_featmat = np.load('data/gts_featuremat.npy')[:, :args.gts_totalstep]

# Optimizer setting
optimizer = optim.Adam(list(adj_inf_model.parameters()) + list(decoder.parameters()), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

# Load dataset
if args.experiment == 'perturb_target_only':
    trainset = LNP_perturb_bin_tar(history, pred_step_p1, p1_bs, 'train', data_length_ratio)
    validset = LNP_perturb_bin_tar(history, pred_step_p1, p1_bs, 'valid', data_length_ratio)
    train_loader = DL_Py(trainset, batch_size=1, shuffle=True) 
    valid_loader = DL_Py(validset, batch_size=1, shuffle=True)
else:
    trainset = LNP_perturb_bin(history, pred_step_p1, p1_bs, 'train', data_length_ratio)
    validset = LNP_perturb_bin(history, pred_step_p1, p1_bs, 'valid', data_length_ratio)
    #testset = LNP_perturb_bin(history, pred_step, p1_bs, 'test')
    train_loader = DL_Py(trainset, batch_size=1, shuffle=True) 
    valid_loader = DL_Py(validset, batch_size=1, shuffle=True)
    #test_loader = DL_Py(testset, batch_size=1, shuffle=True)

def train_perturb(epoch, best_val_loss):
    t = time.time()
    
    cos_tar = torch.ones(1)
    cos_tar = cos_tar.to(device)
    
    gts_input = torch.FloatTensor(gts_featmat)
    gts_input = gts_input.to(device)
    
    poisson_p1_train = []
    poisson_p2_train = []
    z_loss_train = []

    adj_inf_model.train()
    decoder.train()

    adj_inf_model.to(device)
    decoder.to(device)
    decoder.to(device)

    for batch_idx, data in tqdm(enumerate(train_loader)):
        x_spk, tar_spk, edge_idx = data
        x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
        x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)

        optimizer.zero_grad()

        _, z1_corr = adj_inf_model(gts_input, edge_idx)
        z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
        z1 = -torch.sigmoid(z1_corr)
        z1 = z1.reshape(-1, 1)

        out_p1 = decoder(x_spk, edge_idx, z1, pred_step_p1)
        loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

        spike_p2 = []
        with torch.no_grad():
            spk_p2_init = x_spk[:, :, :history]
            decoder.load_state_dict(decoder.state_dict())

            for i in range(perturb_step):
                if i == 0:
                    lam_p2 = decoder(spk_p2_init, edge_idx, z1, 1) #[100, 80, 200] > [100, 80, 1, 1]
                    spk_p2 = torch.poisson(lam_p2.exp()) #cat [100, 80, 1, 1]
                    spk_cat = spk_p2.reshape(num_nodes, p1_bs, 1)
                    spk_next = torch.cat((spk_p2_init[:, :, -199:], spk_cat), dim=-1)
                else:
                    lam_p2 = decoder(spk_next, edge_idx, z1, 1) #[100, 80, 200] > [100, 80, 1, 1]
                    spk_p2 = torch.poisson(lam_p2.exp()) #cat [100, 80, 1, 1]
                    spk_cat = spk_p2.reshape(num_nodes, p1_bs, 1)
                    spk_next = torch.cat((spk_next[:, :, -199:], spk_cat), dim=-1)
            spike_p2.append(spk_next)

            spike_p2 = torch.stack(spike_p2, dim=2) #[nodes, p1_bs, perturb_step, feat_dim]
            spike_p2 = spike_p2.reshape(num_nodes, -1) #[nodes, p1_bs*perturb_step]
            x_spk_p2, tar_spk_p2 = to_dec_batch(spike_p2, num_nodes, history, pred_step_p2, p2_bs)
            
            p2_step = spike_p2.shape[1]
            count_p2 = int(args.gts_totalstep / p2_step)
            spike_p2_featmat = torch.unsqueeze(spike_p2, dim=1)
            spike_p2_featmat = spike_p2_featmat.expand(num_nodes, count_p2+1, p2_step)
            spike_p2_featmat = spike_p2_featmat.reshape(num_nodes, -1)
            spike_p2_featmat = spike_p2_featmat[:, :args.gts_totalstep]

        _, z2_corr = adj_inf_model(spike_p2_featmat, edge_idx)
        z2_corr = make_z_sym_gts(z2_corr, num_nodes, device, args.symmode)
        z2 = -torch.sigmoid(z2_corr)
        z2 = z2.reshape(-1, 1)

        if args.similarity_z_term == 'cos':
            loss_z_similar = F.cosine_embedding_loss(z1.t(), z2.t(), cos_tar)
        elif args.similarity_z_term == 'none':
            loss_z_similar = 0
        else:
            assert False, 'Invalid z loss function'  

        out_p2_list = []
        for i in range(x_spk_p2.shape[1]):
            out_p2 = decoder(x_spk_p2[:, i, :, :], edge_idx, z2, pred_step_p2) #[100, p2_bs, 20, 1]
            out_p2_list.append(out_p2.squeeze())
        out_p2_aggr = torch.stack(out_p2_list, dim=1)
        loss_recon_p2 = F.poisson_nll_loss(out_p2_aggr, tar_spk_p2, log_input=True)

        if args.add_similarity_z_term:
            loss = loss_recon_p1 + loss_recon_p2 + args.loss_scaler*loss_z_similar
        else:
            loss = loss_recon_p1 + loss_recon_p2
        loss.backward()
        optimizer.step()

        poisson_p1_train.append(loss_recon_p1.item())
        poisson_p2_train.append(loss_recon_p2.item())
        z_loss_train.append(loss_z_similar.item())
        
    scheduler.step()

    poisson_p1_valid = []
    poisson_p2_valid = []
    z_loss_valid = []

    adj_inf_model.eval()
    decoder.eval()
    
    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            x_spk, tar_spk, edge_idx = data
            x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
            x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)
            
            optimizer.zero_grad()

            _, z1_corr = adj_inf_model(gts_input, edge_idx)
            z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
            z1 = -torch.sigmoid(z1_corr)
            z1 = z1.reshape(-1, 1)

            out_p1 = decoder(x_spk, edge_idx, z1, pred_step_p1)
            loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

            spike_p2 = []
            with torch.no_grad():
                spk_p2_init = x_spk[:, :, :history]
                decoder.load_state_dict(decoder.state_dict())
            
                for i in range(perturb_step):
                    if i == 0:
                        lam_p2 = decoder(spk_p2_init, edge_idx, z1, 1) #[100, 80, 200] > [100, 80, 1, 1]
                        spk_p2 = torch.poisson(lam_p2.exp()) #cat [100, 80, 1, 1]
                        spk_cat = spk_p2.reshape(num_nodes, p1_bs, 1)
                        spk_next = torch.cat((spk_p2_init[:, :, -199:], spk_cat), dim=-1)
                    else:
                        lam_p2 = decoder(spk_next, edge_idx, z1, 1) #[100, 80, 200] > [100, 80, 1, 1]
                        spk_p2 = torch.poisson(lam_p2.exp()) #cat [100, 80, 1, 1]
                        spk_cat = spk_p2.reshape(num_nodes, p1_bs, 1)
                        spk_next = torch.cat((spk_next[:, :, -199:], spk_cat), dim=-1)
                spike_p2.append(spk_next)

                spike_p2 = torch.stack(spike_p2, dim=2) #[nodes, p1_bs, perturb_step, feat_dim]
                spike_p2 = spike_p2.reshape(num_nodes, -1) #[nodes, p1_bs*perturb_step]
                x_spk_p2, tar_spk_p2 = to_dec_batch(spike_p2, num_nodes, history, pred_step_p2, p2_bs)

                p2_step = spike_p2.shape[1]
                count_p2 = int(args.gts_totalstep / p2_step)
                spike_p2_featmat = torch.unsqueeze(spike_p2, dim=1)
                spike_p2_featmat = spike_p2_featmat.expand(num_nodes, count_p2+1, p2_step)
                spike_p2_featmat = spike_p2_featmat.reshape(num_nodes, -1)
                spike_p2_featmat = spike_p2_featmat[:, :args.gts_totalstep]

            _, z2_corr = adj_inf_model(spike_p2_featmat, edge_idx)
                
            z2_corr = make_z_sym_gts(z2_corr, num_nodes, device, args.symmode)
            z2 = -torch.sigmoid(z2_corr)
            z2 = z2.reshape(-1, 1)

            if args.similarity_z_term == 'cos':
                loss_z_similar = F.cosine_embedding_loss(z1.t(), z2.t(), cos_tar)
            elif args.similarity_z_term == 'none':
                loss_z_similar = 0
            else:
                assert False, 'Invalid z loss function'  

            out_p2_list = []
            for i in range(x_spk_p2.shape[1]):
                out_p2 = decoder(x_spk_p2[:, i, :, :], edge_idx, z2, pred_step_p2) #[100, p2_bs, 20, 1]
                out_p2_list.append(out_p2.squeeze())
            out_p2_aggr = torch.stack(out_p2_list, dim=1)
            loss_recon_p2 = F.poisson_nll_loss(out_p2_aggr, tar_spk_p2, log_input=True)

            if args.add_similarity_z_term:
                loss = loss_recon_p1 + loss_recon_p2 + args.loss_scaler*loss_z_similar
            else:
                loss = loss_recon_p1 + loss_recon_p2

            poisson_p1_valid.append(loss_recon_p1.item())
            poisson_p2_valid.append(loss_recon_p2.item())
            z_loss_valid.append(loss_z_similar.item())
        
        print('Epoch: {:04d}'.format(epoch),
              'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
              'pois_p2_train: {:.10f}'.format(np.mean(poisson_p2_train)),
              'z_cos_train: {:.10f}'.format(np.mean(z_loss_train)),
              'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
              'pois_p2_val: {:.10f}'.format(np.mean(poisson_p2_valid)),
              'z_cos_val: {:.10f}'.format(np.mean(z_loss_valid)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(poisson_p1_valid) < best_val_loss:
            torch.save(adj_inf_model.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder1_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
                  'pois_p2_train: {:.10f}'.format(np.mean(poisson_p2_train)),
                  'z_cos_train: {:.10f}'.format(np.mean(z_loss_train)),
                  'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
                  'pois_p2_val: {:.10f}'.format(np.mean(poisson_p2_valid)),
                  'z_cos_val: {:.10f}'.format(np.mean(z_loss_valid)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
        
        loss = np.array([np.mean(poisson_p1_train), np.mean(poisson_p2_train), np.mean(z_loss_train), 
                         np.mean(poisson_p1_valid), np.mean(poisson_p2_valid), np.mean(z_loss_valid)])
    return np.mean(poisson_p1_valid), loss

def train_eq_only(epoch, best_val_loss):
    t = time.time()
      
    gts_input = torch.FloatTensor(gts_featmat)
    gts_input = gts_input.to(device)
    
    poisson_p1_train = []
    z_loss_train = []

    adj_inf_model.train()
    decoder.train()

    adj_inf_model.to(device)
    decoder.to(device)

    for batch_idx, data in tqdm(enumerate(train_loader)):
        x_spk, tar_spk, edge_idx = data
        x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
        x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)

        optimizer.zero_grad()

        _, z1_corr = adj_inf_model(gts_input, edge_idx)
        z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
        z1 = -torch.sigmoid(z1_corr)
        z1 = z1.reshape(-1, 1)

        out_p1 = decoder(x_spk, edge_idx, z1, pred_step_p1)
        loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)
        loss_recon_p1.backward()
        optimizer.step()

        poisson_p1_train.append(loss_recon_p1.item())
        
    scheduler.step()

    poisson_p1_valid = []
    z_loss_valid = []

    adj_inf_model.eval()
    decoder.eval()
    
    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            x_spk, tar_spk, edge_idx = data
            x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
            x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)
            
            optimizer.zero_grad()

            _, z1_corr = adj_inf_model(gts_input, edge_idx)
            z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
            z1 = -torch.sigmoid(z1_corr)
            z1 = z1.reshape(-1, 1)

            out_p1 = decoder(x_spk, edge_idx, z1, pred_step_p1)
            loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

            poisson_p1_valid.append(loss_recon_p1.item())
        
        print('Epoch: {:04d}'.format(epoch),
              'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
              'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(poisson_p1_valid) < best_val_loss:
            torch.save(adj_inf_model.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
                  'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
        
        loss = np.array([np.mean(poisson_p1_train), np.mean(poisson_p1_valid)])
    return np.mean(poisson_p1_valid), loss

def train_perturb_target(epoch, best_val_p1, best_val_p2):
    t = time.time()
    
    cos_tar = torch.ones(1)
    cos_tar = cos_tar.to(device)
    
    gts_input = torch.FloatTensor(gts_featmat)
    gts_input = gts_input.to(device)
    
    per_input_all = np.load('data/spk_bin_per.npy')
    per_input_all = torch.FloatTensor(per_input_all)
    per_input_train = per_input_all[:, :480000]
    per_input_valid = per_input_all[:, 480000:528000]
    
    x2_input_tr, x2_target_tr = to_dec_batch_ptar(per_input_train, num_nodes, history, pred_step_p1, p2_bs)
    x2_input_val, x2_target_val = to_dec_batch_ptar(per_input_valid, num_nodes, history, pred_step_p2, p2_bs)
    rand_idx_tr = torch.randperm(x2_input_tr.shape[1])
    rand_idx_val = torch.randperm(x2_input_val.shape[1])
    x2_input_tr, x2_target_tr = x2_input_tr[:, rand_idx_tr, :, :], x2_target_tr[:, rand_idx_tr, :, :]
    x2_input_val, x2_target_val = x2_input_val[:, rand_idx_val, :, :], x2_target_val[:, rand_idx_val, :, :]

    per_input = per_input_all[:, :args.gts_totalstep]
    per_input, x2_input_tr, x2_target_tr = per_input.to(device), x2_input_tr.to(device), x2_target_tr.to(device)
    x2_input_val, x2_target_val = x2_input_val.to(device), x2_target_val.to(device)
    
    poisson_p1_train = []
    poisson_p2_train = []
    z_loss_train = []

    adj_inf_model.train()
    decoder.train()

    adj_inf_model.to(device)
    decoder.to(device)

    for batch_idx, data in tqdm(enumerate(train_loader)):
        x_spk, tar_spk, edge_idx = data
        x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
        x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)

        optimizer.zero_grad()

        _, z1_corr = adj_inf_model(gts_input, edge_idx)
        z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
        z1 = -torch.sigmoid(z1_corr)
        z1 = z1.reshape(-1, 1)

        out_p1 = decoder(x_spk, edge_idx, z1, pred_step_p1)
        loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

        _, z2_corr = adj_inf_model(per_input, edge_idx)
        z2_corr = make_z_sym_gts(z2_corr, num_nodes, device, args.symmode)
        z2 = -torch.sigmoid(z2_corr)
        z2 = z2.reshape(-1, 1)

        if args.similarity_z_term == 'cos':
            loss_z_similar = F.cosine_embedding_loss(z1.t(), z2.t(), cos_tar)
        elif args.similarity_z_term == 'none':
            loss_z_similar = 0
        else:
            assert False, 'Invalid z loss function'  

        out_p2 = decoder(x2_input_tr[:, batch_idx, :, :], edge_idx, z2, pred_step_p2) #[100, p2_bs, 20, 1]
        loss_recon_p2 = F.poisson_nll_loss(out_p2.squeeze(), x2_target_tr[:, batch_idx, :, :], log_input=True)

        if args.add_similarity_z_term:
            loss = loss_recon_p1 + loss_recon_p2 + args.loss_scaler*loss_z_similar
        else:
            loss = loss_recon_p1 + loss_recon_p2
        loss.backward()
        optimizer.step()

        poisson_p1_train.append(loss_recon_p1.item())
        poisson_p2_train.append(loss_recon_p2.item())
        z_loss_train.append(loss_z_similar.item())
        
    scheduler.step()

    poisson_p1_valid = []
    poisson_p2_valid = []
    z_loss_valid = []

    adj_inf_model.eval()
    decoder.eval()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            x_spk, tar_spk, edge_idx = data
            x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
            x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)
            
            optimizer.zero_grad()

            _, z1_corr = adj_inf_model(gts_input, edge_idx)
            z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
            z1 = -torch.sigmoid(z1_corr)
            z1 = z1.reshape(-1, 1)

            out_p1 = decoder(x_spk, edge_idx, z1, pred_step_p1)
            loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

            _, z2_corr = adj_inf_model(per_input, edge_idx)
                
            z2_corr = make_z_sym_gts(z2_corr, num_nodes, device, args.symmode)
            z2 = -torch.sigmoid(z2_corr)
            z2 = z2.reshape(-1, 1)

            if args.similarity_z_term == 'cos':
                loss_z_similar = F.cosine_embedding_loss(z1.t(), z2.t(), cos_tar)
            elif args.similarity_z_term == 'none':
                loss_z_similar = 0
            else:
                assert False, 'Invalid z loss function'  

            out_p2 = decoder(x2_input_val[:, batch_idx, :, :], edge_idx, z2, pred_step_p2) #[100, p2_bs, 20, 1]
            loss_recon_p2 = F.poisson_nll_loss(out_p2.squeeze(), x2_target_val[:, batch_idx, :, :], log_input=True)

            if args.add_similarity_z_term:
                loss = loss_recon_p1 + loss_recon_p2 + args.loss_scaler*loss_z_similar
            else:
                loss = loss_recon_p1 + loss_recon_p2

            poisson_p1_valid.append(loss_recon_p1.item())
            poisson_p2_valid.append(loss_recon_p2.item())
            z_loss_valid.append(loss_z_similar.item())
        
        print('Epoch: {:04d}'.format(epoch),
              'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
              'pois_p2_train: {:.10f}'.format(np.mean(poisson_p2_train)),
              'z_cos_train: {:.10f}'.format(np.mean(z_loss_train)),
              'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
              'pois_p2_val: {:.10f}'.format(np.mean(poisson_p2_valid)),
              'z_cos_val: {:.10f}'.format(np.mean(z_loss_valid)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(poisson_p1_valid) < best_val_p1:
            torch.save(adj_inf_model.state_dict(), encoder1_file)
            torch.save(decoder.state_dict(), decoder1_file)
            print('Best equilibrium model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
                  'pois_p2_train: {:.10f}'.format(np.mean(poisson_p2_train)),
                  'z_cos_train: {:.10f}'.format(np.mean(z_loss_train)),
                  'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
                  'pois_p2_val: {:.10f}'.format(np.mean(poisson_p2_valid)),
                  'z_cos_val: {:.10f}'.format(np.mean(z_loss_valid)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()

        if np.mean(poisson_p2_valid) < best_val_p2:
            torch.save(adj_inf_model.state_dict(), encoder2_file)
            torch.save(decoder.state_dict(), decoder2_file)
            print('Best perturbation model so far, saving...')

        loss = np.array([np.mean(poisson_p1_train), np.mean(poisson_p2_train), np.mean(z_loss_train), 
                         np.mean(poisson_p1_valid), np.mean(poisson_p2_valid), np.mean(z_loss_valid)])
    return np.mean(poisson_p1_valid), np.mean(poisson_p2_valid), loss

loss_list = []
best_val_loss = np.inf
best_val_p1, best_val_p2 = np.inf, np.inf
best_epoch = 0

if args.experiment == 'perturbation':
    for epoch in tqdm(range(epochs)):
        poiss_p1_val, loss = train_perturb(epoch, best_val_loss)
        loss_list.append(loss)
        if poiss_p1_val < best_val_loss:
            best_val_loss = poiss_p1_val
            best_epoch = epoch

elif (args.experiment == 'equilibrium') or (args.experiment == 'perturb_target_only'):
    for epoch in tqdm(range(epochs)):
        poiss_p1_val, loss = train_eq_only(epoch, best_val_loss)
        loss_list.append(loss)
        if poiss_p1_val < best_val_loss:
            best_val_loss = poiss_p1_val
            best_epoch = epoch

elif args.experiment == 'perturb_target':
    for epoch in tqdm(range(epochs)):
        poiss_p1_val, poiss_p2_val, loss = train_perturb_target(epoch, best_val_p1, best_val_p2)
        loss_list.append(loss)
        if poiss_p1_val < best_val_p1:
            best_val_p1 = poiss_p1_val
        if poiss_p2_val < best_val_p2:
            best_val_p2 = poiss_p2_val
            
else:
    assert False, 'Invalid Experiment'

np.save(loss_file, np.stack(loss_list, axis=0))