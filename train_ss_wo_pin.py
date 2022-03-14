import time
import os
import argparse

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import yaml
from generate_data import *
from model.nri_vsc_wo_pinputs import RNNDecoder_NRI_BurnIn, VSCEncoder_NRI
from model.nri_vsc_wo_pinputs import MLPDecoder_NRI
from model.nri_vsc_wo_pinputs import MLPDecoder_NRI_Poisson
from model.nri_vsc_wo_pinputs import RNNDecoder_NRI_many_to_one
from model.gts_adj import gts_adj_inf
from utils import *
from torch_geometric import utils as pyg_utils

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='_ss_wo_pin')
parser.add_argument('--nodes', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.8)
# Using ss model, #edge_type is always 1 (continuous), but data generated isn't
parser.add_argument('--edges', type=int, default=1)
parser.add_argument('--device', type=str, default=0)
parser.add_argument('--dataset', type=str, default='neural_spike')
parser.add_argument('--neurons', type=str, default='LNP')
parser.add_argument('--rectime', type=int, default=200)
parser.add_argument('--pred_step', type=int, default=20)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--decoder', type=str, default='RNN_many_to_one')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--percentage', type=int, default=100)
parser.add_argument('--adj_infmodel', type=str, default='gts')
parser.add_argument('--adj_infstyle', type=str, default='circshift')
parser.add_argument('--gts_complete', type=int, default=1)
parser.add_argument('--num_spec_vector', type=int, default=10)
parser.add_argument('--no_enc', type=int, default=0)
parser.add_argument('--pre_trained', type=int, default=0)
parser.add_argument('--one_hop', type=int, default=0)
parser.add_argument('--average_z', type=int, default=0)
parser.add_argument('--bin', type=int, default=1)
parser.add_argument('--enc_hid_dim', type=int, default=64)
parser.add_argument('--dec_hid_dim', type=int, default=64)
parser.add_argument('--dec_msg_dim', type=int, default=64)
# GTS Style Adj Inference Model Hyperparameter
parser.add_argument('--out_channel', type=int, default=8)
parser.add_argument('--kernal_x_1', type=int, default=200)
parser.add_argument('--kernal_x_2', type=int, default=200)
parser.add_argument('--stride_x_1', type=int, default=20)
parser.add_argument('--stride_x_2', type=int, default=20)
parser.add_argument('--gts_totalstep', type=int, default=40000)
parser.add_argument('--znorm', type=str, default='sigmoid') #change
parser.add_argument('--symmode', type=str, default='ltmatonly') #change

args = parser.parse_args()
config_suffix = args.config + '_' + args.dataset + '200'
with open('config/config{}.yml'.format(config_suffix), encoding='UTF8') as f:
    setting = yaml.load(f, Loader=yaml.FullLoader)
settings = setting.get('train')

if args.adj_infmodel == 'nri':
    nri_style = True
    gts_style = False
    if args.adj_infstyle == 'fc':
        circ_shift = False
        spec_dec = False
        fc_style = True
    elif args.adj_infstyle == 'circshift':
        circ_shift = True
        spec_dec = False
        fc_style = False
    elif args.adj_infstyle == 'specdec':
        circ_shift = False
        spec_dec = True
        fc_style = False
    else:
        assert False, 'Invalid structure Inference Model'

elif args.adj_infmodel == 'gts':
    nri_style = False
    gts_style = True
    if args.adj_infstyle == 'fc':
        circ_shift = False
        spec_dec = False
        fc_style = True
    elif args.adj_infstyle == 'circshift':
        circ_shift = True
        spec_dec = False
        fc_style = False
    elif args.adj_infstyle == 'specdec':
        circ_shift = False
        spec_dec = True
        fc_style = False
    else:
        assert False, 'Invalid structure Inference Model'

else:
    assert False, 'Invalid structure Inference Model'
    
# Hyper Parameter Settings
sim_setting = settings.get('sim_setting')
model_params = settings.get('model_params')

num_nodes = args.nodes
edge_types = sim_setting.get('edge_types')
num_edges = num_nodes * (num_nodes - 1)
num_node_features = sim_setting.get('num_node_features')
num_timesteps = args.rectime
num_pinputs = sim_setting.get('num_pinputs')
pred_step = args.pred_step

if args.dataset == 'springs':
    data_suffix = sim_setting.get('data') + str(num_nodes) + '_' + str(args.edges)
elif args.dataset == 'neural_spike':
    data_suffix = sim_setting.get('data') + '/' + args.neurons + str(args.rectime)
model = sim_setting.get('model')

if nri_style:
    if circ_shift:
        z_style = 'nri_circZ'
    elif spec_dec:
        z_style = 'nri_spec_dec'
    elif fc_style:
        z_style = 'nri_fc'
    else:
        assert False, 'Invalid structure Inference Model'

elif gts_style and args.gts_complete:
    if circ_shift:
        z_style = 'gts_cplt_circZ'
    elif spec_dec:
        z_style = 'gts_cplt_spec_dec'
    elif fc_style:
        z_style = 'gts_cplt_fc'
    else:
        assert False, 'Invalid structure Inference Model'

elif gts_style and not(args.gts_complete):
    if circ_shift:
        z_style = 'gts_orig_circZ'
    elif spec_dec:
        z_style = 'gts_orig_spec_dec'
    elif fc_style:
        z_style = 'gts_orig_fc'
    else:
        assert False, 'Invalid structure Inference Model'

else:
    assert False, 'Invalid structure Inference Model'

if not(args.no_enc) and not(gts_style):
    use_enc = True
else:
    use_enc = False

save_folder = (model_params.get('model').get('save_folder') + 
               data_suffix + '/' + args.znorm + '/' + args.decoder + '/' + z_style)
if args.no_enc:
    save_folder = (model_params.get('model').get('save_folder') + data_suffix + 
                   '/' + args.decoder + '/' + z_style + '/' + 'no_enc')
 
plot_folder = 'plot/' + data_suffix
if args.dataset == 'springs':
    suffix = '{}_{}_alphaD{}'.format(data_suffix, model, str(args.alpha).replace('.', ''))
elif args.dataset == 'neural_spike':
    suffix = 'neural_spike_binned'
print(suffix)

alpha = torch.ones(1) * args.alpha
c_reparam = sim_setting.get('c')
lr = model_params.get('training').get('lr')
lr_decay = model_params.get('training').get('lr_decay')
gamma = model_params.get('training').get('gamma')
var = model_params.get('model').get('output_var')
delta_T = 0.01
batchsize = args.batchsize
epochs = args.epochs

enc_hid = args.enc_hid_dim
dec_msg_hid = args.dec_hid_dim
dec_msg_out = args.dec_msg_dim
dec_hid = args.dec_hid_dim
dec_drop = model_params.get('decoder').get('do_prob')

device = sim_setting.get('device') + str(args.device)

al_str = str(args.alpha).replace('.','')
if args.one_hop:
    hop_str = '1'
else:
    hop_str = '2'

if args.average_z:
    aveZ_str = 'aveZ'
else:
    aveZ_str = 'oriZ'

if args.bin:
    bin_str = 'binned'
else:
    bin_str = 'raw'

if nri_style:
    encoder_file = os.path.join(save_folder, 'encoder_hid{}_alpha{}_hop{}_{}_n{}_{}.pt'
                                .format(args.enc_hid_dim, al_str, hop_str, aveZ_str, 
                                        str(args.nodes), bin_str))
    decoder_file = os.path.join(save_folder, 'decoder_hid{}_alpha{}_hop{}_{}_n{}_{}.pt'
                                .format(args.enc_hid_dim, al_str, hop_str, aveZ_str, 
                                        str(args.nodes), bin_str))
    loss_file = os.path.join(save_folder, 'loss_hid{}_alpha{}_hop{}_{}_n{}_{}.npy'
                            .format(args.enc_hid_dim, al_str, hop_str, aveZ_str, 
                                    str(args.nodes), bin_str))
    log_file = os.path.join(save_folder, 'log_hid{}_alpha{}_hop{}_{}_n{}_{}.txt'
                            .format(args.enc_hid_dim, al_str, hop_str, aveZ_str, 
                                    str(args.nodes), bin_str))
elif gts_style:
    encoder_file = os.path.join(save_folder, 'encoder_hid{}_n{}_{}_pred{}_{}.pt'
                                .format(args.enc_hid_dim, str(args.nodes), bin_str, 
                                        str(args.pred_step), str(args.percentage)))
    decoder_file = os.path.join(save_folder, 'decoder_hid{}_n{}_{}_pred{}_{}.pt'
                                .format(args.enc_hid_dim, str(args.nodes), bin_str, 
                                        str(args.pred_step), str(args.percentage)))
    loss_file = os.path.join(save_folder, 'loss_hid{}_n{}_{}_pred{}_{}.npy'
                            .format(args.enc_hid_dim, str(args.nodes), bin_str, 
                                    str(args.pred_step), str(args.percentage)))
    log_file = os.path.join(save_folder, 'log_hid{}_n{}_{}_pred{}_{}.txt'
                            .format(args.enc_hid_dim, str(args.nodes), bin_str, 
                                    str(args.pred_step), str(args.percentage)))    
else:
    assert False, 'Invalid Structure Inference Model'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

if os.path.isfile(log_file):
    log = open(log_file, 'a')
else:
    log = open(log_file, 'w')

device = torch.device(device if torch.cuda.is_available() else 'cpu')

if use_enc:
    encoder = VSCEncoder_NRI(num_node_features*num_timesteps, enc_hid, num_nodes)

if gts_style:
    adj_inf_model = gts_adj_inf(num_nodes, enc_hid, args.out_channel, args.kernal_x_1, args.kernal_x_2, 
                                args.stride_x_1, args.stride_x_2, args.gts_totalstep)
    gts_featmat = np.load('data/gts_featuremat.npy')[:, :args.gts_totalstep]

if args.dataset == 'springs':
    decoder = MLPDecoder_NRI(num_node_features * args.rectime, dec_msg_hid, 
                            dec_msg_out, dec_hid, sim_setting)
elif args.dataset == 'neural_spike':
    if args.decoder == 'MLP':
        decoder = MLPDecoder_NRI_Poisson(num_node_features, dec_msg_hid, 
                                dec_msg_out, dec_hid, sim_setting)
    elif args.decoder == 'RNN':
        decoder = RNNDecoder_NRI_BurnIn(num_node_features, dec_msg_hid, sim_setting,
                                        args.rectime, pred_step, device)
    elif args.decoder == 'RNN_many_to_one':
        decoder = RNNDecoder_NRI_many_to_one(num_node_features, dec_msg_hid, sim_setting, 
                                             args.rectime, pred_step, device)

if args.pre_trained:
    print('Using pre-trained model')
    encoder.load_state_dict(torch.load(encoder_file, map_location=device))
    decoder.load_state_dict(torch.load(decoder_file, map_location=device))
    encoder_file = os.path.join(save_folder, 'encoder_{}_hid{}_alpha{}_hop{}_{}_2.pt'
                                .format(suffix, args.enc_hid_dim, al_str, hop_str, aveZ_str))
    decoder_file = os.path.join(save_folder, 'decoder_{}_hid{}_alpha{}_hop{}_{}_2.pt'
                                .format(suffix, args.enc_hid_dim, al_str, hop_str, aveZ_str))
    loss_file = os.path.join(save_folder, 'loss_{}_hid{}_alpha{}_hop{}_{}_2.npy'
                            .format(suffix, args.enc_hid_dim, al_str, hop_str, aveZ_str))
    log_file = os.path.join(save_folder, 'log_{}_hid{}_alpha{}_hop{}_{}_2.txt'
                            .format(suffix, args.enc_hid_dim, al_str, hop_str, aveZ_str))   

if nri_style:
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
elif gts_style:
    optimizer = optim.Adam(list(adj_inf_model.parameters()) + list(decoder.parameters()), lr=lr)
else: 
    assert False, 'Unknown Assertion'

scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

# Loading datset
if args.dataset == 'springs':
    data_train, data_valid, data_test = load_springs_with_alphaD(num_nodes, args.edges,
                                                                 data_dir, args.alpha)
elif args.dataset == 'neural_spike':
    if args.bin:
        data_train, data_valid, data_test = load_neural_spike_bin(data_dir, args.pred_step, args.percentage)
    else:        
        data_train, data_valid, data_test = load_neural_spike(data_dir, args.neurons, 
                                                              args.rectime, args.percentage)


# batch_size: num of graphs(sims)
train_loader = DL_PyG(data_train, batch_size=batchsize, shuffle=True) 
valid_loader = DL_PyG(data_valid, batch_size=batchsize, shuffle=True)
test_loader = DL_PyG(data_test, batch_size=batchsize, shuffle=True)

A = np.load('data/connectivity_W100.npy')
if spec_dec:
    _, _, A = spec_decompose(A, args.num_spec_vector)
weight_profile = A[~np.eye(A.shape[0],dtype=bool)].reshape(-1, 1)

# generate edge_index for circshift/Fully Connected Graph
fully_connected = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
circshift_edge = np.zeros((num_nodes, num_nodes))
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

def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()
    
    encoder.to(device)
    decoder.to(device)
    alpha.to(device)
    lam = 0
    c_repar_warm = c_reparam

    for batch_idx, data in tqdm(enumerate(train_loader)):
        data = data.to(device)
        x_data, tar_lam_spk = data.x, data.y
        
        data_encoder = x_data[:, :num_node_features * args.rectime]
        batch_size = int(x_data.shape[0] / num_nodes)
        lam_target = tar_lam_spk[:,:,0]
        spk_target = tar_lam_spk[:,:,1]
        
        if circ_shift:
            edge_index = get_edgeidx_by_batchsize(edge_cs, batch_size, num_nodes, device)
            edge_index = edge_index.to(device)

        weight = np.tile(weight_profile, (batch_size, 1))
        #z = torch.FloatTensor(weight)
        #z = z.to(device)

        optimizer.zero_grad()
        
        if args.one_hop:
            mu_sig, log_spike = encoder(data_encoder, edge_index, one_hop=True)
        else:
            mu_sig, log_spike = encoder(data_encoder, edge_index, one_hop=False)
        
        # Force spike variable to be 
        if ((39 < batch_idx) and (batch_idx < 60)) and (epoch < 10):
            mu, logvar = mu_sig[:, 0]*lam, mu_sig[:, 1]*lam + (1 - lam)
            lam = lam + 0.05            
        else:
            mu, logvar = mu_sig[:, 0], mu_sig[:, 1]
        
        mu, logvar = mu_sig[:, 0:1], mu_sig[:, 1:2]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = torch.sigmoid(c_repar_warm*(eta + log_spike.exp() - 1))
        z = selection.mul(gaussian) # z is Coupling capacity
        
        if args.average_z:
            z = z.reshape(batch_size, -1, 1)
            z = z.mean(0)
        if circ_shift:
            if args.average_z:
                z = circshift_z(z, num_nodes, batch_size, ave_z=True)
            else:
                z = circshift_z(z, num_nodes, batch_size)

        # Force z to be symmetric
        else:
            z = make_z_sym(z, num_nodes, batch_size)
        
        edge_decoder = get_edgeidx_by_batchsize(edge_fc, batch_size, num_nodes, device)
        edge_decoder = edge_decoder.to(device)
        output = decoder(x_data, edge_decoder, z, pred_step)
        spike = torch.poisson(output.exp())
        
        #target = x_data.reshape((-1, num_timesteps, num_node_features))[:, 1:, :]
        target = spk_target.reshape(batch_size*num_nodes, pred_step, num_node_features)
        
        if args.dataset == 'springs':
            loss_reconstruction = nll_gaussian(output, target, var)
        elif args.dataset == 'neural_spike':
            loss_reconstruction = F.poisson_nll_loss(output, target, log_input=True)
            #loss_reconstruction = poisson_nll(output, target, 1e-6)
        loss_kl = kl_spike_only(mu_sig, log_spike, alpha, device)

        loss = loss_reconstruction + loss_kl
        loss.backward()
        optimizer.step()

        mse_train.append(F.mse_loss(spike, target).item())
        nll_train.append(loss_reconstruction.item())
        kl_train.append(loss_kl.item())
        
    scheduler.step()

    nll_val = []
    kl_val = []
    mse_val = []

    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            data = data.to(device)
            x_data, tar_lam_spk = data.x, data.y
            data_encoder = x_data[:, :num_node_features * args.rectime]
            batch_size = int(x_data.shape[0] / num_nodes)
            lam_target = tar_lam_spk[:,:,0]
            spk_target = tar_lam_spk[:,:,1]

            if circ_shift:
                edge_index = get_edgeidx_by_batchsize(edge_cs, batch_size, num_nodes, device)
                edge_index = edge_index.to(device)

            weight = np.tile(weight_profile, (batch_size, 1)) 
            #z = torch.FloatTensor(weight)
            #z = z.to(device)

            optimizer.zero_grad()
            if args.one_hop:
                mu_sig, log_spike = encoder(data_encoder, edge_index, one_hop=True)
            else:
                mu_sig, log_spike = encoder(data_encoder, edge_index, one_hop=False)            
            
            # Reparametrization Trick
            mu, logvar = mu_sig[:, 0:1], mu_sig[:, 1:2]
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            gaussian = eps.mul(std).add_(mu)
            eta = torch.rand_like(std)
            selection = torch.sigmoid(c_repar_warm*(eta + log_spike.exp() - 1))
            z = selection.mul(gaussian) # z is Coupling capacity

            if args.average_z:
                z = z.reshape(batch_size, -1, 1)
                z = z.mean(0)
            if circ_shift:
                if args.average_z:
                    z = circshift_z(z, num_nodes, batch_size, ave_z=True)
                else:
                    z = circshift_z(z, num_nodes, batch_size)
                    
            # Force z to be symmetric
            else:
                z = make_z_sym(z, num_nodes, batch_size)
            
            edge_decoder = get_edgeidx_by_batchsize(edge_fc, batch_size, num_nodes, device)
            edge_decoder = edge_decoder.to(device)
            output = decoder(x_data, edge_decoder, z, pred_step)
            spike = torch.poisson(output.exp())
            #target = x_data.reshape((-1, num_timesteps, num_node_features))[:, 1:, :]
            target = spk_target.reshape(batch_size*num_nodes, pred_step, num_node_features)

            if args.dataset == 'springs':
                loss_reconstruction = nll_gaussian(output, target, var)
            elif args.dataset == 'neural_spike':
                loss_reconstruction = F.poisson_nll_loss(output, target, log_input=True)
                #loss_reconstruction = poisson_nll(output, target, 1e-6)
            loss_kl = kl_spike_only(mu_sig, log_spike, alpha, device)
            
            loss = loss_reconstruction + loss_kl

            mse_val.append(F.mse_loss(spike, target).item())
            nll_val.append(loss_reconstruction.item())
            kl_val.append(loss_kl.item())
        
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(nll_val) < best_val_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'kl_train: {:.10f}'.format(np.mean(kl_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'nll_val: {:.10f}'.format(np.mean(nll_val)),
                  'kl_val: {:.10f}'.format(np.mean(kl_val)),
                  'mse_val: {:.10f}'.format(np.mean(mse_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
        
        loss = np.array([np.mean(nll_train), np.mean(kl_train), np.mean(mse_train), 
                         np.mean(nll_val), np.mean(kl_val), np.mean(mse_val)])
    return np.mean(nll_val), loss

def train_no_enc(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    mse_train = []

    encoder.train()
    decoder.train()
    
    encoder.to(device)
    decoder.to(device)
    alpha.to(device)
    lam = 0
    c_repar_warm = c_reparam
    
    for batch_idx, data in tqdm(enumerate(train_loader)):
        data = data.to(device)
        x_data, edge_index, tar_lam_spk = data.x, data.edge_index, data.y
        batch_size = int(x_data.shape[0] / num_nodes)
        lam_target = tar_lam_spk[:,:,0]
        spk_target = tar_lam_spk[:,:,1]
        
        weight = np.tile(weight_profile, (batch_size, 1))
        z = torch.FloatTensor(weight)
        z = z.to(device)

        optimizer.zero_grad()
        
        output = decoder(x_data, edge_index, z, pred_step)
        spike = torch.poisson(output.exp())
        
        target = spk_target.reshape(batch_size*num_nodes, pred_step, num_node_features)
        
        if args.dataset == 'springs':
            loss_reconstruction = nll_gaussian(output, target, var)
        elif args.dataset == 'neural_spike':
            loss_reconstruction = F.poisson_nll_loss(output, target, log_input=True)

        loss = loss_reconstruction
        loss.backward()
        optimizer.step()

        mse_train.append(F.mse_loss(spike, target).item())
        nll_train.append(loss_reconstruction.item())
        
    scheduler.step()

    nll_val = []
    mse_val = []

    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            data = data.to(device)
            x_data, edge_index, tar_lam_spk = data.x, data.edge_index, data.y
            batch_size = int(x_data.shape[0] / num_nodes)
            lam_target = tar_lam_spk[:,:,0]
            spk_target = tar_lam_spk[:,:,1]

            weight = np.tile(weight_profile, (batch_size, 1)) 
            z = torch.FloatTensor(weight)
            z = z.to(device)

            optimizer.zero_grad()

            
            output = decoder(x_data, edge_index, z, pred_step)
            spike = torch.poisson(output.exp())
            target = spk_target.reshape(batch_size*num_nodes, pred_step, num_node_features)

            if args.dataset == 'springs':
                loss_reconstruction = nll_gaussian(output, target, var)
            elif args.dataset == 'neural_spike':
                loss_reconstruction = F.poisson_nll_loss(output, target, log_input=True)

            loss = loss_reconstruction

            mse_val.append(F.mse_loss(spike, target).item())
            nll_val.append(loss_reconstruction.item())
        
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(nll_val) < best_val_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'nll_val: {:.10f}'.format(np.mean(nll_val)),
                  'mse_val: {:.10f}'.format(np.mean(mse_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
        
        loss = np.array([np.mean(nll_train), np.mean(mse_train), 
                         np.mean(nll_val), np.mean(mse_val)])
    return np.mean(nll_val), loss

def train_gts(epoch, best_val_loss):
    t = time.time()
    
    if gts_style:
        gts_input = torch.FloatTensor(gts_featmat)
        gts_input = gts_input.to(device)
    
    nll_train = []
    mse_train = []

    adj_inf_model.train()
    decoder.train()
    
    adj_inf_model.to(device)
    decoder.to(device)

    for batch_idx, data in tqdm(enumerate(train_loader)):
        data = data.to(device)
        x_data, tar_lam_spk = data.x, data.y
        
        batch_size = int(x_data.shape[0] / num_nodes)
        spk_target = tar_lam_spk[:,:,1]
        
        optimizer.zero_grad()
        if args.adj_infstyle == 'fc':
            z_logits, z_corr = adj_inf_model(gts_input, edge_fc)
        elif args.adj_infstyle == 'circshift':
            z_logits, z_corr = adj_inf_model(gts_input, edge_cs)
        else:
            assert False, 'Assertion'
        
        z_adj = F.gumbel_softmax(z_logits, hard=True)
        # Complete Graph Mode
        if args.gts_complete:
            if args.adj_infstyle == 'fc':
                z_corr = make_z_sym_gts(z_corr, num_nodes, device, args.symmode)
                z = z_corr

            elif args.adj_infstyle == 'circshift':
                z = circshift_z(z_corr, num_nodes, 1, ave_z=True)

            else:
                assert False, 'Assertion'
        # Using Gumbel-Softmax GTS Style
        else:
            if args.adj_infstyle == 'fc':
                z_adj = make_z_sym_gts(z_adj[:, 0].clone(), num_nodes, device, args.symmode)
                z_corr = make_z_sym_gts(z_corr, num_nodes, device, args.symmode)
                
                z_corr_idx = torch.where(z_adj)
                z = z_corr[z_corr_idx].clone()

            elif args.adj_infstyle == 'circshift':
                z_adj_cs_all = circshift_z(z_adj[:, 0].clone(), num_nodes, 1, ave_z=True)
                z = circshift_z(z_corr, num_nodes, 1, ave_z=True)
                
                z_adj = make_z_sym_gts(z_adj_cs_all, num_nodes, device, args.symmode)

                z_corr_idx = torch.where(z_adj)
                z = z[z_corr_idx].clone()

            else:
                assert False, 'Assertion'

        if args.znorm == 'relu':
            z = -F.relu(z)
        elif args.znorm == 'sigmoid':
            z = -torch.sigmoid(z)
        elif args.znorm == 'exp':
            z = -z.exp()
        elif args.znorm == 'raw':
            z = z
        else:
            assert False, 'Invalid Normalization of Z'

        expand_size = [batch_size] + list(z.shape)
        z_expand = z.unsqueeze(0).expand(expand_size).clone()
        z_expand = z_expand.reshape(-1, 1)

        if args.gts_complete:
            adj = get_edgeidx_by_batchsize(edge_fc, batch_size, num_nodes, device)
            edge_dec_batch = adj.to(device)
        else: 
            adj = z_to_adj(z_adj, num_nodes, device)
            edge_dec = adj_to_edge_idx_pyg(adj)
            edge_dec_batch = get_edgeidx_by_batchsize(edge_dec, batch_size, num_nodes, device)

        output = decoder(x_data, edge_dec_batch, z_expand, pred_step)
        spike = torch.poisson(output.exp())
        target = spk_target.reshape(batch_size*num_nodes, pred_step, num_node_features)
        
        if args.dataset == 'springs':
            loss_reconstruction = nll_gaussian(output, target, var)
        elif args.dataset == 'neural_spike':
            loss_reconstruction = F.poisson_nll_loss(output, target, log_input=True)

        loss = loss_reconstruction
        loss.backward()
        optimizer.step()

        mse_train.append(F.mse_loss(spike, target).item())
        nll_train.append(loss_reconstruction.item())
        
    scheduler.step()

    nll_val = []
    mse_val = []

    adj_inf_model.eval()
    decoder.eval()
    
    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            data = data.to(device)
            x_data, tar_lam_spk = data.x, data.y
            
            batch_size = int(x_data.shape[0] / num_nodes)
            spk_target = tar_lam_spk[:,:,1]
            
            if args.adj_infstyle == 'fc':
                z_logits, z_corr = adj_inf_model(gts_input, edge_fc)
            elif args.adj_infstyle == 'circshift':
                z_logits, z_corr = adj_inf_model(gts_input, edge_cs)
            else:
                assert False, 'Assertion'
            
            z_adj = F.gumbel_softmax(z_logits, hard=True)
            
            if args.gts_complete:
                if args.adj_infstyle == 'fc':
                    z_corr = make_z_sym_gts(z_corr, num_nodes, device, args.symmode)
                    z = z_corr

                elif args.adj_infstyle == 'circshift':
                    z = circshift_z(z_corr, num_nodes, 1, ave_z=True)

                else:
                    assert False, 'Assertion'
            # Using Gumbel-Softmax GTS Style
            else:
                if args.adj_infstyle == 'fc':
                    z_adj = make_z_sym_gts(z_adj[:, 0].clone(), num_nodes, device, args.symmode)
                    z_corr = make_z_sym_gts(z_corr, num_nodes, device, args.symmode)
                    
                    z_corr_idx = torch.where(z_adj)
                    z = z_corr[z_corr_idx].clone()

                elif args.adj_infstyle == 'circshift':
                    z_adj_cs_all = circshift_z(z_adj[:, 0].clone(), num_nodes, 1, ave_z=True)
                    z = circshift_z(z_corr, num_nodes, 1, ave_z=True)
                    
                    z_adj = make_z_sym_gts(z_adj_cs_all, num_nodes, device, args.symmode)

                    z_corr_idx = torch.where(z_adj)
                    z = z[z_corr_idx].clone()

                else:
                    assert False, 'Assertion'

            if args.znorm == 'relu':
                z = -F.relu(z)
            elif args.znorm == 'sigmoid':
                z = -torch.sigmoid(z)
            elif args.znorm == 'exp':
                z = -z.exp()
            elif args.znorm == 'raw':
                z = z
            else:
                assert False, 'Invalid Normalization of Z'

            expand_size = [batch_size] + list(z.shape)
            z_expand = z.unsqueeze(0).expand(expand_size).clone()
            z_expand = z_expand.reshape(-1, 1)

            if args.gts_complete:
                adj = get_edgeidx_by_batchsize(edge_fc, batch_size, num_nodes, device)
                edge_dec_batch = adj.to(device)
            else: 
                adj = z_to_adj(z_adj, num_nodes, device)
                edge_dec = adj_to_edge_idx_pyg(adj)
                edge_dec_batch = get_edgeidx_by_batchsize(edge_dec, batch_size, num_nodes, device)

            output = decoder(x_data, edge_dec_batch, z_expand, pred_step)
            spike = torch.poisson(output.exp())
            target = spk_target.reshape(batch_size*num_nodes, pred_step, num_node_features)

            if args.dataset == 'springs':
                loss_reconstruction = nll_gaussian(output, target, var)
            elif args.dataset == 'neural_spike':
                loss_reconstruction = F.poisson_nll_loss(output, target, log_input=True)
            
            loss = loss_reconstruction

            mse_val.append(F.mse_loss(spike, target).item())
            nll_val.append(loss_reconstruction.item())
        
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(nll_val) < best_val_loss:
            torch.save(adj_inf_model.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'nll_val: {:.10f}'.format(np.mean(nll_val)),
                  'mse_val: {:.10f}'.format(np.mean(mse_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
        
        loss = np.array([np.mean(nll_train), np.mean(mse_train), 
                         np.mean(nll_val), np.mean(mse_val)])
    return np.mean(nll_val), loss

t_total = time.time()
best_val_loss = np.inf
best_epoch = 0

loss_list = []
if args.no_enc:
    for epoch in tqdm(range(epochs)):
        val_loss, loss = train_no_enc(epoch, best_val_loss)
        loss_list.append(loss)
        c_reparam += 0.001
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

elif gts_style:
    for epoch in tqdm(range(epochs)):
        val_loss, loss = train_gts(epoch, best_val_loss)
        loss_list.append(loss)
        c_reparam += 0.001
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

else:
    for epoch in tqdm(range(epochs)):
        val_loss, loss = train(epoch, best_val_loss)
        loss_list.append(loss)
        c_reparam += 0.001
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

np.save(loss_file, np.stack(loss_list, axis=0))