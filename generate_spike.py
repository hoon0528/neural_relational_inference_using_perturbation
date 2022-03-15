import torch
import numpy as np
import mat73
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as DL_PyG
from torch.utils.data import DataLoader as DL_Py
from torch.utils.data.dataset import TensorDataset
from scipy.io import loadmat

from torch.utils.data import Dataset

import os
from tqdm import tqdm
root_dir = os.getcwd()
data_dir = root_dir + '/data'

# [sims, tsteps, features, nodes] / x = [nodes, features], edge_index = []
# edge > adjacent matrix [sims, nodes, nodes]

# Using PyTorch Geometric

class spike_LNP_raw_train_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_raw_train_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return ['neuron/spike/LNP/raw/train_whole_20.pt']
        return ['neuron/spike/LNP/raw/train_n50.pt']
        #return ['neuron/spike/LNP/raw/train_whole_40.pt']
        #return ['neuron/spike/LNP/raw/train_whole_200.pt']
    
    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/raw'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/raw')

        #lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_all.mat')
        #spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_all.mat')
        lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_n50.mat')
        spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_n50.mat')
        
        lam = lam['lambda']
        spk = spk['spikes']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[:4000000]
        lam = lam[:4000000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        for i in tqdm(range(batch_size)):
            step = i * (window_size+1)
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_raw_valid_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_raw_valid_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return ['neuron/spike/LNP/raw/valid_whole_20.pt']
        return ['neuron/spike/LNP/raw/valid_n50.pt']
        #return ['neuron/spike/LNP/raw/valid_whole_40.pt']
        #return ['neuron/spike/LNP/raw/valid_whole_200.pt']
    
    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/raw'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/raw')

        #lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_all.mat')
        #spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_all.mat')
        lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_n50.mat')
        spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_n50.mat')
        
        lam = lam['lambda']
        spk = spk['spikes']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[4000000:4400000]
        lam = lam[4000000:4400000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        for i in tqdm(range(batch_size)):
            step = i * (window_size+1)
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_raw_test_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_raw_test_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return ['neuron/spike/LNP/raw/test_whole_20.pt']
        return ['neuron/spike/LNP/raw/test_n50.pt']
        #return ['neuron/spike/LNP/raw/test_whole_40.pt']
        #return ['neuron/spike/LNP/raw/test_whole_200.pt']
    
    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/raw'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/raw')

        #lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_all.mat')
        #spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_all.mat')
        lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_n50.mat')
        spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_n50.mat')
        
        lam = lam['lambda']
        spk = spk['spikes']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[4400000:4800000]
        lam = lam[4400000:4800000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        #pred_steps = 40 #steps to predict
        #pred_steps = 200 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        iter_over = int((total_time - time_steps) / pred_steps)
        for i in tqdm(range(iter_over)):
            step = i * pred_steps
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_bin_train_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_bin_train_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['neuron/spike/LNP/bin/train_whole_n100.pt']

    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/bin'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/bin')

        lam = mat73.loadmat('data/binary_spike_bin/lam_bin_n100.mat')
        spk = mat73.loadmat('data/binary_spike_bin/spk_bin_n100.mat')
        
        lam = lam['lam_bin']
        spk = spk['spk_bin']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[:4000000]
        lam = lam[:4000000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        for i in tqdm(range(batch_size)):
            step = i * (window_size+1)
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_bin_valid_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_bin_valid_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['neuron/spike/LNP/bin/valid_whole_n100.pt']
    
    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/bin'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/bin')

        lam = mat73.loadmat('data/binary_spike_bin/lam_bin_n100.mat')
        spk = mat73.loadmat('data/binary_spike_bin/spk_bin_n100.mat')
        
        lam = lam['lam_bin']
        spk = spk['spk_bin']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[4000000:4400000]
        lam = lam[4000000:4400000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        for i in tqdm(range(batch_size)):
            step = i * (window_size+1)
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_bin_test_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_bin_test_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['neuron/spike/LNP/bin/test_whole_n100.pt']

    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/bin'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/bin')

        lam = mat73.loadmat('data/binary_spike_bin/lam_bin_n100.mat')
        spk = mat73.loadmat('data/binary_spike_bin/spk_bin_n100.mat')
        
        lam = lam['lam_bin']
        spk = spk['spk_bin']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[4400000:4800000]
        lam = lam[4400000:4800000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        iter_over = int((total_time - time_steps) / pred_steps)
        for i in tqdm(range(iter_over)):
            step = i * pred_steps
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class LNP_perturb_bin(Dataset):
    def __init__(self, pre_step, pred_step, p1_bs, dataset, length_ratio):
        '''For dataset, use (train, valid or test)'''
        self.spike_mat = mat73.loadmat('data/binary_spike_bin/spk_bin_n100.mat')
        self.history = pre_step
        self.pred = pred_step
        self.phase1_batchsize = p1_bs
        self.dataset = dataset

        spike_whole = self.spike_mat['spk_bin']
        spike_whole = spike_whole.transpose((1, 0))

        num_neurons = spike_whole.shape[0]
        total_steps = spike_whole.shape[1]
        history = self.history
        window_size = history + pred_step - 1
        iter_step = int(np.floor(total_steps  / (history*length_ratio)) - 1) 
        # Memory Issue of getting 48K steps during Training Phase 2
        phase1_batch = self.phase1_batchsize
        
        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        edge_idx_complete = np.where(fully_connected)
        edge_idx_complete = np.array([edge_idx_complete[0], edge_idx_complete[1]], dtype=np.int64)

        spike_whole = torch.FloatTensor(spike_whole)
        self.edge_idx = torch.LongTensor(edge_idx_complete)

        spike_window = []
        spike_target = []

        for i in tqdm(range(iter_step)):
            step = i * history
            spike_window.append(spike_whole[:, step:step+window_size])
            spike_target.append(spike_whole[:, step+history:step+history+pred_step])
        
        spike_window_whole = torch.stack(spike_window, dim=1)
        spike_target_whole = torch.stack(spike_target, dim=1)

        # Cut last few steps of whole windowed spike
        whole_cut = int(spike_window_whole.shape[1] / phase1_batch) * phase1_batch 

        spike_window_whole = spike_window_whole[:, :whole_cut, :]
        spike_target_whole = spike_target_whole[:, :whole_cut, :]

        spike_window_whole = spike_window_whole.reshape(num_neurons, -1, phase1_batch, window_size)
        spike_target_whole = spike_target_whole.reshape(num_neurons, -1, phase1_batch, pred_step)
        total_samples = spike_window_whole.shape[1]

        tr_eidx = int(total_samples * 5 / 6) + 1
        val_eidx = tr_eidx + int(total_samples / 12) + 1

        if self.dataset == 'train':
            self.x = spike_window_whole[:, :tr_eidx, :, :]
            self.y = spike_target_whole[:, :tr_eidx, :, :]
            self.len = self.x.shape[1]

        elif self.dataset == 'valid':
            self.x = spike_window_whole[:, tr_eidx:val_eidx, :, :]
            self.y = spike_target_whole[:, tr_eidx:val_eidx, :, :]
            self.len = self.x.shape[1]
        
        elif self.dataset == 'test':
            self.x = spike_window_whole[:, val_eidx:, :, :]
            self.y = spike_target_whole[:, val_eidx:, :, :]
            self.len = self.x.shape[1]

        else:
            assert False, 'Invalid dataset type'

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[:, idx, :, :]
        target = self.y[:, idx, :, :]
        return x, target, self.edge_idx

class LNP_perturb_bin_plot(Dataset):
    def __init__(self, pre_step, pred_step, p1_bs, dataset, length_ratio):
        '''For dataset, use (train, valid or test)'''
        self.spike_mat = mat73.loadmat('data/binary_spike_bin/spk_bin_n100.mat')
        self.lam_mat = mat73.loadmat('data/binary_spike_bin/lam_bin_n100.mat')
        self.history = pre_step
        self.pred = pred_step
        self.phase1_batchsize = p1_bs
        self.dataset = dataset

        spike_whole = self.spike_mat['spk_bin']
        spike_whole = spike_whole.transpose((1, 0))
        spike_whole = spike_whole[:, 4400000:]

        lam_whole = self.lam_mat['lam_bin']
        lam_whole = lam_whole.transpose((1, 0))
        lam_whole = lam_whole[:, 4400000:]

        num_neurons = spike_whole.shape[0]
        total_steps = spike_whole.shape[1]
        history = self.history
        window_size = history + pred_step - 1
        iter_step = int(np.floor((total_steps - history)  / pred_step))
        # Memory Issue of getting 48K steps during Training Phase 2
        phase1_batch = self.phase1_batchsize
        
        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        edge_idx_complete = np.where(fully_connected)
        edge_idx_complete = np.array([edge_idx_complete[0], edge_idx_complete[1]], dtype=np.int64)

        spike_whole = torch.FloatTensor(spike_whole)
        lam_whole = torch.FloatTensor(lam_whole)
        self.edge_idx = torch.LongTensor(edge_idx_complete)

        spike_window = []
        spike_target = []
        lam_window = []
        lam_target = []

        for i in tqdm(range(iter_step)):
            step = i * pred_step
            spike_window.append(spike_whole[:, step:step+window_size])
            spike_target.append(spike_whole[:, step+history:step+history+pred_step])
            lam_window.append(lam_whole[:, step:step+window_size])
            lam_target.append(lam_whole[:, step+history:step+history+pred_step])

        spike_window_whole = torch.stack(spike_window, dim=1)
        spike_target_whole = torch.stack(spike_target, dim=1)
        lam_window_whole = torch.stack(lam_window, dim=1)
        lam_target_whole = torch.stack(lam_target, dim=1)
        # Cut last few steps of whole windowed spike
        whole_cut = int(spike_window_whole.shape[1] / phase1_batch) * phase1_batch 

        spike_window_whole = spike_window_whole[:, :whole_cut, :]
        spike_target_whole = spike_target_whole[:, :whole_cut, :]
        lam_window_whole = lam_window_whole[:, :whole_cut, :]
        lam_target_whole = lam_target_whole[:, :whole_cut, :]

        spike_window_whole = spike_window_whole.reshape(num_neurons, -1, phase1_batch, window_size)
        spike_target_whole = spike_target_whole.reshape(num_neurons, -1, phase1_batch, pred_step)
        lam_window_whole = lam_window_whole.reshape(num_neurons, -1, phase1_batch, window_size)
        lam_target_whole = lam_target_whole.reshape(num_neurons, -1, phase1_batch, pred_step)
        total_samples = spike_window_whole.shape[1]

        self.spk_ins = spike_window_whole
        self.spk_tar = spike_target_whole
        self.lam_ins = lam_window_whole
        self.lam_tar = lam_target_whole
        self.len = self.lam_tar.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        spk_ins = self.spk_ins[:, idx, :, :]
        spk_tar = self.spk_tar[:, idx, :, :]
        lam_tar = self.lam_tar[:, idx, :, :]
        lam_ins = self.lam_ins[:, idx, :, :]
        return spk_ins, spk_tar, lam_tar, lam_ins, self.edge_idx

class LNP_perturb_bin_tar(Dataset):
    def __init__(self, pre_step, pred_step, p1_bs, dataset, length_ratio):
        '''For dataset, use (train, valid or test)'''
        self.spike_mat = np.load('data/spk_bin_per.npy')
        self.history = pre_step
        self.pred = pred_step
        self.phase1_batchsize = p1_bs
        self.dataset = dataset

        spike_whole = torch.FloatTensor(self.spike_mat)

        num_neurons = spike_whole.shape[0]
        total_steps = spike_whole.shape[1]
        history = self.history
        window_size = history + pred_step - 1
        iter_step = int(np.floor(total_steps  / (history*length_ratio)) - 1) 
        # Memory Issue of getting 48K steps during Training Phase 2
        phase1_batch = self.phase1_batchsize
        
        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        edge_idx_complete = np.where(fully_connected)
        edge_idx_complete = np.array([edge_idx_complete[0], edge_idx_complete[1]], dtype=np.int64)

        spike_whole = torch.FloatTensor(spike_whole)
        self.edge_idx = torch.LongTensor(edge_idx_complete)

        spike_window = []
        spike_target = []

        for i in tqdm(range(iter_step)):
            step = i * history
            spike_window.append(spike_whole[:, step:step+window_size])
            spike_target.append(spike_whole[:, step+history:step+history+pred_step])
        
        spike_window_whole = torch.stack(spike_window, dim=1)
        spike_target_whole = torch.stack(spike_target, dim=1)

        # Cut last few steps of whole windowed spike
        whole_cut = int(spike_window_whole.shape[1] / phase1_batch) * phase1_batch 

        spike_window_whole = spike_window_whole[:, :whole_cut, :]
        spike_target_whole = spike_target_whole[:, :whole_cut, :]

        spike_window_whole = spike_window_whole.reshape(num_neurons, -1, phase1_batch, window_size)
        spike_target_whole = spike_target_whole.reshape(num_neurons, -1, phase1_batch, pred_step)
        total_samples = spike_window_whole.shape[1]

        tr_eidx = int(total_samples * 5 / 6) + 1
        val_eidx = tr_eidx + int(total_samples / 12) + 1

        if self.dataset == 'train':
            self.x = spike_window_whole[:, :tr_eidx, :, :]
            self.y = spike_target_whole[:, :tr_eidx, :, :]
            self.len = self.x.shape[1]

        elif self.dataset == 'valid':
            self.x = spike_window_whole[:, tr_eidx:val_eidx, :, :]
            self.y = spike_target_whole[:, tr_eidx:val_eidx, :, :]
            self.len = self.x.shape[1]
        
        elif self.dataset == 'test':
            self.x = spike_window_whole[:, val_eidx:, :, :]
            self.y = spike_target_whole[:, val_eidx:, :, :]
            self.len = self.x.shape[1]

        else:
            assert False, 'Invalid dataset type'

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[:, idx, :, :]
        target = self.y[:, idx, :, :]
        return x, target, self.edge_idx

class LNP_perturb_bin_tar_plot(Dataset):
    def __init__(self, pre_step, pred_step, p1_bs, dataset):
        '''For dataset, use (train, valid or test)'''
        self.spike_mat = np.load('data/spk_bin_per.npy')
        self.lam_mat = np.load('data/lam_bin_per.npy')
        self.history = pre_step
        self.pred = pred_step
        self.phase1_batchsize = p1_bs
        self.dataset = dataset

        spike_whole = torch.FloatTensor(self.spike_mat)
        spike_whole = spike_whole[:, 528000:]

        lam_whole = torch.FloatTensor(self.lam_mat)
        lam_whole = lam_whole[:, 528000:]

        num_neurons = spike_whole.shape[0]
        total_steps = spike_whole.shape[1]
        history = self.history
        window_size = history + pred_step - 1
        iter_step = int(np.floor((total_steps - history)  / pred_step))
        # Memory Issue of getting 48K steps during Training Phase 2
        phase1_batch = self.phase1_batchsize
        
        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        edge_idx_complete = np.where(fully_connected)
        edge_idx_complete = np.array([edge_idx_complete[0], edge_idx_complete[1]], dtype=np.int64)

        spike_whole = torch.FloatTensor(spike_whole)
        lam_whole = torch.FloatTensor(lam_whole)
        self.edge_idx = torch.LongTensor(edge_idx_complete)

        spike_window = []
        spike_target = []
        lam_window = []
        lam_target = []

        for i in tqdm(range(iter_step)):
            step = i * pred_step
            spike_window.append(spike_whole[:, step:step+window_size])
            spike_target.append(spike_whole[:, step+history:step+history+pred_step])
            lam_window.append(lam_whole[:, step:step+window_size])
            lam_target.append(lam_whole[:, step+history:step+history+pred_step])

        spike_window_whole = torch.stack(spike_window, dim=1)
        spike_target_whole = torch.stack(spike_target, dim=1)
        lam_window_whole = torch.stack(lam_window, dim=1)
        lam_target_whole = torch.stack(lam_target, dim=1)
        # Cut last few steps of whole windowed spike
        whole_cut = int(spike_window_whole.shape[1] / phase1_batch) * phase1_batch 

        spike_window_whole = spike_window_whole[:, :whole_cut, :]
        spike_target_whole = spike_target_whole[:, :whole_cut, :]
        lam_window_whole = lam_window_whole[:, :whole_cut, :]
        lam_target_whole = lam_target_whole[:, :whole_cut, :]

        spike_window_whole = spike_window_whole.reshape(num_neurons, -1, phase1_batch, window_size)
        spike_target_whole = spike_target_whole.reshape(num_neurons, -1, phase1_batch, pred_step)
        lam_window_whole = lam_window_whole.reshape(num_neurons, -1, phase1_batch, window_size)
        lam_target_whole = lam_target_whole.reshape(num_neurons, -1, phase1_batch, pred_step)

        self.spk_ins = spike_window_whole
        self.spk_tar = spike_target_whole
        self.lam_ins = lam_window_whole
        self.lam_tar = lam_target_whole
        self.len = self.lam_tar.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        spk_ins = self.spk_ins[:, idx, :, :]
        spk_tar = self.spk_tar[:, idx, :, :]
        lam_tar = self.lam_tar[:, idx, :, :]
        lam_ins = self.lam_ins[:, idx, :, :]
        return spk_ins, spk_tar, lam_tar, lam_ins, self.edge_idx

class data_spike_poisson_rate(Dataset):
    def __init__(self, history, pred_step, p1_bs, dataset, if_binned, 
                 neuron_type, recording, length_ratio, data_circshift):
        '''
        For dataset, use (train, valid or test)
        recording: {eq, pt} Equilibrium or Perturbation
        neuron_type: {LNP, ring}
        '''
        if if_binned:
            self.spike_mat = np.load('data/spk_bin_{}_{}.npy'.format(neuron_type, recording))
            self.lam_mat = np.load('data/lam_bin_{}_{}.npy'.format(neuron_type, recording))
        else:
            self.spike_mat = np.load('data/spk_raw_{}_{}.npy'.format(neuron_type, recording))
            self.lam_mat = np.load('data/lam_raw_{}_{}.npy'.format(neuron_type, recording))
        self.history = history
        self.pred = pred_step
        self.phase1_batchsize = p1_bs
        self.dataset = dataset
        self.data_cs = data_circshift

        spike_whole = torch.FloatTensor(self.spike_mat)
        lam_whole = torch.FloatTensor(self.lam_mat)

        num_neurons = spike_whole.shape[0]
        total_steps = spike_whole.shape[1]
        history = self.history
        window_size = history + pred_step - 1
        
        data_length = int(total_steps / length_ratio)
        idx_tr_end = int(data_length*0.8)
        idx_val_end = int(data_length*0.9)
        idx_test_end = data_length

        if dataset == 'train':
            iter_step = int(np.floor((data_length*0.8-history) / pred_step) - 1)            
            spike_whole = spike_whole[:, :idx_tr_end]
            lam_whole = lam_whole[:, :idx_tr_end]
        elif dataset == 'valid':
            iter_step = int(np.floor((data_length*0.1-history) / pred_step) - 1)
            spike_whole = spike_whole[:, idx_tr_end:idx_val_end]
            lam_whole = lam_whole[:, idx_tr_end:idx_val_end]
        elif dataset == 'test':
            iter_step = int(np.floor((data_length*0.1-history) / pred_step) - 1)
            spike_whole = spike_whole[:, idx_val_end:idx_test_end]
            lam_whole = lam_whole[:, idx_val_end:idx_test_end]
        else:
            assert False, 'Invalid dataset, put train/valid/test'
        # Memory Issue of getting 48K steps during Training Phase 2
        phase1_batch = self.phase1_batchsize
        
        edge_fullyconnected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        edge_idx_fullyconnected = np.where(edge_fullyconnected)
        edge_idx_fullyconnected = np.array([edge_idx_fullyconnected[0], edge_idx_fullyconnected[1]], dtype=np.int64)

        spike_whole = torch.FloatTensor(spike_whole)
        lam_whole = torch.FloatTensor(lam_whole)
        self.edge_idx = torch.LongTensor(edge_idx_fullyconnected)

        spike_window = []
        spike_target = []
        lam_target = []

        for i in tqdm(range(iter_step)):
            if (dataset == 'train') or (dataset == 'valid'):
                #step = i * history
                step = i * pred_step
                spike_window.append(spike_whole[:, step:step+window_size])
                spike_target.append(spike_whole[:, step+history:step+history+pred_step])
                lam_target.append(lam_whole[:, step+history:step+history+pred_step])
            elif dataset == 'test':
                step = i * pred_step
                spike_window.append(spike_whole[:, step:step+window_size])
                spike_target.append(spike_whole[:, step+history:step+history+pred_step])
                lam_target.append(lam_whole[:, step+history:step+history+pred_step])
            else:
                assert False, 'Invalid dataset, put train/valid/test'
        
        spike_window_whole = torch.stack(spike_window, dim=1)
        spike_target_whole = torch.stack(spike_target, dim=1)
        lam_target_whole = torch.stack(lam_target, dim=1)

        # Cut last few steps of whole windowed spike
        if spike_window_whole.shape[1] < phase1_batch:
            phase1_batch = spike_window_whole.shape[1]
        whole_cut = int(spike_window_whole.shape[1] / phase1_batch) * phase1_batch 

        spike_window_whole = spike_window_whole[:, :whole_cut, :]
        spike_target_whole = spike_target_whole[:, :whole_cut, :]
        lam_target_whole = lam_target_whole[:, :whole_cut, :]

        # Including all nodes
        spike_window_whole = spike_window_whole.reshape(num_neurons, -1, phase1_batch, window_size)
        spike_target_whole = spike_target_whole.reshape(num_neurons, -1, phase1_batch, pred_step)
        lam_target_whole = lam_target_whole.reshape(num_neurons, -1, phase1_batch, pred_step)
        
        if data_circshift:
            window_nodewise_size = [num_neurons] + list(spike_window_whole.shape)
            
            #[nodes, 1, num_batchs, p1_bs, window_size] > [nodes, nodes, num_batchs, p1_bs, window_size]
            spike_window_nodewise = torch.empty(window_nodewise_size)
        
            for node_index in range(num_neurons):
                spike_window_nodewise[:,node_index,:,:,:] = torch.roll(spike_window_whole, shifts=node_index, dims=0)

            spike_window_nodewise = spike_window_nodewise.reshape(num_neurons, -1, phase1_batch, window_size)
            spike_target_whole = spike_target_whole.reshape(-1, phase1_batch, pred_step)
            lam_target_whole = lam_target_whole.reshape(-1, phase1_batch, pred_step)

            self.x = spike_window_nodewise
            self.y = spike_target_whole
            self.y_lam = lam_target_whole
            self.len = self.x.shape[1]

        else:
            self.x = spike_window_whole
            self.y = spike_target_whole
            self.y_lam = lam_target_whole
            self.len = spike_window_whole.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.data_cs:
            if self.dataset == 'train' or self.dataset == 'valid':
                x = self.x[:, idx, :, :]
                target = self.y[idx, :, :]
                return x, target, self.edge_idx
            elif self.dataset == 'test':
                x = self.x[:, idx, :, :]
                target = self.y[idx, :, :]
                target_lam = self.y_lam[idx, :, :]
                return x, target, target_lam, self.edge_idx
            else:
                assert False, 'Invalid dataset, put train/valid/test'
        else:
            if self.dataset == 'train' or self.dataset == 'valid':
                x = self.x[:, idx, :, :]
                y = self.y[:, idx, :, :]
                return x, y, self.edge_idx
            else:
                x = self.x[:, idx, :, :]
                y = self.y[:, idx, :, :]
                lam_tar = self.y_lam[:, idx, :, :]
                return x, y, lam_tar, self.edge_idx
