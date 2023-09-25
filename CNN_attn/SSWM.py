import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from math import nan
from loading_data import CustomData
from loading_data import DeviceDataLoader
from neural_network import NeuralNetwork
from utils import get_default_device
from utils import to_device
from utils import evaluate_raw
from utils import load_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import scipy.special as sp
from math import nan
import sys

# direction of selection in head and testis
#print(sys.argv)
direction = [float(sys.argv[1]), float(sys.argv[2])]




test_coor = pd.read_csv('../model_indat/test_coors__2tis_2057')
# sample a thousand nonpeaks for example
samples = (test_coor.pipe(lambda df: df[(df.label_head==1) & (df.label_testis==1)]).
sample(1000, random_state=1).
index.values)

def load_testdata(suffix):
    X_test = np.load('../model_indat/X_test_' + suffix + '.npy')

    y_test = np.load('../model_indat/y_test_'+ suffix + '.npy')
    return (X_test, y_test)
def get_loader(X_test, y_test, device):
    batch_size = 32
    # train_data = CustomData(X_train, y_train)
    test_data = CustomData(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)
    test_loader = DeviceDataLoader(test_loader, device)
    return test_loader

def get_seqdata_mut(test_seq):
    test_seq = test_seq.copy()
    seqlen = test_seq.shape[1]
    X = np.zeros([4*seqlen, 4, seqlen], dtype=np.float32)
    y = np.zeros([4*seqlen, 2])
    ref = np.zeros([4*seqlen])
    mut = np.tile([0.0,1.0,2.0,3.0], seqlen)
    pos_array = np.repeat(range(seqlen),4)
    # scan over entire sequence and mutate
    # a t c g is the code
    for pos in range(seqlen):
        #pos_array[4*pos:(4*pos + 4)] = pos
        ref_slice = test_seq[:, pos].copy()
        # index of reference base
        if sum(ref_slice == 1) == 1:
            ref[4*pos:(4*pos + 4)] = np.where(ref_slice == 1)[0].item()
        else:
            ref[4*pos:(4*pos + 4)] = nan
        test_seq[:, pos] = np.array([0., 0., 0., 0.])
        # generate each mutation at given loc
        for i in range(4):
            sl = np.array([0., 0., 0., 0.])
            sl[i] = 1.
            test_seq[:, pos] = sl
            X[4*pos + i, :, :] = test_seq

        # reset back to reference
        test_seq[:, pos] = ref_slice
    return (X, y, ref, mut, pos_array)



# can do single tissue or two tissue selection
def max_effect(res_head, 
               res_testis, 
               X, 
               direction):
    ''' direction [1 0] = increase head, do nothing testis
        direction [1 -1] = increase head, decrease testis
        etc.. '''
    dx = direction[0]*sp.logit(res_head) + direction[1]*sp.logit(res_testis)
    i = dx.numpy().argmax()
    return (X[i].numpy(), i)

def write_header(file):
    file.write('pos' + ',' + 
               'generation' + ',' +
               'ref_b' + ',' + 
                'mut_b' + ',' + 
                'head_out' + ',' + 
                'testis_out' + ',' + 
                'sample_ind' + '\n')

def write2file(res, file):
    res = [str(i) for i in res]
    res_str = ','.join(res) + '\n'
    file.write(res_str)

def SSWM_sample(sample_ind, 
                X_test, 
                base_head, # base model output values
                base_testis,
                direction, 
                file):
    
    sample = X_test[sample_ind].copy()
    baseline = [-1, # pos
                0, # generation
                -1, # refb
                -1, # mut b
                base_head[sample_ind], 
                base_testis[sample_ind], 
                sample_ind]
    write2file(baseline, file)
    for i in range(30):
        X, y, ref, mut, pos_array = get_seqdata_mut(sample)
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        test_loader = get_loader(X, y, device)
        result_raw = evaluate_raw(model, test_loader)
        res_head = torch.cat([r['head_out'] for r in result_raw])
        res_testis = torch.cat([r['testis_out'] for r in result_raw])
   
    
        sample, ind = max_effect(res_head.cpu(), res_testis.cpu(), X, direction)
        
        out = [pos_array[ind], #position
               i + 1, #generation
               ref[ind], # ref base
               mut[ind], # mutated base
               res_head.cpu().numpy()[ind], 
               res_testis.cpu().numpy()[ind], 
               sample_ind]
        write2file(out, file)


path= '../insilico/SSWM_OriginallyPeaks_' + str(direction[0]) + '_' + str(direction[1]) + '.csv'
device = get_default_device()
model = load_model('2tis_2057')

X_test, y_test = load_testdata('2tis_2057')
X = torch.tensor(X_test.copy(), dtype=torch.float)
y = torch.tensor(y_test.copy(), dtype=torch.float)
test_loader = get_loader(X, y, device)
result_raw = evaluate_raw(model, test_loader)
res_head = torch.cat([r['head_out'] for r in result_raw]).cpu().numpy()
res_testis = torch.cat([r['testis_out'] for r in result_raw]).cpu().numpy()
with open(path, 'w') as file:
    write_header(file)

with open(path, 'a') as file:
    for i in samples:
        SSWM_sample(i, X_test, res_head, res_testis, direction, file)
