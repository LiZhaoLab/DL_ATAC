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
from math import nan


def load_testdata(suffix):
    X_test = np.load('../model_indat/X_test_' + suffix + '.npy')

    y_test = np.load('../model_indat/y_test_'+ suffix + '.npy')
    return (X_test, y_test)


# make data sized (N, C, L) where L is the length of sequence, C is the 4 channels
# and N is the number of datapoints
def get_seqdata_mut(test_seq):
    test_seq = test_seq.copy()
    seqlen = test_seq.shape[1]
    X = np.zeros([4*seqlen, 4, seqlen], dtype=np.float32)
    y = np.zeros([4*seqlen, 2])
    ref = np.zeros([4*seqlen])
    pos_array = np.zeros([4*seqlen])
    # scan over entire sequence and mutate
    # a t c g is the code
    for pos in range(seqlen):
        pos_array[4*pos:(4*pos + 4)] = pos
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
    return (X, y, ref, pos_array)
#

def get_loader(X_test, y_test, device):
    batch_size = 128
    # train_data = CustomData(X_train, y_train)
    test_data = CustomData(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)
    test_loader = DeviceDataLoader(test_loader, device)
    return test_loader


# functions for writing outputs to file
def write_header(file):
    file.write('a' + ',' +
               't' + ',' +
               'c' + ',' +
               'g' + ',' +
               'ref' + ',' +
               'pos' + ',' +
               'sample_ind' + '\n')
# res contains the test results


def write2file(res, ref_i, pos_a, sample_i, file, seqlen):
    for j in range(seqlen):
            file.write(str(res[4*j].item()) + ',' +
                              str(res[4*j + 1].item()) + ',' +
                              str(res[4*j + 2].item()) + ',' +
                              str(res[4*j + 3].item()) + ',' +
                              str(ref_i[4*j]) + ',' +
                              str(pos_a[4*j]) + ',' +
                              str(sample_i)+ '\n')


if __name__ == "__main__":

    X_test, y_test = load_testdata('2tis_2057')
    device = get_default_device()
    model = load_model('2tis_2057')

    # ref is reference base (index)
    # pos is position within the example
    # sample ind is index of sample
    path_head = '../insilico/insilico_headmel.csv'
    path_testis = '../insilico/insilico_testismel.csv'
    with open(path_head, 'w') as file_head, open(path_testis, 'w') as file_testis:
        write_header(file_head)
        write_header(file_testis)
    with open(path_head, 'a') as file_head, open(path_testis, 'a') as file_testis:
        for sample_ind in range(X_test.shape[0]):
            sample = X_test[sample_ind, :, :].copy()
            X, y, ref_ind, pos_array = get_seqdata_mut(sample)
            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            test_loader = get_loader(X, y, device)
            result_raw = evaluate_raw(model, test_loader)
            res_head = torch.cat([r['head_out'] for r in result_raw])
            res_testis = torch.cat([r['testis_out'] for r in result_raw])
            # write results to file
            write2file(res_head, ref_ind, pos_array, sample_ind, file_head, seqlen=1000)
            write2file(res_testis, ref_ind, pos_array, sample_ind, file_testis, seqlen=1000)

