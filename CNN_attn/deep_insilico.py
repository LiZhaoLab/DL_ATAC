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
import sklearn.metrics as metrics
import random
from math import nan


def load_testdata(suffix):
    X_test = np.load('../model_indat/X_test_' + suffix + '.npy')

    y_test = np.load('../model_indat/y_test_'+ suffix + '.npy')
    return (X_test, y_test)


def fill_nas(array):
    nas = np.where(np.sum(array[:, :], axis=0) == 0)[0]
    newhots = np.random.randint(0, 4, size=nas.shape[0])

    array[newhots, nas] = 1
    return array


def mutate_no_replacement(array, n):
    '''mutates the array N times w/o replacement'''
    # array = array.copy()
    if array.sum() != 1000:
        # np.nonzero behaves bizarrely if there are Ns
        array = fill_nas(array)

    idxs = random.sample(range(array.shape[1]), n)
    oldhots = np.nonzero(array[:, idxs].transpose())[1]
    # transpose so that the sorting from np.nonzero is the way we want
    newhots = (oldhots + np.random.randint(1, 4, size=n)) % 4

    array[:, idxs] = 0
    array[newhots, idxs] = 1

    return array

def mutate_example(example, y, nmut, nexp):
    ''' nmut is number of mutations
    and nexp is number of experiments'''
    X_deepmut = np.repeat(example[np.newaxis,...], nexp+1, axis=0)
    y_deepmut = np.repeat(y[np.newaxis,...], nexp+1, axis=0)
    for i in X_deepmut[1:]:
        mutate_no_replacement(i, nmut)

    return (X_deepmut, y_deepmut)


def get_loader(X_test, y_test, device):
    batch_size = 128
    # train_data = CustomData(X_train, y_train)
    test_data = CustomData(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)
    test_loader = DeviceDataLoader(test_loader, device)
    return test_loader

def write_header(file):
    file.write('out' + ',' +
               'nmut' + ',' +
               'sample_ind' + ',' +
                'exp_num' + '\n')
# res contains the test results
def write2file(res, file, nmut, sample_ind):
    for j in range(1001):
            file.write(str(res[j].item()) + ',' +
                              str(nmut) + ',' +
                              str(sample_ind) + ',' +
                              str(j) + '\n')


if __name__ == "__main__":
    X_test, y_test = load_testdata('2tis_2057')
    device = get_default_device()
    model = load_model('2tis_2057')
    # sample ind is index of sample
    path_head = '../insilico/deep_insilico_headmel.csv'
    path_testis = '../insilico/deep_insilico_testismel.csv'
    with open(path_head, 'w') as file_head, open(path_testis, 'w') as file_testis:
        write_header(file_head)
        write_header(file_testis)
    with open(path_head, 'a') as file_head, open(path_testis, 'a') as file_testis:
        # X_test.shape[0]
        for sample_ind in range(X_test.shape[0]):
            for nmut in [50, 100, 200, 500, 1000]:
                sample = X_test[sample_ind, :, :]
                y = y_test[sample_ind]
                X, y = mutate_example(sample, y, nmut, 1000)
                X = torch.tensor(X, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                test_loader = get_loader(X, y, device)
                result_raw = evaluate_raw(model, test_loader)
                res_head = torch.cat([r['head_out'] for r in result_raw])
                res_testis = torch.cat([r['testis_out'] for r in result_raw])
                # write results to file
                write2file(res_head, file_head, nmut, sample_ind)
                write2file(res_testis, file_testis, nmut, sample_ind)
