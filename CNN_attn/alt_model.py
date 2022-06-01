# various functions by/from Sam for model stuff
import torch
import torch.nn as nn
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
from utils import fit
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

LR = 5e-4

def load_model(suffix,
              param_folder = Path('/lustre/fs4/zhao_lab/scratch/lzhao/Khodursky/atac_DL_2t_attention/saved_models'),
              param_suffix = '2tis_2057',
              model_folder = Path('/lustre/fs4/zhao_lab/scratch/lzhao/Khodursky/atac_DL_2t_attention/saved_models')
              ):

    params = np.load(param_folder / f'params_{param_suffix}.npy', allow_pickle='TRUE').item()
    model = NeuralNetwork(4,
                          params["h"],
                          params["f"],
                          2,
                          params["fcs"],
                          params["p"],
                          params["mha_p"])

    device = get_default_device()
    to_device(model, device)
    model.load_state_dict(torch.load(model_folder / f"model_{suffix}", map_location=torch.device('cpu')))
    model.eval()
    return model

def retrain_model(X_data, y_data, oldsuffix, batch_size, seed=10, stratify=None,
                 param_folder = Path('/lustre/fs4/zhao_lab/scratch/lzhao/Khodursky/atac_DL_2t_attention/saved_models'),
                 param_suffix = '2tis_2057'):
    newsuffix = '_'+oldsuffix+'_retrained'
    params = np.load(param_folder / f'params_{param_suffix}.npy', allow_pickle='TRUE').item()
    model = load_model(oldsuffix)
    device = get_default_device()
    to_device(model, device)

    if not isinstance(X_data, torch.Tensor):
        X_data = torch.cat([torch.tensor(X_data, dtype=torch.float32)])
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.cat([torch.tensor(y_data, dtype=torch.float32)])

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=seed, stratify=stratify)
    train_loader = get_loader(X_train, y_train, device, batch_size, shuffle=True)
    val_loader = get_loader(X_val, y_val, device, batch_size)
    model.train()
    fit(1000, LR/10, params["wd"], params["mo"], model, train_loader, val_loader, newsuffix)
    return load_model(newsuffix[1:], model_folder = Path('../saved_models')) # strip the leading '_'

def train_model(X_data, y_data, suffix, batch_size=32, seed=10, stratify=None,
              param_folder = Path('/lustre/fs4/zhao_lab/scratch/lzhao/Khodursky/atac_DL_2t_attention/saved_models'),
              param_suffix = '2tis_2057'):
    params = np.load(param_folder / f'params_{param_suffix}.npy', allow_pickle='TRUE').item()
    model = NeuralNetwork(4,
                          params["h"],
                          params["f"],
                          2,
                          params["fcs"],
                          params["p"],
                          params["mha_p"])
    device = get_default_device()
    to_device(model, device)

    if not isinstance(X_data, torch.Tensor):
        X_data = torch.cat([torch.tensor(X_data, dtype=torch.float32)])
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.cat([torch.tensor(y_data, dtype=torch.float32)])

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=seed, stratify=stratify)
    train_loader = get_loader(X_train, y_train, device, batch_size, shuffle=True)
    val_loader = get_loader(X_val, y_val, device, batch_size)
    model.train()
    fit(1000, LR, params["wd"], params["mo"], model, train_loader, val_loader, suffix)
    return load_model(suffix[1:], model_folder = Path('../saved_models')) # strip the leading '_'

def get_loader(X_test, y_test, device, batch_size=200, shuffle=False):
    
    #train_data = CustomData(X_train, y_train)
    test_data = CustomData(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    test_loader = DeviceDataLoader(test_loader, device)
    return test_loader

def get_data_roc(model, test_loader):
    result_raw = evaluate_raw(model, test_loader)
    tot_output = torch.cat([x['output'] for x in result_raw])
    tot_y = torch.cat([x['y'] for x in result_raw])
    fpr, tpr, thresholds = metrics.roc_curve(tot_y.detach().cpu(), tot_output.detach().cpu())
    pr, re, thresholds = metrics.precision_recall_curve(tot_y.detach().cpu(), tot_output.detach().cpu())
    roc_auc = metrics.auc(fpr, tpr)
    pr_auc = metrics.auc(re, pr)
    return (fpr, tpr, roc_auc, pr, re, pr_auc)

def get_aucs(outputs, tissue):
    
    tot_output = torch.cat([x[tissue + '_out'] for x in outputs])
    tot_y = torch.cat([x[tissue + '_y'] for x in outputs])
    #tot_outputtestis = torch.cat([x['testis_out'] for x in outputs])
    #tot_ytestis = torch.cat([x['testis_y'] for x in outputs])
     
    #tot_output = torch.cat([x['output'] for x in result_raw])
    #tot_y = torch.cat([x['y'] for x in result_raw])
    fpr, tpr, thresholds = metrics.roc_curve(tot_y.detach().cpu(), tot_output.detach().cpu())
    pr, re, thresholds = metrics.precision_recall_curve(tot_y.detach().cpu(), tot_output.detach().cpu())
    roc_auc = metrics.auc(fpr, tpr)
    pr_auc = metrics.auc(re, pr)
    return (roc_auc, pr_auc)

def get_aucs_bt(model, test_loader):
    result_raw = evaluate_raw(model, test_loader)
    roc_head, pr_head, thres_head = get_aucs(result_raw, "head")
    roc_testis, pr_testis, thres_testis = get_aucs(result_raw, "testis")
    return (roc_head, pr_head, thres_head, roc_testis, pr_testis, thres_testis)

def get_auc_ratios(model1, model2, loader):
    # model 1 is trained in a different species
    # model 2 is trained in same species as loader
    roc_head1, pr_head1, roc_testis1, pr_testis1 = get_aucs_bt(model1, loader)
    roc_head2, pr_head2, roc_testis2, pr_testis2 = get_aucs_bt(model2, loader)
    
    return(roc_head1/roc_head2, 
           pr_head1/pr_head2, 
           roc_testis1/roc_testis2, 
           pr_testis1/pr_testis2)

#suffix = '2000_peakctr_w501'
def get_test_results(X_test,y_test, model):    
    test_data = CustomData(X_test, y_test)
    device = get_default_device()
    test_loader = get_loader(X_test, y_test, device)
    return get_aucs_bt(model, test_loader)

def parse_bulk_results(result, labels=False):
    '''given a raw pytorch result, returns the model outputs, nicely-formatted.
    output: head, testis.
    if labels=True:
      output is (head output, head labels), (testis output, testis labels)'''
    head_out = []
    testis_out = []
    
    for i in result:
        head_out.append(pd.Series(i['head_out'].cpu()))
        testis_out.append(pd.Series(i['testis_out'].cpu()))
    
    if labels:
        head_y = []
        testis_y = []
        for i in result:
            head_y.append(pd.Series(i['head_y'].cpu()))
            testis_y.append(pd.Series(i['testis_y'].cpu()))
            
        return (pd.concat(head_out), pd.concat(head_y)), (pd.concat(testis_out), pd.concat(testis_y))
        
    return pd.concat(head_out), pd.concat(testis_out)

def melt_paired(data):
    '''ASSUMING paired data w no rearranging, unstacks that'''
  
    df = pd.DataFrame([data[0].reset_index(drop=True), data[1].reset_index(drop=True)],
                      index=['output', 'label']).T
    cutpoint = int((df.shape[0]+1)/2) # assumes pairs aka equal length
    
    # more assumptions
    assert (df[cutpoint:]['label'] == 0).all()
    assert (df[:cutpoint]['label'] == 1).all()
    
    cleaned = pd.DataFrame({'neg':df['output'].iloc[:cutpoint].values,
                            'pos':df['output'].iloc[cutpoint:].values})
    cleaned = cleaned.assign(delta=lambda df: df.pos - df.neg)
    
    return cleaned

def inference(X_data, y_data, model, batch_size=200):
    '''higher-level function to complement the other sam functions'''
    if not isinstance(X_data, torch.Tensor):
        X_data = torch.cat([torch.tensor(X_data)])
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.cat([torch.tensor(y_data)])

    device = get_default_device()
    loader = get_loader(X_data, y_data, device, batch_size=batch_size)
    return evaluate_raw(model, loader)
