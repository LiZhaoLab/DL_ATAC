#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:34:20 2021

@author: skhodursky
various functions needed for neural network training and testing
"""
import torch
import numpy as np
from neural_network import NeuralNetwork
import sklearn.metrics as metrics
import pandas as pd


def one_hot(seq):
    encode = np.zeros([4, len(seq)], dtype=np.float32)
    for i, nuc in enumerate(seq):
        if nuc == 'a':
            encode[0, i] = 1
        elif nuc == 't':
            encode[1, i] = 1
        elif nuc == 'c':
            encode[2, i] = 1
        elif nuc == 'g':
            encode[3, i] = 1
    return encode


def rev_comp(x):
    """rev complement for one hot encoded data"""
    # flip order
    out = np.zeros([x.shape[0], x.shape[1], x.shape[2]])
    flipped = np.flip(x, 2)
    # switch A to T and G to C
    out[:, 0, :] = flipped[:, 1, :]
    out[:, 1, :] = flipped[:, 0, :]
    out[:, 2, :] = flipped[:, 3, :]
    out[:, 3, :] = flipped[:, 2, :]
    return out

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def evaluate(model, val_loader):
    """Evaluate the model's performance can be used
    for both validation and testing """
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def evaluate_raw(model, val_loader):
    """Raw model output"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return outputs


def fit(epochs, lr, weight_decay, momentum, model,
        train_loader, val_loader, file_suffix):

    """Train the model using gradient descent"""
    opt_func = torch.optim.SGD
    optimizer = opt_func(model.parameters(), lr, 
                         weight_decay = weight_decay, momentum = momentum)
    result = float('inf')
    counter = 0
    #result = 0
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        model.eval()  # switch model state to evaluation
        with torch.no_grad():
            result_epoch = evaluate(model, val_loader)["loss"]
            counter += 1
            if result_epoch < result:
                result = result_epoch
                torch.save(model.state_dict(), "../saved_models/model" + file_suffix)
                counter = 0
        # if more than 5 epochs go by without improvement terminate training
        if counter > 10:  # changed from 2
            break
        model.train()  # switch model state back to training
            
    return result # changed from result


def load_model(suffix):
    params = np.load('../saved_models/params_2tis_2057.npy', allow_pickle='TRUE').item()
    model = NeuralNetwork(4,
                          params["h"],
                          params["f"],
                          2,
                          params["fcs"],
                          params["p"],
                          params["mha_p"])

    device = get_default_device()
    to_device(model, device)
    model.load_state_dict(torch.load("../saved_models/model_" + suffix))
    model.eval()
    return model


def load_testdata(suffix):
    """load data for testing"""
    X_test = np.load('../model_indat/X_test_' + suffix + '.npy')
    y_test = np.load('../model_indat/y_test_'+ suffix + '.npy')
    return X_test, y_test


def get_aucs(outputs, tissue, from_torch=False, idx_tokeep=None):
    '''
    idx_tokeep are indices to keep after filtering 
    for proximity to peak example or whatever you want
    '''
    if from_torch:
        tot_output = torch.cat([x[tissue + '_out'] for x in outputs]).detach().cpu()
        tot_y = torch.cat([x[tissue + '_y'] for x in outputs]).detach().cpu()
    if not from_torch:
        """
        assume that outputs is a df with columns out and label
        """
        tot_output = outputs.out
        tot_y = outputs.label

    if idx_tokeep is not None:
        tot_output = tot_output[idx_tokeep]
        tot_y = tot_y[idx_tokeep]
    
    fpr, tpr, thresholds = metrics.roc_curve(tot_y, tot_output)
    pr, re, thresholds = metrics.precision_recall_curve(tot_y, tot_output)
    roc_auc = metrics.auc(fpr, tpr)
    pr_auc = metrics.auc(re, pr)
    return (fpr, tpr, roc_auc, pr, re, pr_auc)

def format_4AUCplot(outputs, tissue, idx_dict=None, from_torch=False):
    """
    outputs is a dictionary with
    entry for every species. Idx_dict contains a dictionary
    of indices to keep after filtering test set.


    """

    species = ['D. mel', 'D. sim', 'D. yak']
    result_roc = []
    result_pr = []
    for sp in species:
        print(sp)
        if idx_dict is None:
            fpr, tpr, roc_auc, pr, re, pr_auc = get_aucs(outputs[sp], tissue,
                                                         from_torch=from_torch)
        else:
            fpr, tpr, roc_auc, pr, re, pr_auc = get_aucs(outputs[sp], tissue,
                                                         from_torch=from_torch,  idx_tokeep=idx_dict[sp])
        roc_auc = round(roc_auc, 3)
        pr_auc = round(pr_auc, 3)
        print(roc_auc)
        result_roc.append(pd.DataFrame({'True positive rate': tpr,
                                        'False positive rate': fpr,
                                        'ROC_AUC': roc_auc,
                                        'species': sp + ': ' + str(roc_auc)}))

        result_pr.append(pd.DataFrame({'Precision': pr,
                                       'Recall': re,
                                       'PR_AUC': pr_auc,
                                       'species': sp + ': ' + str(pr_auc)}))
    return pd.concat(result_roc), pd.concat(result_pr)

