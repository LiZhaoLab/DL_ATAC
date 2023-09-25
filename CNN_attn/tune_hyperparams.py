#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:45:08 2021

@author: skhodursky

main script to tune network hyperparameters
"""
import torch
import numpy as np
from loading_data import CustomData
from utils import get_default_device
from loading_data import DeviceDataLoader
from utils import fit
from utils import to_device
from torch.utils.data.dataloader import DataLoader
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from math import nan
from neural_network import NeuralNetwork
import joblib
import os
import sklearn.utils as skutils
from utils import rev_comp
# file names
file_suffix = '_2tis_2057'


# define  the optuna trial function


def objective(trial):
    # initialize model
    #nlayers_conv = trial.suggest_int("n", 2, 4)
    convl_size = trial.suggest_int("h", 10, 250, 10)
    filter_size = trial.suggest_int("f", 3, 15)

    pool_size = 2

    fc_size = trial.suggest_int("fcs", 10, 1000, 10)
    p = trial.suggest_float("p", 0.0, 0.999)
    mha_p = trial.suggest_float("mha_p", 0.0, 0.999)

    model = NeuralNetwork(4, convl_size, filter_size, pool_size, fc_size, p, mha_p)
    to_device(model, device)
    wd = trial.suggest_float("wd", 1e-10, 1e-1, log=True)
    mo = trial.suggest_float("mo", 0.0, 0.999)
    try:
        l = fit(10, 5e-4, wd, mo, model, train_loader, val_loader, file_suffix)
    except:
        l = nan
    return l


if __name__ == "__main__":


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    X_train = np.load('../model_indat/X_train' + file_suffix + '.npy')
    X_train_rev = rev_comp(X_train)
    X_train = np.concatenate([X_train, X_train_rev])

    y_train = np.load('../model_indat/y_train' + file_suffix + '.npy')
    y_train = np.concatenate([y_train, y_train])

    X_train, y_train = skutils.shuffle(X_train, y_train, random_state=2)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)

    X_val = np.load('../model_indat/X_val' + file_suffix + '.npy')
    X_val = torch.tensor(X_val, dtype=torch.float)

    y_val = np.load('../model_indat/y_val' + file_suffix + '.npy')
    y_val = torch.tensor(y_val, dtype=torch.float)

    train_data = CustomData(X_train, y_train)
    val_data = CustomData(X_val, y_val)
    # determine if device is CPU or GPU
    device = get_default_device()

    batch_size = 32  # this is the batch size
    train_loader = DataLoader(train_data, batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size,
                            num_workers=4, pin_memory=True)

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    torch.manual_seed(1)
    sampler = TPESampler(seed=10)  # random seed

    study = optuna.create_study(direction="minimize")  # minimize the BCE loss
    study.optimize(objective, n_trials=100)
    trial_ = study.best_trial
    np.save('../saved_models/params' + file_suffix + '.npy', trial_.params)
    print(trial_.values)

    # save optuna study
    joblib.dump(study, '../saved_models/study' + file_suffix + '.pkl')
