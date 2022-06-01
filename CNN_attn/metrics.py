#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:12:24 2021

@author: skhodursky
module to store AUC and similar metrics
"""
import sklearn.metrics as metrics


def auc(outputs, actual):
    auc = metrics.roc_auc_score(actual, outputs)
    return auc
