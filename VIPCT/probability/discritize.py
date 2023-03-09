# This file contains auxiliary routines for VIP-CT v2 evaluation.
# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite our paper described in the readme file:
# Roi Ronen and Yoav. Y. Schechner.
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# author. The author is not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F

def to_onehot(gt, min, max, bins):
    # one_hot = torch.zeros((gt.shape[0],bins),device=gt.device)
    delta = (max - min)/(bins-1)
    index = torch.round(gt / delta).long()
    index[index>(bins-1)] = bins-1
    one_hot = F.one_hot(index, bins).float()
    return one_hot

def to_discrete(gt, min, max, bins):
    # one_hot = torch.zeros((gt.shape[0],bins),device=gt.device)
    delta = (max - min)/(bins-1)
    index = torch.round(gt / delta).long()
    index[index>(bins-1)] = bins-1
    return index

def get_pred_and_conf_from_discrete(discrete_preds, min, max, bins, pred_type='max'):
    preds = []
    confidences = []
    for discrete_pred in discrete_preds:
        if pred_type == 'max':
            i_max = discrete_pred.argmax(-1)
            scores = torch.exp(discrete_pred)
            max_score = scores.gather(-1,i_max[...,None]).squeeze()
            confidence = max_score / scores.sum(-1)
            delta = (max - min) / (bins - 1)
            pred = i_max * delta + min
        elif pred_type == 'mean':
            bin_values = torch.linspace(min,max,bins,device=discrete_pred.device)
            prob = torch.exp(discrete_pred) / torch.exp(discrete_pred).sum(-1)[...,None]
            # prob[:,0] /= 100
            weighted_bins = prob * bin_values
            pred = weighted_bins.sum(-1) # discrete_pred.sum(-1) should be 1
            # confidence = ((prob*(bin_values.repeat(pred.shape[0], 1) - pred[...,None])**2).sum(-1) / (prob.sum(-1)-1))**0.5
            confidence = ((prob*(bin_values.repeat(pred.shape[0], 1) - pred[...,None])**2).sum(-1) )**0.5
            # pred = torch.vstack((pred,pred_std)).T
        else:
            NotImplementedError()
        preds.append(pred)
        confidences.append(confidence)

    return preds, confidences




def get_pred_from_discrete(discrete_preds, min, max, bins, pred_type='max'):
    preds = []
    for discrete_pred in discrete_preds:
        if pred_type == 'max':
            i_max = discrete_pred.argmax(-1)
            delta = (max - min) / (bins - 1)
            pred = i_max * delta + min
        elif pred_type == 'mean':
            bin_values = torch.linspace(min,max,bins,device=discrete_pred.device)
            prob = torch.exp(discrete_pred)
            weighted_bins = prob * bin_values
            pred = weighted_bins.sum(-1) / prob.sum(-1) # discrete_pred.sum(-1) should be 1
        else:
            NotImplementedError()
        preds.append(pred)

    return preds