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

def get_pred_and_conf_from_discrete(discrete_preds, min, max, bins, pred_type='max', conf_type='prob'):
    preds = []
    probs = []
    confidences = []
    for discrete_pred in discrete_preds:
        # i_max = discrete_pred.argmax(-1)
        prob = torch.exp(discrete_pred) / torch.exp(discrete_pred).sum(-1)[..., None]
        i_max = prob.argmax(-1)
        bin_values = torch.linspace(min, max, bins, device=discrete_pred.device)
        delta = (max - min) / (bins - 1)

        if pred_type == 'max':
            # scores = torch.exp(discrete_pred)
            # confidence = max_score / scores.sum(-1)
            delta = (max - min) / (bins - 1)
            pred = i_max * delta + min
        elif pred_type == 'max_conv':
            # scores = torch.exp(discrete_pred)
            # confidence = max_score / scores.sum(-1)
            prob_conv = prob[:,1:][:,None]
            prob_conv = torch.nn.functional.pad(prob_conv,(1,1),mode='replicate')
            filter = torch.tensor(([0.333,0.333,0.333]),device=prob.device).reshape(1,1,-1)
            prob_conv1 = torch.nn.functional.conv1d(prob_conv,filter).squeeze(1)
            prob[:, 1:] = prob_conv1
            i_max = prob.argmax(-1)
            delta = (max - min) / (bins - 1)
            pred = i_max * delta + min
        elif pred_type == 'mean':
            # prob[:,0] /= 100
            weighted_bins = prob * bin_values
            pred = weighted_bins.sum(-1) # discrete_pred.sum(-1) should be 1
            # confidence = ((prob*(bin_values.repeat(pred.shape[0], 1) - pred[...,None])**2).sum(-1) / (prob.sum(-1)-1))**0.5
            # pred = torch.vstack((pred,pred_std)).T
        elif pred_type == 'differentiable_max':
            # prob[:,0] /= 100
            weights = prob**10
            weights /= weights.sum(-1)[...,None]
            weighted_bins = weights * bin_values
            pred = weighted_bins.sum(-1) * delta # discrete_pred.sum(-1) should be 1
            pred[torch.isnan(pred)] = 0
        elif pred_type == 'differentiable_max_correct':
            # prob[:,0] /= 100
            weights = prob**10
            weights /= weights.sum(-1)[...,None]
            weighted_bins = weights * bin_values
            pred = weighted_bins.sum(-1)  # discrete_pred.sum(-1) should be 1
            pred[torch.isnan(pred)] = 0
        else:
            NotImplementedError()
        if conf_type=='prob':
            confidence = prob.gather(-1,i_max[...,None]).squeeze()
        elif conf_type =='rel_prob':
            prob2 = prob.clone()
            prob2[i_max>0,0]=0
            prob_values, indices = torch.topk(prob2,2,dim=1)
            ext_bin_values = bin_values[indices] / bin_values[-1]
            confidence = torch.squeeze(torch.diff(prob_values,1)**2 + (torch.diff(ext_bin_values,1))**2)
        elif conf_type =='std':
            weighted_bins = prob * bin_values
            avg = weighted_bins.sum(-1)
            std = ((prob*(bin_values.repeat(avg.shape[0], 1) - avg[...,None])**2).sum(-1) )**0.5
            confidence = (prob.shape[1] - std*2) / prob.shape[1]
        elif conf_type =='conf_interval':
            conf_interval = torch.zeros_like(pred)
            # cum_prob = prob.gather(-1, i_max[...,None]).squeeze()

            arange1 = torch.arange(prob.shape[-1],device=i_max.device).view((1, -1)).repeat((prob.shape[0],1))
            arange1r = torch.arange(prob.shape[-1],0,step=-1,device=i_max.device).view((1, -1)).repeat((prob.shape[0],1))
            arange2 = (arange1 - (prob.shape[-1] -  i_max[...,None])) % prob.shape[-1]
            arange2r = (arange1r - (prob.shape[-1] -  i_max[...,None])) % prob.shape[-1]



            mask_ind = torch.where(arange2<i_max[...,None],True,False)
            arange3r = arange2.clone()
            arange3r[mask_ind==False] = 0
            arange3r = arange3r.fliplr()

            # mask_arange2 = (arange1 - i_max[...,None])
            # mask_arange2 = torch.sign(mask_arange2[:,0]+0.1)[...,None] != torch.sign(mask_arange2+0.1)
            #
            # mask_arange2r = (arange1r - i_max[...,None])
            # mask_arange2r = torch.sign(mask_arange2r[:, 0] + 0.1)[..., None] != torch.sign(mask_arange2r + 0.1)
            # prob_nan = torch.hstack((prob,torch.full(i_max.shape,torch.nan,device=prob.device)[...,None]))

            prob_rearange1 = prob.gather(-1, arange2).squeeze()
            prob_rearange2 = prob.gather(-1, arange3r).squeeze()
            prob_rearange1[mask_ind] = 0
            prob_rearange2[mask_ind.fliplr()==False] = 0
            cumsum_prob1 = prob_rearange1.cumsum(-1)
            cumsum_prob2 = prob_rearange2.cumsum(-1)
            cumsum_prob_total = cumsum_prob1
            cumsum_prob_total[:,1:] += cumsum_prob2[:,:-1]
            alpha = 0.7
            x = cumsum_prob_total - alpha
            conf_interval = (x <= 0).sum(dim=1).float() * 2.0
            confidence = (cumsum_prob_total.shape[-1] - conf_interval) / cumsum_prob_total.shape[-1]
            # confidence = conf_interval.float()


            # for bin in range(1, prob.shape[-1]):
            #     if i_max>bin:
            #         cum_prob += prob_rearange1[:,1]
            #     if i_max
            #
            #
            #
            # cum_prob1 = prob_rearange1.cumsum(1)
            # # prob_rearange1[mask_arange2] = torch.nan
            # # prob_rearange2[mask_arange2r] = torch.nan
            #
            # cum_prob_around_imax = prob_rearange1
            # cum_prob = torch.cumsum(prob,-1) - prob.gather(-1, i_max[...,None])
            # cum_prob[cum_prob<0] = 0
            # A= torch.roll(cum_prob,i_max,dims=0)
            # alpha = 0.95
            # keep = torch.ones(prob.shape[0],dtype=bool,device=prob.device)
            # for bin in range(1,prob.shape[-1]):
            #     keep = keep * (cum_prob<alpha)
            #     print(torch.sum(keep))
            #     if torch.sum(keep)==0:
            #         break
            #     curr_prob = prob[keep]
            #     curr_i_max = i_max[keep,None]
            #     i_before = curr_i_max - bin
            #     prob_before = torch.zeros((curr_prob.shape[0],1),dtype=curr_prob.dtype,device=curr_prob.device)
            #     prob_before[i_before>=0] = curr_prob.gather(-1,i_before[i_before>=0][...,None]).squeeze()
            #     i_after= curr_i_max + bin
            #     prob_after = torch.zeros((curr_prob.shape[0],1),dtype=curr_prob.dtype,device=curr_prob.device)
            #     prob_after[i_after < bins] = curr_prob.gather(-1, i_after[i_after < bins][..., None]).squeeze()
            #     cum_prob[keep] += (prob_before + prob_after).squeeze()
            #     conf_interval[keep] +=1
            # for bin in range(1,prob.shape[-1]):
            #     keep = keep * (cum_prob<alpha)
            #     print(torch.sum(keep))
            #     if torch.sum(keep)==0:
            #         break
            #     curr_prob = prob[keep]
            #     curr_i_max = i_max[keep,None]
            #     i_before = curr_i_max - bin
            #     prob_before = torch.zeros((curr_prob.shape[0],1),dtype=curr_prob.dtype,device=curr_prob.device)
            #     prob_before[i_before>=0] = curr_prob.gather(-1,i_before[i_before>=0][...,None]).squeeze()
            #     i_after= curr_i_max + bin
            #     prob_after = torch.zeros((curr_prob.shape[0],1),dtype=curr_prob.dtype,device=curr_prob.device)
            #     prob_after[i_after < bins] = curr_prob.gather(-1, i_after[i_after < bins][..., None]).squeeze()
            #     cum_prob[keep] += (prob_before + prob_after).squeeze()
            #     conf_interval[keep] +=1
            # print()






        preds.append(pred)
        probs.append(prob)
        confidences.append(confidence)

    return preds, confidences, probs




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


def roll_by_gather(mat, dim, shifts: torch.LongTensor, forward=True):
    # assumes 2D array
    n_rows, n_cols = mat.shape

    if dim == 0:
        # print(mat)
        arange1 = torch.arange(n_rows,device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
        if forward:
            arange2 = (arange1 - shifts) % n_rows
        else:
            arange2 = (arange1 - (n_rows-shifts)) % n_rows
            # print(arange1)
        # print(arange2)
        return torch.gather(mat, 0, arange2)
    elif dim == 1:
        arange1 = torch.arange(n_cols,device=mat.device).view((1, n_cols)).repeat((n_rows, 1))
        # print(arange1)
        arange2 = (arange1 - shifts) % n_cols
        if forward:
            arange2 = (arange1 - shifts) % n_cols
        else:
            arange2 = (arange1 - (n_rows-shifts)) % n_rows
        # print(arange2)
        return torch.gather(mat, 1, arange2)