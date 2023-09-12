from typing_extensions import final
import torch
from torch._C import ThroughputBenchmark
import torch.nn.functional as F
import math 
import copy
   
LOG_EPSILON = 1e-5

'''
helper functions
'''

def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def log_loss(preds, targs):
    return targs * neg_log(preds)

def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos)**2
    else:
        raise NotImplementedError
    return reg
    

def loss_an(preds,label_vec_obs, P):
    # unpack:
    observed_labels = label_vec_obs
    # input validation: 
    # assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx


def loss_spbc(preds,label_vec_obs, P):
    # unpack:
    observed_labels = label_vec_obs
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    
    loss_mtx = neg_log(preds[observed_labels == 1])

    return loss_mtx


def batch_loss_spbc_ran_l2(logits, logits_mask, logits_s, label_vec, P, gap_thres, unannotated_dist, epoch): # "preds" are actually logits (not sigmoid activated !)
    
    ############################################################################
    
    preds = torch.sigmoid(logits)
    preds_s = torch.sigmoid(logits_s)

    gap = torch.abs(logits - logits_mask).detach()
    pseudo = ((gap > gap_thres) & (logits > unannotated_dist)).type(torch.float32).detach()
    
    # warm up phase
    if epoch<=P["warm_up"]:
        mask = torch.zeros_like(label_vec)
    else:
        mask = (pseudo - label_vec).clip(max=1.0, min=0.0)

    fn_mask = (torch.ones_like(mask) - mask).detach()


    loss_ran_mtx = loss_an(preds_s, label_vec, P)
    loss_spbc_mtx= loss_spbc(preds, label_vec , P)


    SPBC = loss_spbc_mtx.mean()
    RAN = loss_ran_mtx[fn_mask.type(torch.bool)].mean()
    consistency_l2 = F.mse_loss(logits_s, logits)

    main_loss = P["spbc_weight"]*SPBC + P["ran_weight"]*RAN + P["l2_weight"]*consistency_l2 

    return main_loss

 