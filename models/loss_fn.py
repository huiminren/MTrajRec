#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/11/25 18:06

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from map_matching.candidate_point import CandidatePoint
from map_matching.utils import find_shortest_path
from common.spatial_func import SPoint, distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_dis_loss(predict, target, trg_len):
    """
    Calculate MAE and RMSE between predicted and targeted GPS sequence.
    Args:
    -----
        predict = [seq len, batch size, 2]
        target = [seq len, batch size, 2]
        trg_len = [batch size]  if not considering target length, the loss will smaller than the real one.
        predict and target have been removed sos 
    Returns:
    -------
        MAE of a batch in meter.
        RMSE of a batch in meter.
    """
    predict = predict.permute(1, 0, 2)  # [batch size, seq len, 2]
    target = target.permute(1, 0, 2)  # [batch size, seq len, 2]
    bs = predict.size(0)

    ls_dis = []
    for bs_i in range(bs):
        for len_i in range(trg_len[bs_i]-1):
            pre = SPoint(predict[bs_i, len_i][0], predict[bs_i, len_i][1])
            trg = SPoint(target[bs_i, len_i][0], target[bs_i, len_i][1])
            dis = distance(pre, trg)
            ls_dis.append(dis)

    ls_dis = np.array(ls_dis)
    mae = ls_dis.mean()
    rmse = np.sqrt((ls_dis**2).mean())
    return mae, rmse


def check_rn_dis_loss(predict_gps, predict_id, predict_rate, target_gps, target_id, target_rate, trg_len, 
                      rn, raw_rn_dict, new2raw_rid_dict):
    """
    Calculate road network based MAE and RMSE between predicted and targeted GPS sequence.
    Args:
    -----
        predict_gps = [seq len, batch size, 2]
        predict_id = [seq len, batch size, id one hot output dim]
        predict_rates = [seq len, batch size]
        target_gps = [seq len, batch size, 2]
        target_id = [seq len, batch size]
        target_rates = [seq len, batch size]
        trg_len = [batch size]  if not considering target length, the loss will smaller than the real one.
        
        predict and target have been removed sos 
    Returns:
    -------
        MAE of a batch in meter.
        RMSE of a batch in meter.
    """
    seq_len = target_id.size(0)
    batch_size = target_id.size(1)
    predict_gps = predict_gps.permute(1, 0, 2)
    predict_id = predict_id.permute(1, 0, 2)
    predict_rate = predict_rate.permute(1, 0)
    target_gps = target_gps.permute(1, 0, 2)
    target_id = target_id.permute(1, 0)
    target_rate = target_rate.permute(1, 0)
    
    ls_dis, rn_ls_dis = [], []
    for bs in range(batch_size):
        for len_i in range(trg_len[bs]-1): # don't calculate padding points
            pre_rid = predict_id[bs, len_i].argmax()
            convert_pre_rid = new2raw_rid_dict[pre_rid.tolist()]
            pre_rate = predict_rate[bs, len_i]
            pre_offset = raw_rn_dict[convert_pre_rid]['length'] * pre_rate
            pre_candi_pt = CandidatePoint(predict_gps[bs,len_i][0], predict_gps[bs,len_i][1], 
                                          convert_pre_rid, 0, pre_offset, pre_rate)
            
            trg_rid = target_id[bs, len_i]
            convert_trg_rid = new2raw_rid_dict[trg_rid.tolist()]
            trg_rate = target_rate[bs, len_i]
            trg_offset = raw_rn_dict[convert_trg_rid]['length'] * trg_rate
            trg_candi_pt = CandidatePoint(target_gps[bs,len_i][0], target_gps[bs,len_i][1], 
                                          convert_trg_rid, 0, trg_offset, trg_rate)
            
            if pre_candi_pt.lat == trg_candi_pt.lat and pre_candi_pt.lng == trg_candi_pt.lng:
                rn_dis = 0
                dis = 0
            else:
                rn_dis, _ = min(find_shortest_path(rn, pre_candi_pt, trg_candi_pt), 
                             find_shortest_path(rn, trg_candi_pt, pre_candi_pt))
                if type(rn_dis) is not float:
                    rn_dis = rn_dis.tolist()
                dis = distance(pre_candi_pt, trg_candi_pt)
                
            if rn_dis == np.inf:
                rn_dis = 1000
            rn_ls_dis.append(rn_dis)
            ls_dis.append(dis)
    
    ls_dis = np.array(ls_dis)
    rn_ls_dis = np.array(rn_ls_dis)
    
    mae = ls_dis.mean()
    rmse = np.sqrt((ls_dis**2).mean())
    rn_mae = rn_ls_dis.mean()
    rn_rmse = np.sqrt((rn_ls_dis**2).mean())
    return mae, rmse, rn_mae, rn_rmse


def shrink_seq(seq):
    """remove repeated ids"""
    s0 = seq[0]
    new_seq = [s0]
    for s in seq[1:]:
        if s == s0:
            continue
        else:
            new_seq.append(s)
        s0 = s
    
    return new_seq

def memoize(fn):
    '''Return a memoized version of the input function.

    The returned function caches the results of previous calls.
    Useful if a function call is expensive, and the function 
    is called repeatedly with the same arguments.
    '''
    cache = dict()
    def wrapped(*v):
        key = tuple(v) # tuples are hashable, and can be used as dict keys
        if key not in cache:
            cache[key] = fn(*v)
        return cache[key]
    return wrapped

def lcs(xs, ys):
    '''Return the longest subsequence common to xs and ys.

    Example
    >>> lcs("HUMAN", "CHIMPANZEE")
    ['H', 'M', 'A', 'N']
    '''
    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i-1], ys[j-1]
            if xe == ye:
                return lcs_(i-1, j-1) + [xe]
            else:
                return max(lcs_(i, j-1), lcs_(i-1, j), key=len)
        else:
            return []
    return lcs_(len(xs), len(ys))


def cal_id_acc(predict, target, trg_len):
    """
    Calculate RID accuracy between predicted and targeted RID sequence.
    1. no repeated rid for two consecutive road segments
    2. longest common subsequence
    http://wordaligned.org/articles/longest-common-subsequence
    Args:
    -----
        predict = [seq len, batch size, id one hot output dim]
        target = [seq len, batch size, 1]
        predict and target have been removed sos 
    Returns:
    -------
        mean matched RID accuracy.
    """
    predict = predict.permute(1, 0, 2)  # [batch size, seq len, id dim]
    target = target.permute(1, 0)  # [batch size, seq len, 1]
    bs = predict.size(0)

    correct_id_num = 0
    ttl_trg_id_num = 0
    ttl_pre_id_num = 0
    ttl = 0
    cnt = 0
    for bs_i in range(bs):
        pre_ids = []
        trg_ids = []
        # -1 because predict and target are removed sos.
        for len_i in range(trg_len[bs_i] - 1):
            pre_id = predict[bs_i][len_i].argmax()
            trg_id = target[bs_i][len_i]
            pre_ids.append(pre_id)
            trg_ids.append(trg_id)
            if pre_id == trg_id:
                cnt += 1
            ttl += 1

        # compute average rid accuracy
        shr_trg_ids = shrink_seq(trg_ids)
        shr_pre_ids = shrink_seq(pre_ids)
        correct_id_num += len(lcs(shr_trg_ids, shr_pre_ids))
        ttl_trg_id_num += len(shr_trg_ids)
        ttl_pre_id_num += len(shr_pre_ids)

    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num
    return rid_acc, rid_recall, rid_precision