#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/10/20 18:24


import torch
import math
import numpy as np
import pandas as pd
import networkx as nx

from common.spatial_func import distance, cal_loc_along_line, SPoint
from map_matching.candidate_point import get_candidates, CandidatePoint

from utils.utils import load_json_data


#####################################################################################################
#
# Load Files
#
#####################################################################################################

def load_rid_freqs(dir, file_name):
    """
    load rid freqs and convert key from str to int
    """
    rid_freqs = load_json_data(dir, file_name)
    rid_freqs = {int(k): int(v) for k, v in rid_freqs.items()}  # convert key from str to int

    return rid_freqs


def load_rn_dict(dir, file_name):
    """
    This function will be use in rate2gps.
    """
    rn_dict = load_json_data(dir, file_name)
    new_rn_dict = {}
    for k, v in rn_dict.items():
        new_rn_dict[int(k)] = {}
        new_rn_dict[int(k)]['coords'] = [SPoint(coord[0], coord[1]) for coord in v['coords']]  
        # convert str to SPoint() to calculate distance 
        new_rn_dict[int(k)]['length'] = v['length']
        new_rn_dict[int(k)]['level'] = v['level']
    del rn_dict
    return new_rn_dict

def load_online_features(dir, file_name):
    """
    load POI or road network and covert key from str to tuple
    """
    data = load_json_data(dir, file_name)
    data = {}
    
    return data

#####################################################################################################
#
# RID + Rate 2 GPS
#
#####################################################################################################
def rate2gps(rn_dict, eid, rate, parameters):
    """
    Convert road rate to GPS on the road segment.
    Since one road contains several coordinates, iteratively computing length can be more accurate.
    Args:
    -----
    rn_dict:
        dictionary of road network
    eid,rate:
        single value from model prediction
    Returns:
    --------
    project_pt:
        projected GPS point on the road segment.
    """
    eid = eid.tolist() # convert tensor to normal value
    rate = rate.tolist()
    if eid <= 0 or rate < 0 or eid > (parameters.id_size-1) or rate > 1:
        # force eid and rate in the right range
        return SPoint(0, 0)

    coords = rn_dict[eid]['coords']
    offset = rn_dict[eid]['length'] * rate
    dist = 0  # temp distance for coords
    pre_dist = 0  # coords distance is smaller than offset

    if rate == 1.0:
        return coords[-1]
    if rate == 0.0:
        return coords[0]

    for i in range(len(coords) - 1):
        if i > 0:
            pre_dist += distance(coords[i - 1], coords[i])
        dist += distance(coords[i], coords[i + 1])
        if dist >= offset:
            coor_rate = (offset - pre_dist) / distance(coords[i], coords[i + 1])
            project_pt = cal_loc_along_line(coords[i], coords[i + 1], coor_rate)
            break

    return project_pt


def toseq(rn_dict, rids, rates, paramters):
    """
    Convert batched rids and rates to gps sequence.
    Args:
    -----
    rn_dict:
        use for rate2gps()
    rids:
        [trg len, batch size, id one hot dim]
    rates:
        [trg len, batch size]
    Returns:
    --------
    seqs:
        [trg len, batch size, 2]
    """
    batch_size = rids.size(1)
    trg_len = rids.size(0)
    seqs = torch.zeros(trg_len, batch_size, 2).to(paramters.device)

    for i in range(1, trg_len):
        for bs in range(batch_size):
            rid = rids[i][bs].argmax()
            rate = rates[i][bs]
            pt = rate2gps(rn_dict, rid, rate, paramters)
            seqs[i][bs][0] = pt.lat
            seqs[i][bs][1] = pt.lng
    return seqs


#####################################################################################################
#
# Constraint mask
#
#####################################################################################################
def get_rid_grid(mbr, grid_size, rn_dict):
    """
    Create a dict {key: grid id, value: rid}
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    max_xid = int((mbr.max_lat - mbr.min_lat) / lat_unit) + 1
    max_yid = int((mbr.max_lng - mbr.min_lng) / lng_unit) + 1

    grid_rn_dict = {}
    for k, v in rn_dict.items():
        pre_lat = v['coords'][0].lat
        pre_lng = v['coords'][0].lng
        pre_locgrid_x = max(1, int((pre_lat - mbr.min_lat) / lat_unit) + 1)
        pre_locgrid_y = max(1, int((pre_lng - mbr.min_lng) / lng_unit) + 1)
        
        
        if (pre_locgrid_x, pre_locgrid_y) not in grid_rn_dict.keys():
            grid_rn_dict[(pre_locgrid_x, pre_locgrid_y)] = [k]
        else:
            grid_rn_dict[(pre_locgrid_x, pre_locgrid_y)].append(k)

        for coord in v['coords'][1:]:
            lat = coord.lat
            lng = coord.lng
            locgrid_x = max(1, int((lat - mbr.min_lat) / lat_unit) + 1)
            locgrid_y = max(1, int((lng - mbr.min_lng) / lng_unit) + 1)

            if (locgrid_x, locgrid_y) not in grid_rn_dict.keys():
                grid_rn_dict[(locgrid_x, locgrid_y)] = [k]
            else:
                grid_rn_dict[(locgrid_x, locgrid_y)].append(k)

            mid_x_num = abs(locgrid_x - pre_locgrid_x)
            mid_y_num = abs(locgrid_y - pre_locgrid_y)

            if mid_x_num > 1 and mid_y_num <= 1:
                for mid_x in range(1, mid_x_num):
                    if (min(pre_locgrid_x,locgrid_x)+mid_x, locgrid_y) not in grid_rn_dict.keys():
                        grid_rn_dict[(min(pre_locgrid_x,locgrid_x)+mid_x, locgrid_y)] = [k]
                    else:
                        grid_rn_dict[(min(pre_locgrid_x,locgrid_x)+mid_x, locgrid_y)].append(k)

            elif mid_x_num <= 1 and mid_y_num > 1:
                for mid_y in range(1, mid_y_num):
                    if (locgrid_x, min(pre_locgrid_y,locgrid_y)+mid_y) not in grid_rn_dict.keys():
                        grid_rn_dict[(locgrid_x, min(pre_locgrid_y,locgrid_y)+mid_y)] = [k]
                    else:
                        grid_rn_dict[(locgrid_x, min(pre_locgrid_y,locgrid_y)+mid_y)].append(k)

            elif mid_x_num > 1 and mid_y_num > 1: 
                ttl_num = mid_x_num + mid_y_num + 1
                for mid in range(1, ttl_num):
                    mid_xid = min(lat, pre_lat) + mid*abs(lat - pre_lat)/ttl_num
                    mid_yid = min(lng, pre_lng) + mid*abs(lng - pre_lng)/ttl_num

            pre_lat = lat
            pre_lng = lng
            pre_locgrid_x = locgrid_x
            pre_locgrid_y = locgrid_y

    for k, v in grid_rn_dict.items():
        grid_rn_dict[k] = list(set(v))

    return grid_rn_dict, max_xid, max_yid

def exp_prob(beta, x):
    """
    error distance weight.
    """
    return  math.exp(-pow(x,2)/pow(beta,2))

def get_reachable_inds(pre_grid, cur_grid, grid_rn_dict,time_diff, parameters):
    reachable_inds = list(range(parameters.id_size))

    return reachable_inds

def get_dis_prob_vec(gps, rn, raw2new_rid_dict, parameters):
    """
    Args:
    -----
    gps: [SPoint, tid]
    """
    cons_vec = torch.zeros(parameters.id_size) + 1e-10
    candis = get_candidates(gps[0], rn, parameters.search_dist)
    if candis is not None:
        for candi_pt in candis:
            if candi_pt.eid in raw2new_rid_dict.keys():
                new_rid = raw2new_rid_dict[candi_pt.eid]
                prob = exp_prob(parameters.beta, candi_pt.error)
                cons_vec[new_rid] = prob
    else:
        cons_vec = torch.ones(parameters.id_size)
    return cons_vec

def get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths, trg_lengths, grid_rn_dict, rn, raw2new_rid_dict, parameters):
    max_trg_len = max(trg_lengths)
    batch_size = src_grid_seqs.size(0)

    constraint_mat = torch.zeros(batch_size, max_trg_len, parameters.id_size) + 1e-10
    pre_grids = torch.zeros(batch_size, max_trg_len, 3)
    cur_grids = torch.zeros(batch_size, max_trg_len, 3)

    for bs in range(batch_size):
        # first src gps
        pre_t = 1
        pre_grid = [int(src_grid_seqs[bs][pre_t][0].tolist()),
                    int(src_grid_seqs[bs][pre_t][1].tolist()),
                    pre_t]
        pre_gps = [SPoint(src_gps_seqs[bs][pre_t][0].tolist(),
                          src_gps_seqs[bs][pre_t][1].tolist()),
                   pre_t]
        pre_grids[bs, pre_t] = torch.tensor(pre_grid)
        cur_grids[bs, pre_t] = torch.tensor(pre_grid)

        if parameters.dis_prob_mask_flag:
            cons_vec = get_dis_prob_vec(pre_gps, rn, raw2new_rid_dict, parameters)
            constraint_mat[bs][pre_t] = cons_vec
        else:
            reachable_inds = get_reachable_inds(pre_grid, pre_grid, grid_rn_dict, 0, parameters)
            constraint_mat[bs][pre_t][reachable_inds] = 1

        # missed gps
        for i in range(2, src_lengths[bs]):
            cur_t = int(src_grid_seqs[bs,i,2].tolist())
            cur_grid = [int(src_grid_seqs[bs][i][0].tolist()),
                        int(src_grid_seqs[bs][i][1].tolist()),
                        cur_t]
            cur_gps = [SPoint(src_gps_seqs[bs][i][0].tolist(),
                          src_gps_seqs[bs][i][1].tolist()),
                       cur_t]
            pre_grids[bs, cur_t] = torch.tensor(cur_grid)
            cur_grids[bs, cur_t] = torch.tensor(cur_grid)

            time_diff = cur_t - pre_t
            reachable_inds = get_reachable_inds(pre_grid, cur_grid, grid_rn_dict, time_diff, parameters)

            for t in range(pre_t+1, cur_t):
                constraint_mat[bs][t][reachable_inds] = 1
                pre_grids[bs, t] = torch.tensor(pre_grid)
                cur_grids[bs, t] = torch.tensor(cur_grid)

            # middle src gps
            if parameters.dis_prob_mask_flag:
                cons_vec = get_dis_prob_vec(cur_gps, rn, raw2new_rid_dict, parameters)
                constraint_mat[bs][cur_t] = cons_vec
            else:
                reachable_inds = get_reachable_inds(cur_grid, cur_grid, grid_rn_dict, 0, parameters)
                constraint_mat[bs][cur_t][reachable_inds] = 1

            pre_t = cur_t
            pre_grid = cur_grid
            pre_gps = cur_gps

    return constraint_mat, pre_grids, cur_grids

#####################################################################################################
#
# Use for extracting POI features
#
#####################################################################################################
def get_poi_info(grid_poi_df, parameters):
    """
    ['company','food', 'gym', 'education','shopping','gov', 'viewpoint','entrance','house','life',
    'traffic','car','hotel','beauty','hospital','media','finance','entertainment','road','nature','landmark','address']
    """
    types = parameters.poi_type.split(',')
    norm_grid_poi_df=(grid_poi_df[types]-grid_poi_df[types].min())/(grid_poi_df[types].max()-grid_poi_df[types].min())
    norm_grid_poi_df = norm_grid_poi_df.fillna(0)
    
    norm_grid_poi_dict = {}
    for i in range(len(norm_grid_poi_df)):
        k = norm_grid_poi_df.index[i]
        v = norm_grid_poi_df.iloc[i].values
        norm_grid_poi_dict[k] = list(v)
        
    for xid in range(1, parameters.max_xid+1):
        for yid in range(1, parameters.max_yid+1):
            if (xid,yid) not in norm_grid_poi_dict.keys():
                norm_grid_poi_dict[(xid,yid)] = [0.] * len(types)
    return norm_grid_poi_dict

#####################################################################################################
#
# Use for extracting RN features
#
#####################################################################################################
def get_edge_results(eids, rn_dict):
    edge_results = []
    for eid in eids:
        u = rn_dict[eid]['coords'][0]
        v = rn_dict[eid]['coords'][-1]
        edge_results.append(((u.lng,u.lat),(v.lng,v.lat)))
    return edge_results

def extract_single_rn_features(edge_results, rn):
    part_g = nx.Graph()
    for u, v in edge_results:
        part_g.add_edge(u, v, **rn[u][v])
        
    tot_length = 0.0
    level_2_cnt = 0
    level_3_cnt = 0
    level_4_cnt = 0
    for u, v, data in part_g.edges(data=True):
        tot_length += data['length']
        if data['highway'] == 'trunk':
            level_2_cnt += 1
        elif data['highway'] == 'primary':
            level_3_cnt += 1
        elif data['highway'] == 'secondary':
            level_4_cnt += 1
    nb_intersections = 0
    for node, degree in part_g.degree():
        if degree > 2:
            nb_intersections += 1
    
    rn_features = np.array([tot_length, nb_intersections, level_2_cnt, level_3_cnt, level_4_cnt])

    return rn_features

def get_rn_info(rn, mbr, grid_size, grid_rn_dict, rn_dict):
    """
    rn_dict contains rn information
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    max_xid = int((mbr.max_lat - mbr.min_lat) / lat_unit) + 1
    max_yid = int((mbr.max_lng - mbr.min_lng) / lng_unit) + 1
    
    grid_rnfea_dict = {}
    for k,v in grid_rn_dict.items():
        eids = grid_rn_dict[k]
        edge_results = get_edge_results(eids, rn_dict)
        grid_rnfea_dict[k] = extract_single_rn_features(edge_results, rn)
        
    grid_rnfea_df = pd.DataFrame(grid_rnfea_dict).T
    norm_grid_rnfea_df=(grid_rnfea_df-grid_rnfea_df.min())/(grid_rnfea_df.max()-grid_rnfea_df.min())  # col norm
        
    norm_grid_rnfea_dict = {}
    for i in range(len(norm_grid_rnfea_df)):
        k = norm_grid_rnfea_df.index[i]
        v = norm_grid_rnfea_df.iloc[i].values
        norm_grid_rnfea_dict[k] = list(v)
        
    for xid in range(1, max_xid+1):
        for yid in range(1, max_yid+1):
            if (xid,yid) not in norm_grid_rnfea_dict.keys():
                norm_grid_rnfea_dict[(xid,yid)] = [0.] * len(v)
        
    return norm_grid_rnfea_dict

def get_rid_rnfea_dict(rn_dict, parameters):
    df = pd.DataFrame(rn_dict).T
    
    # standardization length
    df['norm_len'] = [np.log10(l) /np.log10(df['length'].max()) for l in df['length']]
#         df['norm_len'] = (df['length'] - df['length'].mean())/df['length'].std()
    
    # one hot road level
    one_hot_df = pd.get_dummies(df.level, prefix='level')
    df = df.join(one_hot_df)
    
    # get number of neighbours (standardization)
    g = nx.Graph()
    edges = []
    for coords in df['coords'].values:
        start_node = (coords[0].lat, coords[0].lng)
        end_node = (coords[-1].lat, coords[-1].lng)
        edges.append((start_node, end_node))
    g.add_edges_from(edges)

    num_start_neighbors = []
    num_end_neighbors = []
    for coords in df['coords'].values:
        start_node = (coords[0].lat, coords[0].lng)
        end_node = (coords[-1].lat, coords[-1].lng)
        num_start_neighbors.append(len(list(g.edges(start_node))))
        num_end_neighbors.append(len(list(g.edges(end_node))))
    df['num_start_neighbors'] = num_start_neighbors
    df['num_end_neighbors'] = num_end_neighbors
    start = df['num_start_neighbors']
    end = df['num_end_neighbors']
    # distribution is like gaussian --> use min max normalization
    df['norm_num_start_neighbors'] = (start - start.min())/(start.max() - start.min())  
    df['norm_num_end_neighbors'] = (end - end.min())/(end.max() - end.min())
    
    # convert to dict <key:rid, value:fea>
    norm_rid_rnfea_dict = {}
    for i in range(len(df)):
        k = df.index[i]
        v = df.iloc[i][['norm_len','level_2','level_3','level_4','level_5','level_6',\
                        'norm_num_start_neighbors','norm_num_end_neighbors']]
        norm_rid_rnfea_dict[k] = list(v)
    
    norm_rid_rnfea_dict[0] = [0.]*len(list(v)) # add soss
    return norm_rid_rnfea_dict

#####################################################################################################
#
# Use for online features
#
#####################################################################################################
def get_rid_grid_dict(grid_rn_dict):
    rid_grid_dict = {}
    for k, v in grid_rn_dict.items():
        for rid in v:
            if rid not in rid_grid_dict:
                rid_grid_dict[rid] = [k]
            else:
                rid_grid_dict[rid].append(k)

    for k,v in rid_grid_dict.items():
        rid_grid_dict[k] = list(set(v))
        
    return rid_grid_dict

def get_online_info_dict(grid_rn_dict, norm_grid_poi_dict, norm_grid_rnfea_dict, parameters):
    rid_grid_dict = get_rid_grid_dict(grid_rn_dict)
    online_features_dict = {}
    for rid in rid_grid_dict.keys():
        online_feas = [] 
        for grid in rid_grid_dict[rid]:
            try:
                poi = norm_grid_poi_dict[grid]
            except:
                poi = [0.]*5
            try:
                rnfea = norm_grid_rnfea_dict[grid]
            except:
                rnfea = [0.]*5
            online_feas.append(poi + rnfea)
            
        online_feas = np.array(online_feas)
        online_features_dict[rid] = list(online_feas.mean(axis=0))
    
    online_features_dict[0] = [0.]*online_feas.shape[1]  # add soss

    return online_features_dict

def get_dict_info_batch(input_id, features_dict):
    """
    batched dict info
    """
    # input_id = [1, batch size]
    features = []
    for rid in input_id.squeeze(1):
        features.append(features_dict[rid.cpu().tolist()])

    features = torch.tensor(features).float()
    # features = [1, batch size, features dim]
    return features

#####################################################################################################
#
# Use for visualization
#
#####################################################################################################
def get_plot_seq(raw_input, predict, target, src_len, trg_len):
    """
    Get input, prediction and ground truth GPS sequence.
    raw_input, predict, target = [seq len, batch size, 2] and the sos is not removed.
    """
    raw_input = raw_input[1:].permute(1, 0, 2)
    predict = predict[1:].permute(1, 0, 2)  # [batch size, seq len, 2]
    target = target[1:].permute(1, 0, 2)  # [batch size, seq len, 2]

    bs = predict.size(0)

    ls_pre_seq, ls_trg_seq, ls_input_seq =[], [], [] 
    for bs_i in range(bs):
        pre_seq = []
        trg_seq = []
        for len_i in range(trg_len[bs_i]-1):
            pre_seq.append([predict[bs_i, len_i][0].cpu().data.tolist(), predict[bs_i, len_i][1].cpu().data.tolist()])
            trg_seq.append([target[bs_i, len_i][0].cpu().data.tolist(), target[bs_i, len_i][1].cpu().data.tolist()])
        input_seq = []
        for len_i in range(src_len[bs_i]-1):
            input_seq.append([raw_input[bs_i, len_i][0].cpu().data.tolist(), raw_input[bs_i, len_i][1].cpu().data.tolist()])
        ls_pre_seq.append(pre_seq)
        ls_trg_seq.append(trg_seq)
        ls_input_seq.append(input_seq)
    return ls_input_seq, ls_pre_seq, ls_trg_seq


#####################################################################################################
#
# POIs
#
#####################################################################################################
def filterPOI(df, mbr):
    labels = ['公司企业', '美食', '运动健身', '教育培训', '购物', '政府机构', '旅游景点', '出入口', '房地产', '生活服务',
                  '交通设施', '汽车服务', '酒店', '丽人', '医疗', '文化传媒', '金融', '休闲娱乐', '道路','自然地物', '行政地标', '门址']
    eng_labels = ['company','food', 'gym', 'education','shopping','gov', 'viewpoint','entrance','house','life',
                 'traffic','car','hotel','beauty','hospital','media','finance','entertainment','road','nature','landmark','address']
    eng_labels_dict = {}
    for i in range(len(labels)):
        eng_labels_dict[labels[i]] = eng_labels[i]
    
    new_df = {'lat':[],'lng':[],'type':[]}
    for i in range(len(df)):
        gps = df.iloc[i]['经纬度wgs编码'].split(',')
        lat = float(gps[0])
        lng = float(gps[1])
        label = df.iloc[i]['一级行业分类']
        if mbr.contains(lat,lng) and (label is not np.nan):
            new_df['lat'].append(lat)
            new_df['lng'].append(lng)
            new_df['type'].append(eng_labels_dict[label])
    new_df = pd.DataFrame(new_df)
    return new_df


def get_poi_grid(mbr, grid_size, df):
    labels = ['company','food','shopping','viewpoint','house']
    new_df = filterPOI(df, mbr)
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    max_xid = int((mbr.max_lat - mbr.min_lat) / lat_unit) + 1
    max_yid = int((mbr.max_lng - mbr.min_lng) / lng_unit) + 1

    grid_poi_dict = {}
    for i in range(len(new_df)):
        lat = new_df.iloc[i]['lat']
        lng = new_df.iloc[i]['lng']
        label = new_df.iloc[i]['type']
        # only consider partial labels
        if label in labels:
            locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
            locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1

            if (locgrid_x, locgrid_y) not in grid_poi_dict.keys():
                grid_poi_dict[(locgrid_x, locgrid_y)] = {label:1}
            else:
                if label not in grid_poi_dict[(locgrid_x, locgrid_y)].keys():
                    grid_poi_dict[(locgrid_x, locgrid_y)][label] = 1
                else:
                    grid_poi_dict[(locgrid_x, locgrid_y)][label]+=1
                    
    # key: grid, value: [0.1,0.5,0.5] normalized POI by column
    grid_poi_df = pd.DataFrame(grid_poi_dict).T.fillna(0)
    
    norm_grid_poi_df=(grid_poi_df-grid_poi_df.min())/(grid_poi_df.max()-grid_poi_df.min())
#         norm_grid_poi_df = grid_poi_df.div(grid_poi_df.sum(axis=1), axis=0)  # row normalization
        
    norm_grid_poi_dict = {}
    for i in range(len(norm_grid_poi_df)):
        k = norm_grid_poi_df.index[i]
        v = norm_grid_poi_df.iloc[i].values
        norm_grid_poi_dict[k] = v
    
    return norm_grid_poi_dict, grid_poi_df


# extra_info_dir = "../data/map/extra_info/"
# poi_df = pd.read_csv(extra_info_dir+'jnPoiInfo.txt',sep='\t')
# norm_grid_poi_dict, grid_poi_df = get_poi_grid(mbr, args.grid_size, poi_df)

# save_pkl_data(norm_grid_poi_dict, extra_info_dir, 'poi_col_norm.pkl')
# grid_poi_df = pd.to_csv(extra_info_dir+'poi.csv')

#####################################################################################################
#
# others
#
#####################################################################################################
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self