#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/9/3 17:08
# Refernce: https://github.com/sjruan/tptk/blob/master/statistics.py

from utils.utils import create_dir

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import numpy as np
import pandas as pd
import os
import json


def plot_hist(data, x_axis, save_stats_dir, pic_name):
    plt.hist(data, weights=np.ones(len(data)) / len(data))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel(x_axis)
    plt.ylabel('Percentage')
    plt.savefig(os.path.join(save_stats_dir, pic_name))
    plt.clf()


def statistics(trajs, save_stats_dir, stats, save_stats_name, save_plot=False):
    """
    Plot basic statistical analysis , such as
    Args:
    -----
    traj_dir:
        str. directory of raw GPS points
    save_stats_dir:
        str. directory of saving stats results
    stats:
        dict. dictionary of stats
    save_stats_name:
        str. name of saving stats.
    plot_flat:
        boolean. if plot the histogram
    """

    create_dir(save_stats_dir)

    oids = set()
    tot_pts = 0

    distance_data = []  # geographical distance
    duration_data = []  # time difference between end and start time of a trajectory
    seq_len_data = []  # length of each trajectory
    traj_avg_time_interval_data = []
    traj_avg_dist_interval_data = []

    if len(stats) == 0:
        # if new, initialize stats with keys
        stats['#object'], stats['#points'], stats['#trajectories'] = 0, 0, 0
        stats['seq_len_data'], stats['distance_data'], stats['duration_data'], \
        stats['traj_avg_time_interval_data'], stats['traj_avg_dist_interval_data'] = [], [], [], [], []

    for traj in trajs:
        oids.add(traj.oid)
        nb_pts = len(traj.pt_list)
        tot_pts += nb_pts

        seq_len_data.append(nb_pts)
        distance_data.append(traj.get_distance() / 1000.0)
        duration_data.append(traj.get_duration() / 60.0)
        traj_avg_time_interval_data.append(traj.get_avg_time_interval())
        traj_avg_dist_interval_data.append(traj.get_avg_distance_interval())

    print('#objects_single:{}'.format(len(oids)))
    print('#points_single:{}'.format(tot_pts))
    print('#trajectories_single:{}'.format(len(trajs)))

    stats['#object'] += len(oids)
    stats['#points'] += tot_pts
    stats['#trajectories'] += len(trajs)
    stats['seq_len_data'] += seq_len_data
    stats['distance_data'] += distance_data
    stats['duration_data'] += duration_data
    stats['traj_avg_time_interval_data'] += traj_avg_time_interval_data
    stats['traj_avg_dist_interval_data'] += traj_avg_dist_interval_data

    print('#objects_total:{}'.format(stats['#object']))
    print('#points_total:{}'.format(stats['#points']))
    print('#trajectories_total:{}'.format(stats['#trajectories']))

    with open(os.path.join(save_stats_dir, save_stats_name + '.json'), 'w') as f:
        json.dump(stats, f)

    if save_plot:
        plot_hist(stats['seq_len_data'], '#Points', save_stats_dir, save_stats_name + '_nb_points_dist.png')
        plot_hist(stats['distance_data'], 'Distance (KM)', save_stats_dir, save_stats_name + '_distance_dist.png')
        plot_hist(stats['duration_data'], 'Duration (Min)', save_stats_dir, save_stats_name + '_duration_dist.png')
        plot_hist(stats['traj_avg_time_interval_data'], 'Time Interval (Sec)', save_stats_dir,
                  save_stats_name + '_time_interval_dist.png')
        plot_hist(stats['traj_avg_dist_interval_data'], 'Distance Interval (Meter)', save_stats_dir,
                  save_stats_name + '_distance_interval_dist.png')

    return stats


def stats_threshold(trajs):
    """
    Find threshold for trajectory preprocessing,
    including time interval for splitting; minimal length and maximal abnormal ratio.
    Args:
    -----
    trajs:
        list of Trajectory(). Sampled trajectories.
    Returns:
    --------
    thr_min_len, thr_max_abn_ratio, thr_normal_ratio, thr_split_traj_ts
    """
    stats = {'oid': [], 'tid': [], 'traj_len': [], 'num_ab_pts': [], 'abn_ts': [], 'avg_ts': [], 'avg_abn_ts': []}

    for i in range(len(trajs)):
        stats['oid'].append(trajs[i].oid)
        stats['tid'].append(trajs[i].tid)
        stats['traj_len'].append(len(trajs[i].pt_list))

        pt_list = trajs[i].pt_list
        pre_pt = pt_list[0]

        time_spans = []
        abn_ts_list = []
        for cur_pt in pt_list[1:]:
            time_span = (cur_pt.time - pre_pt.time).total_seconds()
            if time_span > 4:
                abn_ts_list.append(time_span)
            time_spans.append(time_span)
            pre_pt = cur_pt

        stats['num_ab_pts'].append(len(abn_ts_list))
        stats['abn_ts'].append(abn_ts_list)
        stats['avg_ts'].append(sum(time_spans) / len(time_spans))
        try:
            # in case len(abn_ts_list) = 0
            stats['avg_abn_ts'].append(sum(abn_ts_list) / len(abn_ts_list))
        except:
            stats['avg_abn_ts'].append(0)

    df_stats = pd.DataFrame(stats)
    df_stats['abn_ratio'] = df_stats.num_ab_pts / df_stats.traj_len

    thr_min_len = df_stats.traj_len.quantile(0.1)  # if less than min length(number of points), remove
    thr_max_abn_ratio = df_stats.abn_ratio.quantile(0.8)  # if larger than max abnormal ratio, remove
    # thr_normal_ratio = 0  # if abn_ratio == 0, use the trajectory directly
    # thr_split_traj_ts = 60  # if time_interval is larger than 60 seconds, split trajectories.
    # print("min len: {}, max abnormal ratio: {}, normal ratio: {}, split threshold: {} seconds".
    #       format(thr_min_len, thr_max_abn_ratio, thr_normal_ratio, thr_split_traj_ts))
    print("min len: {}, max abnormal ratio: {}".format(thr_min_len, thr_max_abn_ratio))

    return df_stats, thr_min_len, thr_max_abn_ratio  #, thr_normal_ratio, thr_split_traj_ts
