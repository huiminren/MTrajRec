#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/9/7 17:07
# Add new method
from statistics import median

from common.spatial_func import distance
from common.trajectory import Trajectory, STPoint

"""
All the methods are refered from 
Zheng, Y. and Zhou, X. eds., 2011. 
Computing with spatial trajectories. Springer Science & Business Media.
Chapter 1 Trajectory Preprocessing

"""


class NoiseFilter:
    def filter(self, traj):
        pass

    def get_tid(self, oid, clean_pt_list):
        return oid + '_' + clean_pt_list[0].time.strftime('%Y%m%d%H%M') + '_' + \
               clean_pt_list[-1].time.strftime('%Y%m%d%H%M')


# modify point
class MedianFilter(NoiseFilter):
    """
    Smooth each point with median value, since mean filter is sensitive to outliers.
    x_i = median(z_{i-n+1}, z_{i-n+2}, ..., z_{i-1}, z_{i}) eq.1.4
    i: ith point
    z: original points
    n: window size
    """

    def __init__(self, win_size=3):
        super(MedianFilter, self).__init__()
        self.win_size = win_size // 2  # put the target value in middle

    def filter(self, traj):
        pt_list = traj.pt_list.copy()
        if len(pt_list) <= 1:
            return None

        for i in range(1, len(pt_list)-1):
            # post-preprocessing. make sure index_i is in the middle.
            wind_size = self.win_size
            lats, lngs = [], []
            if self.win_size > i:
                wind_size = i
            elif self.win_size > len(pt_list) - 1 - i:
                wind_size = len(pt_list) - 1 - i
            # get smoothed location
            for pt in pt_list[i-wind_size:i+wind_size+1]:
                lats.append(pt.lat)
                lngs.append(pt.lng)

            pt_list[i] = STPoint(median(lats), median(lngs), pt_list[i].time)

        if len(pt_list) > 1:
            # return Trajectory(traj.oid, self.get_tid(traj.oid, pt_list), pt_list)
            return Trajectory(traj.oid, traj.tid, pt_list)
        else:
            return None


# modify point
class HeuristicMeanFilter(NoiseFilter):
    """
    Find outlier by speed (if the current speed is out of max speed)
    Replace outlier with mean
    Mean filter usually handles individual noise points with a dense representation.
    """

    def __init__(self, max_speed, win_size=1):
        """
        Args:
        ----
        max_speed:
            int. m/s. threshold of noise speed
        win_size:
            int. prefer odd number. window size of calculating mean value to replace noise.
        """
        super(NoiseFilter, self).__init__()

        self.max_speed = max_speed
        self.win_size = win_size

    def filter(self, traj):
        """
        When previous speed and next speed both are larger than max speed, then considering it as outlier.
        Replace outlier with mean value. The range is defined by window size.
        consider about the boundary.
        make sure noise value is in the middle.

        Args:
        -----
        traj:
            Trajectory(). a single trajectory
        Returns:
        --------
        new_traj:
            Trajectory(). replace noise with mean or median
        """
        pt_list = traj.pt_list.copy()
        if len(pt_list) <= 1:
            return None
        for i in range(1, len(pt_list) - 1):
            time_span_pre = (pt_list[i].time - pt_list[i - 1].time).total_seconds()
            dist_pre = distance(pt_list[i - 1], pt_list[i])
            time_span_next = (pt_list[i + 1].time - pt_list[i].time).total_seconds()
            dist_next = distance(pt_list[i], pt_list[i + 1])
            # compute current speed
            speed_pre = dist_pre / time_span_pre
            speed_next = dist_next / time_span_next
            # if the first point is noise
            if i == 1 and speed_pre > self.max_speed > speed_next:
                lat = pt_list[i].lat * 2 - pt_list[i + 1].lat
                lng = pt_list[i].lng * 2 - pt_list[i + 1].lng
                pt_list[0] = STPoint(lat, lng, pt_list[0].time)
            # if the last point is noise
            elif i == len(pt_list) - 2 and speed_next > self.max_speed >= speed_pre:
                lat = pt_list[i - 1].lat * 2 - pt_list[i - 2].lat
                lng = pt_list[i - 1].lng * 2 - pt_list[i - 2].lng
                pt_list[i + 1] = STPoint(lat, lng, pt_list[i].time)
            # if the middle point is noise
            elif speed_pre > self.max_speed and speed_next > self.max_speed:
                pt_list[i] = STPoint(0, 0, pt_list[i].time)
                lats, lngs = [], []
                # fix index bug. make sure index_i is in the middle.
                wind_size = self.win_size
                if self.win_size > i:
                    wind_size = i
                elif self.win_size > len(pt_list) - 1 - i:
                    wind_size = len(pt_list) - 1 - i
                for pt in pt_list[i-wind_size:i+wind_size+1]:
                    lats.append(pt.lat)
                    lngs.append(pt.lng)

                lat = sum(lats) / (len(lats) - 1)
                lng = sum(lngs) / (len(lngs) - 1)
                pt_list[i] = STPoint(lat, lng, pt_list[i].time)

        if len(pt_list) > 1:
            # return Trajectory(traj.oid, self.get_tid(traj.oid, pt_list), pt_list)
            return Trajectory(traj.oid, traj.tid, pt_list)
        else:
            return None


# remove point
class HeuristicFilter(NoiseFilter):
    """
    Remove outlier if it is out of the max speed
    """

    def __init__(self, max_speed):
        super(NoiseFilter, self).__init__()
        self.max_speed = max_speed

    def filter(self, traj):
        pt_list = traj.pt_list
        if len(pt_list) <= 1:
            return None

        remove_inds = []
        for i in range(1, len(pt_list) - 1):
            time_span_pre = (pt_list[i].time - pt_list[i - 1].time).total_seconds()
            dist_pre = distance(pt_list[i - 1], pt_list[i])
            time_span_next = (pt_list[i + 1].time - pt_list[i].time).total_seconds()
            dist_next = distance(pt_list[i], pt_list[i + 1])
            speed_pre = dist_pre / time_span_pre
            speed_next = dist_next / time_span_next
            # the first point is outlier
            if i == 1 and speed_pre > self.max_speed > speed_next:
                remove_inds.append(0)
            # the last point is outlier
            elif i == len(pt_list) - 2 and speed_next > self.max_speed >= speed_pre:
                remove_inds.append(len(pt_list) - 1)
            # middle point is outlier
            elif speed_pre > self.max_speed and speed_next > self.max_speed:
                remove_inds.append(i)

        clean_pt_list = []
        for j in range(len(pt_list)):
            if j in remove_inds:
                continue
            clean_pt_list.append(pt_list[j])

        if len(clean_pt_list) > 1:
            # return Trajectory(traj.oid, self.get_tid(traj.oid, pt_list), pt_list)
            return Trajectory(traj.oid, traj.tid, pt_list)
        else:
            return None


# remove point
class STFilter(NoiseFilter):
    """
    remove point if it is out of mbr
    """
    def __init__(self, mbr, start_time, end_time):
        super(STFilter, self).__init__()
        self.mbr = mbr
        self.start_time = start_time
        self.end_time = end_time

    def filter(self, traj):
        pt_list = traj.pt_list
        if len(pt_list) <= 1:
            return None
        clean_pt_list = []
        for pt in pt_list:
            if self.start_time <= pt.time < self.end_time and self.mbr.contains(pt.lat, pt.lng):
                clean_pt_list.append(pt)
        if len(clean_pt_list) > 1:
            # return Trajectory(traj.oid, self.get_tid(traj.oid, pt_list), pt_list)
            return Trajectory(traj.oid, traj.tid, pt_list)
        else:
            return None
