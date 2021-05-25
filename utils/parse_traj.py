#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/9/23 15:40
# Reference: https://github.com/huiminren/tptk/blob/master/common/trajectory.py

import re
from datetime import datetime
import pandas as pd

from common.trajectory import Trajectory, STPoint
from map_matching.candidate_point import CandidatePoint


class ParseTraj:
    """
    ParseTraj is an abstract class for parsing trajectory.
    It defines parse() function for parsing trajectory.
    """
    def __init__(self):
        pass

    def parse(self, input_path):
        """
        The parse() function is to load data to a list of Trajectory()
        """
        pass


class ParseRawTraj(ParseTraj):
    """
    Parse original GPS points to trajectories list. No extra data preprocessing
    """
    def __init__(self):
        super().__init__()

    def parse(self, input_path):
        """
        Args:
        -----
        input_path:
            str. input directory with file name
        Returns:
        --------
        trajs:
            list. list of trajectories. trajs contain input_path file's all gps points
        """
        time_format = '%Y/%m/%d %H:%M:%S'
        tid_to_remove = '[:/ ]'
        with open(input_path, 'r') as f:
            trajs = []
            pt_list = []
            for line in f.readlines():
                attrs = line.rstrip().split(',')
                if attrs[0] == '#':
                    if len(pt_list) > 1:
                        traj = Trajectory(oid, tid, pt_list)
                        trajs.append(traj)
                    oid = attrs[2]
                    tid = attrs[1]
                    pt_list = []
                else:
                    lat = float(attrs[1])
                    lng = float(attrs[2])
                    pt = STPoint(lat, lng, datetime.strptime(attrs[0], time_format))
                    # pt contains all the attributes of class STPoint
                    pt_list.append(pt)
            if len(pt_list) > 1:
                traj = Trajectory(oid, tid, pt_list)
                trajs.append(traj)
        return trajs


class ParseMMTraj(ParseTraj):
    """
    Parse map matched GPS points to trajectories list. No extra data preprocessing
    """
    def __init__(self):
        super().__init__()

    def parse(self, input_path):
        """
        Args:
        -----
        input_path:
            str. input directory with file name
        Returns:
        --------
        trajs:
            list. list of trajectories. trajs contain input_path file's all gps points
        """
        time_format = '%Y/%m/%d %H:%M:%S'
        tid_to_remove = '[:/ ]'
        with open(input_path, 'r') as f:
            trajs = []
            pt_list = []
            for line in f.readlines():
                attrs = line.rstrip().split(',')
                if attrs[0] == '#':
                    if len(pt_list) > 1:
                        traj = Trajectory(oid, tid, pt_list)
                        trajs.append(traj)
                    oid = attrs[2]
                    tid = attrs[1]
                    pt_list = []
                else:
                    lat = float(attrs[1])
                    lng = float(attrs[2])
                    if attrs[3] == 'None':
                        candi_pt = None
                    else:
                        eid = int(attrs[3])
                        proj_lat = float(attrs[4])
                        proj_lng = float(attrs[5])
                        error = float(attrs[6])
                        offset = float(attrs[7])
                        rate = float(attrs[8])
                        candi_pt = CandidatePoint(proj_lat, proj_lng, eid, error, offset, rate)
                    pt = STPoint(lat, lng, datetime.strptime(attrs[0], time_format), {'candi_pt': candi_pt})
                    # pt contains all the attributes of class STPoint
                    pt_list.append(pt)
            if len(pt_list) > 1:
                traj = Trajectory(oid, tid, pt_list)
                trajs.append(traj)
        return trajs


class ParseJUSTInputTraj(ParseTraj):
    """
    Parse JUST input format to list of Trajectory()
    """
    def __init__(self):
        super().__init__()

    def parse(self, input_path):
        time_format = '%Y-%m-%d %H:%M:%S'
        with open(input_path, 'r') as f:
            trajs = []
            pt_list = []
            pre_tid = ''
            for line in f.readlines():
                attrs = line.rstrip().split(',')
                tid = attrs[0]
                oid = attrs[1]
                time = datetime.strptime(attrs[2][:19], time_format)
                lat = float(attrs[3])
                lng = float(attrs[4])
                pt = STPoint(lat, lng, time)
                if pre_tid != tid:
                    if len(pt_list) > 1:
                        traj = Trajectory(oid, pre_tid, pt_list)
                        trajs.append(traj)
                    pt_list = []
                pt_list.append(pt)
                pre_tid = tid
            if len(pt_list) > 1:
                traj = Trajectory(oid, tid, pt_list)
                trajs.append(traj)

        return trajs


class ParseJUSTOutputTraj(ParseTraj):
    """
    Parse JUST output to trajectories list. The output format will be the same as Trajectory()
    """
    def __init__(self):
        super().__init__()

    def parse(self, input_path, feature_flag=False):
        """
        Args:
        -----
        input_path:
            str. input directory with file name
        Returns:
        --------
        trajs:
            list of Trajectory()

        'oid': object_id
        'geom': line string of raw trajectory
        'time': trajectory start time
        'tid': trajectory id
        'time_series': line string of map matched trajectory and containing other features
            raw start time, raw lng, raw lat, road segment ID, index of road segment,
            distance between raw point to map matched point, distanced between projected point and start of road segment.
        'start_position': raw start position
        'end_position': raw end position
        'point_number': number of points in the trajectory
        'length': distance of the trajectory in km
        'speed': average speed of the trajectory in km/h
        'signature': signature for GIS
        'id': primary key
        """
        col_names = ['oid', 'geom', 'tid', 'time_series']
        df = pd.read_csv(input_path, sep='|', usecols=col_names)

        str_to_remove = '[LINESTRING()]'
        time_format = '%Y-%m-%d %H:%M:%S'
        trajs = []
        for i in range(len(df)):
            oid = df['oid'][i]
            tid = str(df['tid'][i])
            # prepare to get map matched lat, lng and original datetime for each GPS point
            time_series = df['time_series'][i]
            geom = re.sub(str_to_remove, "", df['geom'][i])  # load geom and remove "LINESTRING()"
            ts_list = time_series.split(';')  # contain datetime of each gps point and original gps location
            geom_list = geom.split(',')  # contain map matched gps points
            assert len(ts_list) == len(geom_list)

            pt_list = []
            for j in range(len(ts_list)):
                tmp_location = geom_list[j].split(" ")
                tmp_features = ts_list[j].split(",")
                lat = tmp_location[2]
                lng = tmp_location[1]
                time = tmp_features[0][:19]  # ts_list[j][:19]

                if feature_flag:
                    rid = int(tmp_features[3])
                    rdis = float(tmp_features[-1][:-1])
                    pt = STPoint(lat, lng, datetime.strptime(time, time_format), rid=rid, rdis=rdis)
                else:
                    pt = STPoint(lat, lng, datetime.strptime(time, time_format))

                pt_list.append(pt)
            traj = Trajectory(oid, tid, pt_list)
            trajs.append(traj)

        return trajs
