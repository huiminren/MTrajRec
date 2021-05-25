#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/9/23 15:40
# Reference: https://github.com/huiminren/tptk/blob/master/common/trajectory.py
from common.trajectory import get_tid
from utils.coord_transform import GCJ02ToWGS84, WGS84ToGCJ02, Convert


class SaveTraj:
    """
    SaveTraj is an abstract class for storing trajectory.
    It defines store() function for storing trajectory to different format.
    """
    def __init__(self, convert_method):
        # GCJ: Auto, Didi
        # WGS: OSM, Tiandi
        if convert_method == 'GCJ02ToWGS84':
            self.convert = GCJ02ToWGS84()
        elif convert_method == 'WGS84ToGCJ02':
            self.convert = WGS84ToGCJ02()
        elif convert_method is None:
            self.convert = Convert()

    def store(self, trajs, target_path):
        pass


class SaveTraj2Raw(SaveTraj):
    def __init__(self, convert_method=None):
        super().__init__(convert_method)

    def store(self, trajs, target_path):
        time_format = '%Y/%m/%d %H:%M:%S'
        with open(target_path, 'w') as f:
            for traj in trajs:
                pt_list = traj.pt_list
                tid = get_tid(traj.oid, pt_list)
                f.write('#,{},{},{},{},{} km\n'.format(tid, traj.oid, pt_list[0].time.strftime(time_format),
                                                       pt_list[-1].time.strftime(time_format),
                                                       traj.get_distance() / 1000))
                for pt in pt_list:
                    lng, lat = self.convert.convert(pt.lng, pt.lat)
                    f.write('{},{},{}\n'.format(
                        pt.time.strftime(time_format), lat, lng))


class SaveTraj2MM(SaveTraj):
    """
    """
    def __init__(self, convert_method=None):
        super().__init__(convert_method)

    def store(self, trajs, target_path):
        time_format = '%Y/%m/%d %H:%M:%S'
        with open(target_path, 'w') as f:
            for traj in trajs:
                pt_list = traj.pt_list
                tid = get_tid(traj.oid, pt_list)
                f.write('#,{},{},{},{},{} km\n'.format(tid, traj.oid, pt_list[0].time.strftime(time_format),
                                                       pt_list[-1].time.strftime(time_format),
                                                       traj.get_distance() / 1000))
                for pt in pt_list:
                    candi_pt = pt.data['candi_pt']
                    if candi_pt is not None:
                        f.write('{},{},{},{},{},{},{},{},{}\n'.format(pt.time.strftime(time_format), pt.lat, pt.lng,
                                                                   candi_pt.eid, candi_pt.lat, candi_pt.lng,
                                                                   candi_pt.error, candi_pt.offset, candi_pt.rate))
                    else:
                        f.write('{},{},{},None,None,None,None,None,None\n'.format(
                            pt.time.strftime(time_format), pt.lat, pt.lng))


class SaveTraj2JUST(SaveTraj):
    """
    Convert trajs to JUST format.
    cvs file. trajectory_id, oid, time, lat, lng
    """
    def __init__(self, convert_method=None):
        super().__init__(convert_method)

    def store(self, trajs, target_path):
        """
        Convert trajs to JUST format.
        cvs file. trajectory_id (primary key), oid, time, lat, lng
        Args:
        ----
        trajs:
            list. list of Trajectory()
        target_path:
            str. target path (directory + file_name)
        """
        with open(target_path, 'w') as f:
            for traj in trajs:
                for pt in traj.pt_list:
                    lng, lat = self.convert.convert(pt.lng, pt.lat)
                    f.write('{},{},{},{},{}\n'.format(traj.tid, traj.oid, pt.time, lat, lng))

