#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/9/22 16:29
# @authoer: Huimin Ren
from common.spatial_func import distance, cal_loc_along_line
from common.trajectory import Trajectory, STPoint
import numpy as np

from map_matching.candidate_point import CandidatePoint
from map_matching.hmm.hmm_map_matcher import TIHMMMapMatcher

class Imputation:
    def __init__(self):
        pass

    def impute(self, traj):
        pass


class LinearImputation(Imputation):
    """
    Uniformly interpolate GPS points into a trajectory.
    Args:
    -----
    time_interval:
        int. unit in seconds. the time interval between two points.
    """
    def __init__(self, time_interval):
        super(LinearImputation, self).__init__()
        self.time_interval = time_interval

    def impute(self, traj):
        pt_list = traj.pt_list
        if len(pt_list) <= 1:
            return []

        pre_pt = pt_list[0]
        new_pt_list = [pre_pt]
        for cur_pt in pt_list[1:]:
            time_span = (cur_pt.time - pre_pt.time).total_seconds()
            num_pts = 0  # do not have to interpolate points
            if time_span % self.time_interval > self.time_interval / 2:
                # if the reminder is larger than half of the time interval
                num_pts = time_span // self.time_interval  # quotient
            elif time_span > self.time_interval:
                # if if the reminder is smaller than half of the time interval and not equal to time interval
                num_pts = time_span // self.time_interval - 1

            interp_list = []
            unit_lat = abs(cur_pt.lat - pre_pt.lat) / (num_pts + 1)
            unit_lng = abs(cur_pt.lng - pre_pt.lng) / (num_pts + 1)
            unit_ts = (cur_pt.time - pre_pt.time) / (num_pts + 1)
            sign_lat = np.sign(cur_pt.lat - pre_pt.lat)
            sign_lng = np.sign(cur_pt.lng - pre_pt.lng)
            for i in range(int(num_pts)):
                new_lat = pre_pt.lat + sign_lat * (i+1) * unit_lat
                new_lng = pre_pt.lng + sign_lng * (i+1) * unit_lng
                new_time = pre_pt.time + (i+1) * unit_ts
                interp_list.append(STPoint(new_lat, new_lng, new_time))

            new_pt_list.extend(interp_list)
            new_pt_list.append(cur_pt)
            pre_pt = cur_pt

        new_traj = Trajectory(traj.oid, traj.tid, new_pt_list)

        return new_traj


class MMLinearImputation(Imputation):

    def __init__(self, time_interval):
        super(MMLinearImputation, self).__init__()
        self.time_interval = time_interval

    def impute(self, traj, rn, rn_dict):
        try:
            map_matcher = TIHMMMapMatcher(rn)
            mm_ls_path = map_matcher.match_to_path(traj)[0]  # find shortest path
        except:
            # cannot find shortest path
            return None

        path_eids = [p.eid for p in mm_ls_path.path_entities]

        pre_mm_pt = traj.pt_list[0]
        new_pt_list = [pre_mm_pt]
        for cur_mm_pt in traj.pt_list[1:]:
            time_span = (cur_mm_pt.time - pre_mm_pt.time).total_seconds()
            num_pts = 0  # do not have to interpolate points
            if time_span % self.time_interval > self.time_interval / 2:
                # if the reminder is larger than half of the time interval
                num_pts = time_span // self.time_interval  # quotient
            elif time_span > self.time_interval:
                # if if the reminder is smaller than half of the time interval and not equal to time interval
                num_pts = time_span // self.time_interval - 1

            if pre_mm_pt.data['candi_pt'] is None or cur_mm_pt.data['candi_pt'] is None:
                return None

            pre_eid = pre_mm_pt.data['candi_pt'].eid
            cur_eid = cur_mm_pt.data['candi_pt'].eid
            two_points_coords, two_points_eids, ttl_dis = self.get_two_points_coords(path_eids, pre_eid, cur_eid,
                                                                                     pre_mm_pt, cur_mm_pt, rn_dict)
            interp_list = self.get_interp_list(num_pts, cur_mm_pt, pre_mm_pt, ttl_dis,
                                               two_points_eids, two_points_coords, rn_dict)

            new_pt_list.extend(interp_list)
            new_pt_list.append(cur_mm_pt)
            pre_mm_pt = cur_mm_pt
        new_traj = Trajectory(traj.oid, traj.tid, new_pt_list)

        # get all coords of shortest path
        # path_coords = []
        # for eid in path_eids:
        #     path_coords.extend(rn_dict[eid]['coords'])
        # path_pt_list = []
        # for pt in path_coords:
        #     path_pt_list.append([pt.lat, pt.lng])

        return new_traj

    def get_interp_list(self, num_pts, cur_mm_pt, pre_mm_pt, ttl_dis, two_points_eids, two_points_coords, rn_dict):
        interp_list = []
        unit_ts = (cur_mm_pt.time - pre_mm_pt.time) / (num_pts + 1)
        for n in range(int(num_pts)):
            new_time = pre_mm_pt.time + (n + 1) * unit_ts
            move_dis = (ttl_dis / num_pts) * n + pre_mm_pt.data['candi_pt'].offset

            # get eid and offset
            pre_road_dist, road_dist = 0, 0
            for i in range(len(two_points_eids)):
                if i > 0:
                    pre_road_dist += rn_dict[two_points_eids[i - 1]]['length']
                road_dist += rn_dict[two_points_eids[i]]['length']
                if move_dis <= road_dist:
                    insert_eid = two_points_eids[i]
                    insert_offset = move_dis - pre_road_dist
                    break

            # get lat and lng
            dist, pre_dist = 0, 0
            for i in range(len(two_points_coords) - 1):
                if i > 0:
                    pre_dist += distance(two_points_coords[i - 1][0], two_points_coords[i][0])
                dist += distance(two_points_coords[i][0], two_points_coords[i + 1][0])
                if dist >= move_dis:
                    coor_rate = (move_dis - pre_dist) / distance(two_points_coords[i][0],
                                                                 two_points_coords[i + 1][0])
                    project_pt = cal_loc_along_line(two_points_coords[i][0], two_points_coords[i + 1][0], coor_rate)
                    break
            data = {'candi_pt': CandidatePoint(project_pt.lat, project_pt.lng, insert_eid, 0, insert_offset, 0)}
            interp_list.append(STPoint(project_pt.lat, project_pt.lng, new_time, data))

        return interp_list

    def get_two_points_coords(self, path_eids, pre_eid, cur_eid, pre_mm_pt, cur_mm_pt, rn_dict):
        if pre_eid == cur_eid:
            # if in the same road
            two_points_eids = [path_eids[path_eids.index(pre_eid)]]
            two_points_coords = [[item, two_points_eids[0]] for item in rn_dict[two_points_eids[0]]['coords']]
            ttl_dis = cur_mm_pt.data['candi_pt'].offset - cur_mm_pt.data['candi_pt'].offset
        else:
            # if in different road
            start = path_eids.index(pre_eid)
            end = path_eids.index(cur_eid) + 1
            if start >= end:
                end = path_eids.index(cur_eid, start)  # cur_eid shows at least twice
            two_points_eids = path_eids[start: end]

            two_points_coords = []  # 2D, [gps, eid]
            ttl_eids_dis = 0
            for eid in two_points_eids[:-1]:
                tmp_coords = [[item, eid] for item in rn_dict[eid]['coords']]
                two_points_coords.extend(tmp_coords)
                ttl_eids_dis += rn_dict[eid]['length']
            ttl_dis = ttl_eids_dis - pre_mm_pt.data['candi_pt'].offset + cur_mm_pt.data['candi_pt'].offset
            tmp_coords = [[item, two_points_eids[-1]] for item in rn_dict[two_points_eids[-1]]['coords']]
            two_points_coords.extend(tmp_coords)  # add coords after computing ttl_dis

        return two_points_coords, two_points_eids, ttl_dis
