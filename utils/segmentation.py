from common.trajectory import Trajectory, get_tid


class Segmentation:
    def __init__(self):
        pass

    def segment(self, traj):
        pass


class TimeIntervalSegmentation(Segmentation):
    """
    Split trajectory if the time interval between two GPS points is larger than max_time_interval_min
    Store sub-trajectory when its length is larger than min_len
    """
    def __init__(self, max_time_interval_min, min_len=1):
        super(Segmentation, self).__init__()
        self.max_time_interval = max_time_interval_min * 60
        self.min_len = min_len

    def segment(self, traj):
        segmented_traj_list = []
        pt_list = traj.pt_list
        if len(pt_list) <= 1:
            return []
        oid = traj.oid
        pre_pt = pt_list[0]
        partial_pt_list = [pre_pt]
        for cur_pt in pt_list[1:]:
            time_span = (cur_pt.time - pre_pt.time).total_seconds()
            if time_span <= self.max_time_interval:
                partial_pt_list.append(cur_pt)
            else:
                if len(partial_pt_list) > self.min_len:
                    segmented_traj = Trajectory(oid, get_tid(oid, partial_pt_list), partial_pt_list)
                    segmented_traj_list.append(segmented_traj)
                partial_pt_list = [cur_pt]  # re-initialize partial_pt_list
            pre_pt = cur_pt
        if len(partial_pt_list) > self.min_len:
            segmented_traj = Trajectory(oid, get_tid(oid, partial_pt_list), partial_pt_list)
            segmented_traj_list.append(segmented_traj)
        return segmented_traj_list


