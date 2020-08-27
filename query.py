import numpy as np


def get_ids_entering_vehicles(df, onramp_id, mainline_id):
    # find all ids whose initial lane is onramp
    # from these, find those who at one point are in mainline
    # assume:
    # - if in onramp at any point, that was the initial lane
    # - no jumping from onramp to a far-left lane)
    df_onramp = df[df['Lane_ID'] == onramp_id]
    vids = df_onramp['Vehicle_ID'].unique()
    df_first_on_onramp = df[df['Vehicle_ID'].isin(vids)]
    entered_vids = []
    for vid in vids:
        df_i = df_first_on_onramp[df_first_on_onramp['Vehicle_ID'] == vid]
        if df_i[df_i['Lane_ID'] == mainline_id].size > 0:
            entered_vids.append(vid)
    return entered_vids


def get_close_ids_in_lane(df, ego_id, ego_lane_id, lane_id, tau_dist):
    # at each time that ego_id is present in ego_lane
    # - find all vehicles in lane
    # - add to set those close to ego
    close_ids = set()
    frames = df[(df['Vehicle_ID'] == ego_id) & (df['Lane_ID'] == ego_lane_id)]['Frame_ID'].values
    for frame in frames:
        df_i = df[df['Frame_ID'] == frame]
        other_df = df_i[(df_i['Vehicle_ID'] != ego_id) & (df_i['Lane_ID'] == lane_id)]
        other_ids = other_df['Vehicle_ID'].values
        other_xy = other_df[['Local_X', 'Local_Y']].values
        ego_xy = df_i[df_i['Vehicle_ID'] == ego_id][['Local_X', 'Local_Y']].values[0]
        mask = ((other_xy - ego_xy)**2).sum(axis=1) < tau_dist**2
        close_ids_i = other_ids[mask]
        close_ids.update(close_ids_i)
    return close_ids


def get_first_vision_clearance(df, passed_id, passer_id, before_frame_id=np.inf):
    # based on passer's midpoint clearing front edge of passed
    # -1 if no clearance ever made
    df_shared = get_shared_frames_df(df, (passed_id, passer_id))
    passed_front_y = df_shared[df_shared['Vehicle_ID'] == passed_id]['Local_Y'].values
    passer_center_y = df_shared[df_shared['Vehicle_ID'] == passer_id]['Local_Y_center'].values
    frames = df_shared['Frame_ID'].unique()  # sorted
    is_cleared = passer_center_y - passed_front_y > 0
    if np.sum(is_cleared) == 0:
        return -1
    clearance_inds = np.arange(frames.size)[is_cleared]
    cleared_frames = frames[clearance_inds]
    cleared_frames = cleared_frames[cleared_frames < before_frame_id]
    if cleared_frames.size == 0:
        return -1
    frame_id = cleared_frames.min()
    return frame_id


def get_first_lane_clearance(df, merger_id, lane_boundary_x, is_right_merge=False,
                             crossing_pt='Local_X_center', before_frame_id=np.inf):
    # first make sure they were in src lane before clearing
    df = df[df['Vehicle_ID'] == merger_id]
    is_merged_left = lane_boundary_x - df[crossing_pt].values > 0
    is_merged = ~is_merged_left if is_right_merge else is_merged_left
    # check that there is at least a streak of 3 not currently merged
    min_ind = np.where((is_merged[2:] == 0) & (is_merged[1:-1] == 0) & (is_merged[:-2] == 0))[0]
    if min_ind.size == 0:
        return -1
    min_ind = min_ind[0]
    if np.sum(is_merged) == 0:
        return -1
    clearance_inds = np.arange(is_merged.size)[is_merged]
    clearance_inds = clearance_inds[min_ind < clearance_inds]  # deal with multiple entries
    cleared_frames = df['Frame_ID'].values[clearance_inds]
    cleared_frames = cleared_frames[cleared_frames < before_frame_id]
    if cleared_frames.size == 0:
        return -1
    frame_id = cleared_frames.min()
    return frame_id


def get_lead_vid(df, lane_id, vid, **kwargs):
    vid = get_leadlag_vid(df, lane_id, vid, leadlag_key='Preceding', **kwargs)
    return vid


def get_lag_vid(df, lane_id, vid, **kwargs):
    vid = get_leadlag_vid(df, lane_id, vid, leadlag_key='Following', **kwargs)
    return vid


def get_leadlag_vid_from_closest_midpoint(df, lane_id, vid, at_frame):
    # based on y-point of vid at given frame, get closed vid before and after in given lane
    y_key = 'Local_Y_center'
    y_vid = df[(df['Vehicle_ID'] == vid) & (df['Frame_ID'] == at_frame)][y_key].values[0]
    df_lane = df[(df['Lane_ID'] == lane_id) & (df['Frame_ID'] == at_frame) & (~df['Vehicle_ID'].isin([vid]))]
    y_offsets = df_lane[y_key].values - y_vid
    if np.all(y_offsets <= 0) or np.all(y_offsets >= 0):
        # No lead or no lag
        return np.nan, np.nan
    lead_ind = np.argmin(y_offsets + (y_offsets <= 0)*1e6)
    lag_ind = np.argmin(-y_offsets + (y_offsets >= 0)*1e6)
    lead_vid = df_lane['Vehicle_ID'].values[lead_ind]
    lag_vid = df_lane['Vehicle_ID'].values[lag_ind]
    return lead_vid, lag_vid


def get_leadlag_vid(df, lane_id, vid, before_frame=np.inf, after_frame_id=0, leadlag_key=''):
    # get lead/lag vehicle of vehicle vid in lane lane_id before/after supplied frame
    # - not entirely accurate near vid`s merges
    df_vid = df[(df['Lane_ID'] == lane_id) & (df['Vehicle_ID'] == vid)]
    vid_frames = df_vid[(df_vid['Frame_ID'] < before_frame) & (df_vid['Frame_ID'] > after_frame_id)]
    if len(vid_frames) == 0:
        return np.nan
    first_frame_id = vid_frames['Frame_ID'].values.min()  # for after_frame_id > 0
    if not np.isinf(before_frame):
        first_frame_id = vid_frames['Frame_ID'].values.max()  # use frame closest to before
    leadlag_vid = vid_frames[vid_frames['Frame_ID'] == first_frame_id][leadlag_key].values[0]
    return leadlag_vid


def get_official_merge_frame_id(df, src_lane_id, dest_lane_id, vid):
    # get last moment after vid was in src lane, then make sure the next lane is dest lane
    src_frames = df[(df['Lane_ID'] == src_lane_id) & (df['Vehicle_ID'] == vid)]['Frame_ID'].values
    dest_frames = df[(df['Lane_ID'] == dest_lane_id) & (df['Vehicle_ID'] == vid) &
                     (df['Frame_ID'] > src_frames.max())]['Frame_ID'].values
    if dest_frames.size == 0:
        return -1
    return dest_frames.min()


def is_vid_in_single_lane(df, lane_id, vid, frame_id_bounds):
    frames = np.arange(frame_id_bounds[0], frame_id_bounds[1] + 1)
    df_vid = df[(df['Vehicle_ID'] == vid) & (df['Frame_ID'].isin(frames))]
    lane_ids = df_vid['Lane_ID'].values
    is_all_frames = frame_id_bounds[0] in df_vid['Frame_ID'].values and\
                    frame_id_bounds[1] in df_vid['Frame_ID'].values
    return np.all(lane_ids == lane_id) and is_all_frames


def is_leadlag_pair(df, lead_vid, lag_vid, frame_id_bounds):
    frames = np.arange(frame_id_bounds[0], frame_id_bounds[1]+1)
    df_lead = df[(df['Vehicle_ID'] == lead_vid) & (df['Frame_ID'].isin(frames))]
    df_lag = df[(df['Vehicle_ID'] == lag_vid) & (df['Frame_ID'].isin(frames))]
    is_lead_paired = np.all(df_lead['Following'].values == lag_vid)
    is_lag_paired = np.all(df_lag['Preceding'].values == lead_vid)
    return is_lead_paired and is_lag_paired


def get_shared_frames_df(df, vids):
    df = df[(df['Vehicle_ID'].isin(vids))]
    shared_frames = set(df[(df['Vehicle_ID'] == vids[0])]['Frame_ID'].values)
    for vid in vids[1:]:
        shared_frames.intersection_update(df[(df['Vehicle_ID'] == vid)]['Frame_ID'].values)
    df_shared = df[df['Frame_ID'].isin(shared_frames)]
    return df_shared



