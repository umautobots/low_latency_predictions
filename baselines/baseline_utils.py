import numpy as np


def get_positions(df, vid, t0, t1):
    return get_vid_key(df, vid, t0, t1, 'Local_Y')


def get_velocities(df, vid, t0, t1):
    return get_vid_key(df, vid, t0, t1, 'v_Vel')


def get_accel(df, vid, t0, t1):
    return get_vid_key(df, vid, t0, t1, 'v_Acc')


def get_spacing(df, vid, t0, t1):
    return get_vid_key(df, vid, t0, t1, 'Space_Headway')


def get_time_spacing(df, vid, t0, t1):
    return get_vid_key(df, vid, t0, t1, 'Time_Headway')


def get_length(df, vid):
    vid_df = df[df['Vehicle_ID'] == vid]
    return vid_df['v_Length'].mean()


def get_vid_key(df, vid, t0, t1, key):
    frames = np.arange(t0, t1)
    lag_df = df[df['Vehicle_ID'] == vid]
    return lag_df[lag_df['Frame_ID'].isin(frames)][key].values


def single_prediction2probabilistic_format(y_hat):
    """
    Convert single-dim prediction to (timesteps, sample) array
    Also return single probability weighting
    :param y_hat: prediction for timesteps [hat_y_t, ..., hat_y_T]
    :return: y_hat, p
    """
    return np.expand_dims(y_hat, 1), np.array([1.])
