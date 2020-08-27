import numpy as np
from baselines import baseline_utils as ut


def predict_constant_velocity(df, lag_vid, lead_vid, t0, t1, n_steps, **kwargs):
    y =  ut.get_positions(df, lag_vid, t0, t1)
    vdt = y[1:] - y[:-1]
    mean_vdt = vdt.mean()
    y_hat = y[-1] + mean_vdt * np.arange(1, n_steps+1)
    return (*ut.single_prediction2probabilistic_format(y_hat)), {}

