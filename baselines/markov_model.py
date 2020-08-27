import numpy as np
from markov_model_baseline import model
from baselines import baseline_utils as ut


def predict_mm(df, lag_vid, lead_vid, t0, t1, n_steps, difP_yield, difP_noyield, **kwargs):
    n_samples = 1000
    y =  ut.get_positions(df, lag_vid, t0, t1)
    v = ut.get_velocities(df, lag_vid, t0, t1)
    v_hat, p = model.sample_predictions(difP_yield, difP_noyield, v, n_samples, n_steps)
    y_hat = y[-1] + model.DT * v_hat.cumsum(axis=0)
    return y_hat, p, {}


def mock_difP():
    difP_yield = np.ones((model.BOUNDS[1] - model.BOUNDS[0] + 1, 3), dtype=np.float) *\
                 np.array([6, 1, 9])
    difP_yield[0, :] = np.array([0, .5, .5])
    difP_yield[-1, :] = np.array([.5, .5, 0])
    difP_yield = (difP_yield.T / difP_yield.sum(axis=1)).T
    difP_noyield = difP_yield.copy()
    difP_noyield[1:-1, :] = np.array([.3, .1, .6])
    return difP_yield, difP_noyield


def trained_difP():
    from display_driver import main_get_merge_pairs as merge_pairs_generator
    from markov_model_baseline.model import fit_model
    # from utils import DatasetTag
    frames_before_obs = 10  # 32-3
    def v_list_is_merge_generator():
        for (df, lag_vid, lead_vid, t0, t1, t2, yield_dict) in merge_pairs_generator(
                is_display=False, frames_before_obs=frames_before_obs):
            # if tag == DatasetTag.us101:
            #     continue
            is_merge = yield_dict['is_merge']
            v = ut.get_velocities(df, lag_vid, t0, t1)
            yield v, is_merge
    difP_yield, difP_noyield = fit_model(v_list_is_merge_generator)
    print('Finished training MM!')
    return difP_yield, difP_noyield
