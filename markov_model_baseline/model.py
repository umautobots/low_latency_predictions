import numpy as np
from sampling import finite_state_markov as fsm

"""
Algorithm detailed in
Dong, et al., "Intention Estimation For Ramp Merging Control In Autonomous Driving". ITSC '17
- markov model
and
Dong, et. al., "Smooth Behavioral Estimation For Ramp Merging Control In Autonomous Driving". IV '18
- markov model just on velocity
- smoothing
"""

MAX_DV = 15
BOUNDS = (0, 45)
DT = 0.1
VARIANCE = 1 * DT  # q*dt, q a parameter
STATE2V = np.arange(BOUNDS[0], BOUNDS[1]+1)  # discretization of 1m/s


def sample_predictions(difP_yield, difP_noyield, observations, n_samples, n_steps):
    # observations of velocity
    assert (n_samples % 2) == 0, 'need even number of samples'
    preds_yield, w_yield = sample_from_difP(difP_yield, observations, int(n_samples/2), n_steps)
    preds_noyield, w_noyield = sample_from_difP(difP_noyield, observations, int(n_samples/2), n_steps)
    preds = np.hstack((preds_yield, preds_noyield))
    w = np.hstack((w_yield, w_noyield))
    w_stable = w - w.max()
    p = np.exp(w_stable)
    p /= p.sum()
    return preds, p


def sample_from_difP(difP, observations, n_samples, n_steps):
    """
    1) draw x0  ~ p(x_1|o_1) = p(o_1|x_1)
    2) draw x_t ~ p(x_t|o_:t,I)
    3) draw x_t+1:T ~ p(x_t+1:T|x_t,I)
    4) draw o_t+1:T ~ p(o_t+1:T|x_t+1:T,I) = p(o_t+1:T|x_t+1:T)
    :param difP: 
    :param observations: 
    :param n_samples: 
    :param n_steps: 
    :return: 
    """
    def gaussian_ll(x, y_obs):
        y_state = STATE2V[x]
        w = -0.5 / VARIANCE * (y_state - y_obs) ** 2
        return w

    x0 = observations[0] + np.sqrt(VARIANCE) * np.random.randn(n_samples)
    states_t, w = fsm.sample_state_obs_weighting_differencedP_v0(
        np.round(x0).astype(np.int), difP, observations[1:],
        gaussian_ll, n_steps=observations.size-1)
    x_t = STATE2V[states_t]
    _, history = fsm.sample_state_history_differencedP_v0(
        states_t.copy(), difP, n_steps=n_steps
    )
    x_pred = np.array([STATE2V[states_i] for states_i in history[1:]])
    o_pred = x_pred + np.sqrt(VARIANCE) * np.random.randn(*x_pred.shape)
    return o_pred, w


def fit_model(generator):
    max_dv = MAX_DV
    bounds = BOUNDS
    dv = np.arange(-max_dv, max_dv+1.)
    # p(vj|vi) = p(vj - vi|vi) = p_ij
    counts_yield = np.zeros((bounds[1] - bounds[0] + 1, dv.size))
    counts_noyield = counts_yield.copy()
    for v_list, is_merge in generator():
        counts = counts_yield if is_merge else counts_noyield
        for (vi, dvi) in zip(v_list[:-1], v_list[1:] - v_list[:-1]):
            i, j = int(vi), int(dvi)
            assert bounds[0] <= i <= bounds[1], 'v {} out of bounds {},{}'.format(i, bounds[0], bounds[1])
            assert -max_dv <= j <= max_dv, 'dv {} out of bounds, {},{}'.format(j, -max_dv, max_dv)
            counts[i, j] += 1

    def normalize(counts):
        # and prevent it from exiting
        counts[0, :max_dv] = 0
        counts[-1, max_dv + 1:] = 0
        # all 0s - make self loop
        counts[(counts == 0).all(axis=1)] = np.hstack((np.zeros(max_dv), 1, np.zeros(max_dv)))
        sums = np.sum(counts, axis=1)
        sums[sums == 0] = 1.  # avoid 0/0
        return (counts.T / sums).T
    difP_yield = normalize(counts_yield)
    difP_noyield = normalize(counts_noyield)
    return difP_yield, difP_noyield
