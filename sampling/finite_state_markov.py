import numpy as np


def sample_state_v0(x0, P, n_steps=10):
    x = x0
    P_cumulative = P.cumsum(axis=1)
    P_cumulative[:, -1] = 1
    step_i = 0
    while step_i < n_steps:
        states = set(x)
        for state in states:
            mask = x == state
            u = np.random.rand(mask.sum())
            new_states = np.searchsorted(P_cumulative[state, :], u, side='left')
            x[mask] = new_states
        step_i += 1
    return x


def sample_state_differencedP_v0(x0, P, n_steps=10, self_loop_col=-1):
    if self_loop_col < 0:
        assert P.shape[1] % 2, 'no center because differencedP has even number of columns'
        self_loop_col = int((P.shape[1]-1)/2)
    x = x0
    P_cumulative = P.cumsum(axis=1)
    P_cumulative[:, -1] = 1
    step_i = 0
    while step_i < n_steps:
        states = set(x)
        for state in states:
            mask = x == state
            u = np.random.rand(mask.sum())
            new_state_difs = np.searchsorted(P_cumulative[state, :], u, side='left')
            x[mask] += new_state_difs - self_loop_col
        step_i += 1
    return x


def sample_state_history_v0(x0, P, n_steps=10):
    x = x0
    history = [x.copy()]
    P_cumulative = P.cumsum(axis=1)
    P_cumulative[:, -1] = 1
    step_i = 0
    while step_i < n_steps:
        states = set(x)
        for state in states:
            mask = x == state
            u = np.random.rand(mask.sum())
            new_states = np.searchsorted(P_cumulative[state, :], u, side='left')
            x[mask] = new_states
        step_i += 1
        history.append(x.copy())
    return x, history


def sample_state_history_differencedP_v0(x0, P, n_steps=10, self_loop_col=-1):
    if self_loop_col < 0:
        assert P.shape[1] % 2, 'no center because differencedP has even number of columns'
        self_loop_col = int((P.shape[1]-1)/2)
    x = x0
    history = [x.copy()]
    P_cumulative = P.cumsum(axis=1)
    P_cumulative[:, -1] = 1
    step_i = 0
    while step_i < n_steps:
        states = set(x)
        for state in states:
            mask = x == state
            u = np.random.rand(mask.sum())
            new_state_difs = np.searchsorted(P_cumulative[state, :], u, side='left')
            x[mask] += new_state_difs - self_loop_col  # where col of P is the *difference* in the state
        step_i += 1
        history.append(x.copy())
    return x, history


def sample_state_obs_weighting_v0(x0, P, obs, log_ll_fcn, w0=(), n_steps=10):
    # assume observations starts at the second - x0 already weighted/sampled from first
    x = x0
    w = w0 if len(w0) > 0 else np.zeros_like(x, dtype=np.float)
    P_cumulative = P.cumsum(axis=1)
    P_cumulative[:, -1] = 1
    step_i = 0
    while step_i < n_steps:
        states = set(x)
        for state in states:
            mask = x == state
            u = np.random.rand(mask.sum())
            new_states = np.searchsorted(P_cumulative[state, :], u, side='left')
            x[mask] = new_states
        if step_i < obs.size:
            w += log_ll_fcn(x, obs[step_i])
        step_i += 1
    return x, w


def sample_state_obs_weighting_differencedP_v0(x0, P, obs, log_ll_fcn, w0=(), n_steps=10, self_loop_col=-1):
    # assume observations starts at the second - x0 already weighted/sampled from first
    if self_loop_col < 0:
        assert P.shape[1] % 2, 'no center because differencedP has even number of columns'
        self_loop_col = int((P.shape[1]-1)/2)
    x = x0
    w = w0 if len(w0) > 0 else np.zeros_like(x, dtype=np.float)
    P_cumulative = P.cumsum(axis=1)
    P_cumulative[:, -1] = 1
    step_i = 0
    while step_i < n_steps:
        states = set(x)
        for state in states:
            mask = x == state
            u = np.random.rand(mask.sum())
            new_state_difs = np.searchsorted(P_cumulative[state, :], u, side='left')
            x[mask] += new_state_difs - self_loop_col
        if step_i < obs.size:
            w += log_ll_fcn(x, obs[step_i])
        step_i += 1
    return x, w


def sample_state_history_obs_weighting_v0(x0, P, obs, log_ll_fcn, w0=(), n_steps=10):
    # assume observations starts at the second - x0 already weighted/sampled from first
    x = x0
    history = [x.copy()]
    w = w0 if len(w0) > 0 else np.zeros_like(x, dtype=np.float)
    P_cumulative = P.cumsum(axis=1)
    P_cumulative[:, -1] = 1
    step_i = 0
    while step_i < n_steps:
        states = set(x)
        for state in states:
            mask = x == state
            u = np.random.rand(mask.sum())
            new_states = np.searchsorted(P_cumulative[state, :], u, side='left')
            x[mask] = new_states
        if step_i < obs.size:
            w += log_ll_fcn(x, obs[step_i])
        step_i += 1
        history.append(x.copy())
    return x, w, history


#
#      Mains
#
# =================


def main_display_absorbing_rw_differenced_p():
    from scipy.linalg import toeplitz
    n = 1000
    P = toeplitz(np.hstack((1, 1, np.zeros(n-2))), np.hstack((1, 1, np.zeros(n-2))))
    P[0, :] = np.hstack((1, np.zeros(n-1)))
    P[-1, :] = np.hstack((np.zeros(n-1), 1))
    P = (P.T / P.sum(axis=1)).T
    state2y = np.arange(n)/n

    n_samples = 50
    T = 500
    x0 = int(n/2) + np.zeros(n_samples, dtype=np.int)
    np.random.seed(0)
    _, history = sample_state_history_v0(x0.copy(), P, n_steps=T)
    y = np.array([state2y[states_i] for states_i in history])

    P_dif = np.ones((n, 3), dtype=np.float)
    P_dif[0, :] = np.array([0, 1, 0])
    P_dif[-1, :] = np.array([0, 1, 0])
    P_dif = (P_dif.T / P_dif.sum(axis=1)).T
    np.random.seed(0)
    _, history = sample_state_history_differencedP_v0(x0.copy(), P_dif, n_steps=T)
    y_dif = np.array([state2y[states_i] for states_i in history])

    import matplotlib  # for mac
    matplotlib.use('TkAgg')  # for mac
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
    ax[0].plot(y, c='blue', alpha=0.2)
    ax[0].set_title('Standard Transitions')
    ax[1].plot(y_dif, c='blue', alpha=0.2)
    ax[1].set_title('Differenced Transitions')
    plt.show()


def main_display_absorbing_rw_with_observations():
    from scipy.linalg import toeplitz
    n = 1000
    P = toeplitz(np.hstack((1, 1, np.zeros(n-2))), np.hstack((1, 1, np.zeros(n-2))))
    P[0, :] = np.hstack((1, np.zeros(n-1)))
    P[-1, :] = np.hstack((np.zeros(n-1), 1))
    P = (P.T / P.sum(axis=1)).T
    state2y = np.arange(n)/n

    # observation function
    variance = (5*(1/n))**2
    def gaussian_ll(x, y_obs):
        y_state = state2y[x]
        w = -0.5/variance * (y_state - y_obs)**2
        return w

    n_samples = 500
    T = 500
    T_obs = 40
    x0 = int(n/2) + np.zeros(n_samples, dtype=np.int)
    observations = state2y[x0[0]] + .4*np.arange(T_obs)/n
    _, w, history = sample_state_history_obs_weighting_v0(
        x0, P, observations[1:], gaussian_ll, n_steps=T, w0=gaussian_ll(x0, observations[0]))
    y = np.array([state2y[states_i] for states_i in history])
    w_stable = w - w.max()
    p = np.exp(w_stable)
    p /= p.sum()

    import matplotlib  # for mac
    matplotlib.use('TkAgg')  # for mac
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    p_quantiles = np.quantile(p, (0.4, 0.8))
    ax.plot(y[:, p_quantiles[1] <= p], c='blue', alpha=0.1)
    ax.plot(y[:, (p_quantiles[0] <= p) & (p < p_quantiles[1])], c='blue', alpha=0.01)
    ax.plot(y[:, p < p_quantiles[0]], c='blue', alpha=0.001)
    ax.plot(np.arange(T_obs), observations, c='green', alpha=0.8)
    plt.show()


def main_display_absorbing_rw():
    from scipy.linalg import toeplitz
    n = 1000
    P = toeplitz(np.hstack((1, 1, np.zeros(n-2))), np.hstack((1, 1, np.zeros(n-2))))
    P[0, :] = np.hstack((1, np.zeros(n-1)))
    P[-1, :] = np.hstack((np.zeros(n-1), 1))
    P = (P.T / P.sum(axis=1)).T
    state2y = np.arange(n)/n

    n_samples = 50
    T = 500
    x0 = int(n/2) + np.zeros(n_samples, dtype=np.int)
    _, history = sample_state_history_v0(x0, P, n_steps=T)
    y = np.array([state2y[states_i] for states_i in history])

    import matplotlib  # for mac
    matplotlib.use('TkAgg')  # for mac
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(y, c='blue', alpha=0.2)
    plt.show()


if __name__ == '__main__':
    # main_display_absorbing_rw()
    # main_display_absorbing_rw_with_observations()
    main_display_absorbing_rw_differenced_p()
