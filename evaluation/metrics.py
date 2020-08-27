import numpy as np
from baselines import baseline_utils as ut


def evaluate_metric_on_eval_results_list(
        eval_results_list, init_fcn, accumulate_fcn,
        reduce_fcn, select_fcn=None, select_fcn_kwargs_list=(), **kwargs):
    n_eval = 0
    accumulator = init_fcn()
    for i, data in enumerate(eval_results_list):
        dict_data = data[-1]
        select_fcn_kwargs = select_fcn_kwargs_list[i] if select_fcn_kwargs_list else {}
        if select_fcn and not select_fcn(**select_fcn_kwargs, **kwargs):
            continue
        n_eval += 1
        accumulate_fcn(accumulator, *data[:-1], **dict_data)
    metric_value = reduce_fcn(accumulator)
    return metric_value, n_eval


def get_expected_dist_fcns():
    # different from average of E[d_t],.. when number of examples per forecast horizon can vary
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        :param accumulator: list of average distances from previous examples
        :param y_hats: n_steps, n_samples
        :param p: n_samples, | probabilities summing to 1
        :param y_true: n_steps
        :return: 
        """
        dist = np.sqrt(((y_hats.T - y_true) ** 2).sum(axis=1))
        expected_dist = p.dot(dist)
        accumulator.append(expected_dist)

    def reduce_fcn(accumulator):
        return np.mean(accumulator)
    return init_fcn, accumulate_fcn, reduce_fcn


def get_prob_of_inclusion_fcns(n_bins=20):
    """
    p(y_hat \in B_{y_hat}(distance)) as a function of distance
    :param n_bins: for discretization of distance (rightmost bin as open)
    :return: 
    """
    last_bin_left_edge = 20.  # (m)
    bin_right_edges = np.hstack((np.linspace(0., last_bin_left_edge, num=n_bins)[1:], np.inf))
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        dist = np.sqrt(((y_hats.T - y_true) ** 2).sum(axis=1))
        inds = np.argsort(dist)
        sorted_dists = dist[inds]
        sorted_p = p[inds]
        bin_vals = np.zeros(n_bins)
        for i in range(n_bins):
            mask = sorted_dists < bin_right_edges[i]
            bin_vals[i] = sorted_p[mask].sum()
        accumulator.append(bin_vals)

    def reduce_fcn(accumulator):
        # n_examples, n_bins -> n_bins
        return np.mean(accumulator, axis=0)
    return init_fcn, accumulate_fcn, reduce_fcn


def get_prob_of_inclusion_at_time_fcns(bins=20, t=10):
    """
    p(y_hat \in B_{y_hat}(distance)) as a function of distance
    :param bins: given to histogram for discretizing distance
    :param t: index of prediction to use
    :return: 
    """
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        if y_true.size <= t:
            return
        dist = np.sqrt(((y_hats[t, :] - y_true[t]) ** 2))
        hist, _ = np.histogram(dist, bins=bins, weights=p)
        accumulator.append(hist.cumsum())

    def reduce_fcn(accumulator):
        # n_examples, n_bins -> n_bins
        print('size: ', np.array(accumulator).shape)
        return np.mean(accumulator, axis=0)
    return init_fcn, accumulate_fcn, reduce_fcn


def get_expected_dist_by_time_fcns(n_steps=100, select_inds=np.arange(9, 100, 10)):
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        :param accumulator: list of average distances from previous examples
        :param y_hats: n_steps, n_samples
        :param p: n_samples, | probabilities summing to 1
        :param y_true: n_steps
        :return: 
        """
        dists = np.sqrt(((y_hats.T - y_true) ** 2))
        expected_dist = dists.T.dot(p)
        expected_dists_cut = np.zeros(n_steps) - 1
        k = min(n_steps, expected_dist.size)
        expected_dists_cut[:k] = expected_dist[:k]
        accumulator.append(expected_dists_cut)

    def reduce_fcn(accumulator):
        expected_dists_measured = np.array(accumulator)  # examples, n_steps
        expected_dists = np.zeros(n_steps)
        for i in range(n_steps):
            mask = expected_dists_measured[:, i] >= 0
            expected_dists[i] = expected_dists_measured[mask, i].sum()/mask.sum()
        return expected_dists[select_inds]
    return init_fcn, accumulate_fcn, reduce_fcn


def get_rmse_dist_by_time_fcns(n_steps=100, select_inds=np.arange(9, 100, 10)):
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        :param accumulator: list of average distances from previous examples
        :param y_hats: n_steps, n_samples
        :param p: n_samples, | probabilities summing to 1
        :param y_true: n_steps
        :return: 
        """
        dists_sq = (y_hats.T - y_true) ** 2
        expected_dist_sq = dists_sq.T.dot(p)
        expected_dists_cut = np.zeros(n_steps) - 1
        k = min(n_steps, expected_dist_sq.size)
        expected_dists_cut[:k] = expected_dist_sq[:k]
        accumulator.append(expected_dists_cut)

    def reduce_fcn(accumulator):
        expected_dists_measured = np.array(accumulator)  # examples, n_steps
        expected_dists = np.zeros(n_steps)
        for i in range(n_steps):
            mask = expected_dists_measured[:, i] >= 0
            expected_dists[i] = np.sqrt(expected_dists_measured[mask, i].sum()/mask.sum())
        return expected_dists[select_inds]
    return init_fcn, accumulate_fcn, reduce_fcn


def get_calibration_metric_by_time_fcns(select_inds=np.arange(9, 100, 10)):
    """
    "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
    https://arxiv.org/pdf/1807.00263.pdf
    equation 9
    - calculate this for each selected timestep
    - F is cdf -> need to calculate empirical cdf = pred_cdf 
    :param select_inds: 
    :return: 
    """
    quantiles = np.linspace(0, 1, num=11)[1:-1]
    init_fcn = lambda: [np.zeros((quantiles.size, select_inds.size)),
                        np.zeros((select_inds.size,))]

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        - calculate pred_cdf(y) = sum_{i=1...n_samples}p_i * indicator{y_{pred i} <= y}
        :param accumulator:
            m, k | running sums of |{y|pred_cdf(y) < p_j}| for each quantile p_j
                at each of k selected timesteps
            k, | total number of predictions y used in the sums, for each timestep
        :param y_hats: n_steps, n_samples
        :param p: n_samples, | probabilities summing to 1
        :param y_true: n_steps
        :return: 
        """
        ind_last = min(select_inds[-1], y_true.size-1)
        inds = select_inds[select_inds <= ind_last]
        accumulator[1][:inds.size] += 1
        y_hats_lt_y = (y_hats[inds, :].T <= y_true[inds]).T  # |inds|, n_samples
        empirical_cdf_at_true = (y_hats_lt_y * p).sum(axis=1)  # |inds|
        # m, |inds|
        ecdf_lt_quantile = empirical_cdf_at_true[np.newaxis, :] < quantiles[:, np.newaxis]
        accumulator[0][:, :inds.size] += ecdf_lt_quantile

    def reduce_fcn(accumulator):
        print(accumulator[0])
        print(accumulator[1])
        q_hats = accumulator[0]/accumulator[1]
        dif = (q_hats.T - quantiles).T
        return (dif**2).sum(axis=0)
    return init_fcn, accumulate_fcn, reduce_fcn


def get_calibration_metric_fcns(select_inds=np.arange(9, 100, 10)):
    quantiles = np.linspace(0, 1, num=11)[1:-1]
    init_fcn = lambda: [0*quantiles, 0]

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        - calculate pred_cdf(y) = sum_{i=1...n_samples}p_i {y_{pred i} - y}
        :param accumulator:
            m, | running sums of |{y|pred_cdf(y) < p_j}| for each quantile p_j
            scalar | total number of predictions y used in the sums
        :param y_hats: n_steps, n_samples
        :param p: n_samples, | probabilities summing to 1
        :param y_true: n_steps
        :return: 
        """
        ind_last = min(select_inds[-1], y_true.size-1)
        inds = select_inds[select_inds <= ind_last]
        accumulator[1] += inds.size
        y_hats_lt_y = (y_hats[inds, :].T <= y_true[inds]).T  # |inds|, n_samples
        empirical_cdf_at_true = (y_hats_lt_y * p).sum(axis=1)  # |inds|
        # m, |inds|
        ecdf_lt_quantile = empirical_cdf_at_true[np.newaxis, :] < quantiles[:, np.newaxis]
        accumulator[0] += ecdf_lt_quantile.sum(axis=1)

    def reduce_fcn(accumulator):
        return accumulator[0]/accumulator[1]
    return init_fcn, accumulate_fcn, reduce_fcn


def is_far_fcn(df=None, lag_vid=np.nan, t0=np.nan, t1=np.nan, **kwargs):
    assert df.size > 0
    g = ut.get_spacing(df, lag_vid, t0, t1)
    ts = ut.get_time_spacing(df, lag_vid, t0, t1)
    g_far = 20  # m
    ts_far = 2.0  # s
    return (g >= g_far).all() and (ts >= ts_far).all()


def get_is_merge_select_fcn(select_mode):
    if select_mode == 'all':
        select_fcn = (lambda is_merge=False, **kwargs: True)
    elif select_mode == 'true':
        select_fcn = (lambda is_merge=False, **kwargs: is_merge)
    elif select_mode == 'false':
        select_fcn = (lambda is_merge=False, **kwargs: not is_merge)
    else:
        raise ValueError('select_mode not one of all/true/false')
    return select_fcn


def get_effective_sample_size_fcns():
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        ESS = 1 / sum(p_i^2)
        from (9.13) in Owen A., Monte Carlo Theory, Methods, and Examples.
        :param accumulator: list of ess for each example
        :param y_hats: n_steps, n_samples
        :param p: n_samples, | probabilities summing to 1
        :param y_true: n_steps
        :return: 
        """
        sum_p_sq = (p ** 2).sum()
        ess = 1.0 / sum_p_sq
        accumulator.append(ess)

    def reduce_fcn(accumulator):
        accumulator = np.array(accumulator)  # examples,
        # return np.mean(accumulator)
        n_quantiles = 10
        q = np.arange(n_quantiles, dtype=np.float)/n_quantiles
        return np.quantile(accumulator, q)
    return init_fcn, accumulate_fcn, reduce_fcn


def get_f_theta_distribution(n_quantiles=10):
    init_fcn = lambda: []
    q = np.arange(n_quantiles, dtype=np.float) / n_quantiles

    def accumulate_fcn(accumulator, y_hats, p, y_true, f_thetas=(), **kwargs):
        """
        median instead of mean in case of all [inf ...]
        :param accumulator: list of ess for each example
        :param y_hats: n_steps, n_samples
        :param p: n_samples, | probabilities summing to 1
        :param y_true: n_steps,
        :param f_thetas: n_samples, | if method provided its f_thetas
        :return: 
        """
        if len(f_thetas) == 0:
            return
        accumulator.append(np.quantile(f_thetas, q))

    def reduce_fcn(accumulator):
        accumulator = np.array(accumulator)  # examples, n_quantiles
        if accumulator.size == 0:
            return -1
        return np.median(accumulator, axis=0)
    return init_fcn, accumulate_fcn, reduce_fcn


def get_f_theta_gap_distribution(gaps=(), gap_step=0.2, gap_max=10):
    init_fcn = lambda: []
    gaps = gaps if len(gaps) > 0 else np.arange(0, gap_max, gap_step) + gap_step

    def accumulate_fcn(accumulator, y_hats, p, y_true, f_thetas=(), f_theta_min=np.nan, **kwargs):
        """
        median instead of mean in case of all [inf ...]
        :param accumulator: list of ess for each example
        :param y_hats: n_steps, n_samples
        :param p: n_samples, | probabilities summing to 1
        :param y_true: n_steps,
        :param f_thetas: n_samples, | if method provided its f_thetas
        :return: 
        """
        if len(f_thetas) == 0:
            return
        dif = f_thetas - f_theta_min
        p_less_gap = 0.*gaps
        for i in range(gaps.size):
            p_less_gap[i] = (dif <= gaps[i]).sum()/dif.size
        accumulator.append(p_less_gap)

    def reduce_fcn(accumulator):
        accumulator = np.array(accumulator)  # examples, n_gaps
        if accumulator.size == 0:
            return -1
        return np.mean(accumulator, axis=0)
    return init_fcn, accumulate_fcn, reduce_fcn
