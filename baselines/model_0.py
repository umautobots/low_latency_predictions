import numpy as np
from baselines import baseline_utils as ut
import scipy.optimize as opt
import scipy.stats as ss
import cvxopt as cvx
import picos as pic

DT = 0.1
K_UB = 0.5
EPS = 1e-3


def predict_no_sampling(df, lag_vid, lead_vid, t0, t1, n_steps, **kwargs):
    # make A, 'b'
    # solve for parameters [kv, kg, kg*g*]
    # predict with parameters
    v = ut.get_velocities(df, lag_vid, t0, t1)
    vl = ut.get_velocities(df, lead_vid, t0, t1)
    g = ut.get_spacing(df, lag_vid, t0, t1)
    A = np.array([vl - v, g, 0*g - 1]).T  # n, 3
    b = ut.get_accel(df, lag_vid, t0, t1)
    print(np.linalg.cond(A))
    params = solve_ls(A, b)
    kv, kg, g_star = params[0], params[1], params[2]/params[1]

    y = ut.get_positions(df, lag_vid, t0, t1)
    yl = ut.get_positions(df, lead_vid, t0, t1)
    vl_hat = np.zeros(n_steps) + np.mean(vl)
    y_hat, _ = apply_model_v0(y[-1], v[-1], yl[-1], vl_hat, kv, kg, g_star)
    return (*ut.single_prediction2probabilistic_format(y_hat)), {}


def predict_sampling(df, lag_vid, lead_vid, t0, t1, n_steps, **kwargs):
    n_samples = 1000
    v = ut.get_velocities(df, lag_vid, t0, t1)
    vl = ut.get_velocities(df, lead_vid, t0, t1)
    g = ut.get_spacing(df, lag_vid, t0, t1)
    A = np.array([vl - v, g, 0*g - 1]).T  # n, 3
    b = ut.get_accel(df, lag_vid, t0, t1)
    params = solve_ls(A, b)
    kv, kg, g_star = params[0], params[1], params[2]/params[1]
    theta_hat = np.array([kv, kg, g_star])
    sigmas = np.array([.1, .1, 1])
    kv_kg_gs, p, w_f = importance_sample_thetas_no_reg(theta_hat, sigmas, A, b, n_samples)
    y = ut.get_positions(df, lag_vid, t0, t1)
    yl = ut.get_positions(df, lead_vid, t0, t1)
    vehicle_length = yl[-1] - y[-1] - g[-1]
    xv = apply_model_v1(y[-1], v[-1], yl[-1], np.mean(vl), vehicle_length, kv_kg_gs, n_steps)
    y_hats = xv[:, 0, :]
    return y_hats, p, {}


def predict_sampling_propagate(df, lag_vid, lead_vid, t0, t1, n_steps, leadlead_vid=np.nan, **kwargs):
    n_samples = 100
    if np.isnan(leadlead_vid):
        return predict_sampling(df, lag_vid, lead_vid, t0, t1, n_steps)
    y_hats_lead, p_lead = predict_sampling(df, lead_vid, leadlead_vid, t0, t1, n_steps)[:2]
    yl = ut.get_positions(df, lead_vid, t0, t1)
    vl_hats = y_hats_lead[1:, :] - y_hats_lead[:-1, :]
    vl_hats = np.vstack((y_hats_lead[0, :] - yl[-1], vl_hats)) / DT

    v = ut.get_velocities(df, lag_vid, t0, t1)
    vl = ut.get_velocities(df, lead_vid, t0, t1)
    g = ut.get_spacing(df, lag_vid, t0, t1)
    A = np.array([vl - v, g, 0*g - 1]).T  # n, 3
    b = ut.get_accel(df, lag_vid, t0, t1)
    params = solve_ls(A, b)
    kv, kg, g_star = params[0], params[1], params[2]/params[1]

    theta_hat = np.array([kv, kg, g_star])
    sigmas = np.array([.1, .1, 1])
    kv_kg_gs, p, w_f1 = importance_sample_thetas_no_reg(theta_hat, sigmas, A, b, n_samples)
    p *= p_lead
    p /= p.sum()
    y_hats = np.zeros((n_steps, n_samples), dtype=np.float)
    y = ut.get_positions(df, lag_vid, t0, t1)
    for i in range(n_samples):
        y_hats[:, i], _ = apply_model_v0(y[-1], v[-1], yl[-1], vl_hats[:, i], *kv_kg_gs[i, :])
    return y_hats, p, {}


def predict_sampling_gprior(df, lag_vid, lead_vid, t0, t1, n_steps, **kwargs):
    n_samples = 100
    alpha = 1./50**2
    v = ut.get_velocities(df, lag_vid, t0, t1)
    vl = ut.get_velocities(df, lead_vid, t0, t1)
    g = ut.get_positions(df, lead_vid, t0, t1) - ut.get_positions(df, lag_vid, t0, t1)
    g0 = g.mean()
    A = np.array([vl - v, g, 0*g - 1]).T[:-1, :]  # k-1, 3
    Ap = np.vstack((np.hstack((A, 0*A[:, [0]])), np.array([0, 0, 0, np.sqrt(alpha)])))  # n, 4
    b = (v[1:] - v[:-1])/DT
    bp = np.hstack((b, np.sqrt(alpha)*g0))
    if np.linalg.matrix_rank(A) < A.shape[1]:
        print('bad rank')
        params = np.array([0., 0, 0, g0])
    else:
        params = solve_sdp(Ap, bp)
    kv, kg, g_star = params[0], params[1], params[3]
    print(kv, kg, g_star, np.linalg.cond(A))
    print(g.mean())
    # print(ut.get_time_spacing(df, lag_vid, t0, t1).mean())

    theta_hat = np.array([kv, kg, g_star])
    sigmas = np.array([.1, .1, 1])
    kv_kg_gs, p, w_f1 = importance_sample_thetas(theta_hat, sigmas, Ap, bp, n_samples)
    y = ut.get_positions(df, lag_vid, t0, t1)
    yl = ut.get_positions(df, lead_vid, t0, t1)
    vehicle_length = yl[-1] - y[-1] - g[-1]
    xv = apply_model_v1(y[-1], v[-1], yl[-1], np.mean(vl), vehicle_length, kv_kg_gs, n_steps)
    y_hats = xv[:, 0, :]
    p, w_f2 = post_weight1_samples(y_hats, p, v[-1])
    f_thetas = w_f1 + w_f2
    return y_hats, p, dict(f_thetas=f_thetas)


def predict_sampling_ess_baseline(df, lag_vid, lead_vid, t0, t1, n_steps, n_samples=100, **kwargs):
    # n_samples = 10000
    alpha = 1. / 50 ** 2
    v = ut.get_velocities(df, lag_vid, t0, t1)
    vl = ut.get_velocities(df, lead_vid, t0, t1)
    g = ut.get_spacing(df, lag_vid, t0, t1)
    g0 = g.mean()
    A = np.array([vl - v, g, 0*g - 1]).T[:-1,:]  # k-1, 3
    b = (v[1:] - v[:-1]) / DT

    # baseline monte carlo sampler
    kv_kg_gs = np.zeros((n_samples, 3), dtype=np.float)
    kv_kg_gs[:, :2] = np.random.rand(n_samples, 2) * (K_UB - EPS) + EPS
    sigma_a = 1/np.sqrt(2*alpha)
    kv_kg_gs[:, 2] = ss.truncnorm.rvs((EPS - g0)/sigma_a, np.inf, size=n_samples) * sigma_a + g0

    p = np.ones(n_samples) / n_samples
    y = ut.get_positions(df, lag_vid, t0, t1)
    yl = ut.get_positions(df, lead_vid, t0, t1)
    vehicle_length = yl[-1] - y[-1] - g[-1]
    xv = apply_model_v1(y[-1], v[-1], yl[-1], np.mean(vl), vehicle_length, kv_kg_gs, n_steps)
    y_hats = xv[:, 0, :]

    thetas = kv_kg_gs.copy()
    thetas[:, 2] *= thetas[:, 1]
    r = (A.dot(thetas.T)).T - b  # n_samples, b.size
    w_f1 = np.sqrt((r ** 2).sum(axis=1))
    p, w_f2 = post_weight1_samples(y_hats, p, v[-1])
    f_thetas = w_f1 + w_f2
    return y_hats, p, dict(f_thetas=f_thetas)


def importance_sample_thetas(theta_hat, sigmas, A, b, n_samples):
    dim = theta_hat.size
    rand_bounds = (np.array([[EPS, EPS, EPS], [K_UB, K_UB, np.inf]]) - theta_hat)/sigmas
    rands = np.vstack((0 * theta_hat, ss.truncnorm.rvs(
        rand_bounds[0], rand_bounds[1], size=(n_samples-1, dim))))
    w_normal = np.sum(rands**2, axis=1)
    thetas = rands * sigmas + theta_hat  # n_samples, dim
    kv_kg_gs = thetas.copy()
    thetas[:, 2] *= thetas[:, 1]
    thetas = np.hstack((thetas, kv_kg_gs[:, [-1]]))
    r = (A.dot(thetas.T)).T - b  # n_samples, b.size
    w_f1 = np.sqrt((r ** 2).sum(axis=1))
    w = -w_f1 + w_normal  # log(-nll_f - -n_ll_sample) = p_f/p_sample
    w_stable = w - w.max()
    p = np.exp(w_stable)
    p /= p.sum()
    return kv_kg_gs, p, w_f1


def importance_sample_thetas_no_reg(theta_hat, sigmas, A, b, n_samples):
    # no regularization terms
    dim = theta_hat.size
    rand_bounds = (np.array([[EPS, EPS, EPS], [K_UB, K_UB, np.inf]]) - theta_hat)/sigmas
    rands = np.vstack((0 * theta_hat, ss.truncnorm.rvs(
        rand_bounds[0], rand_bounds[1], size=(n_samples-1, dim))))
    w_normal = np.sum(rands**2, axis=1)
    thetas = rands * sigmas + theta_hat  # n_samples, dim
    kv_kg_gs = thetas.copy()
    thetas[:, 2] *= thetas[:, 1]
    r = (A.dot(thetas.T)).T - b  # n_samples, b.size
    w_f1 = np.sqrt((r ** 2).sum(axis=1))
    w = -w_f1 + w_normal  # log(-nll_f - -n_ll_sample) = p_f/p_sample
    w_stable = w - w.max()
    p = np.exp(w_stable)
    p /= p.sum()
    return kv_kg_gs, p, w_f1


def apply_model_v0(x_0, v_0, xl_0, vl_0, vehicle_length, kv_kg_gs, n_steps):
    n_samples = kv_kg_gs.shape[0]
    vl = np.zeros(n_steps) + vl_0
    states = np.zeros((n_steps, 2, n_samples))  # state = [x, v]
    for i in range(n_samples):
        states[:, 0, i], v_i = apply_model_single(x_0, v_0, xl_0, vl, vehicle_length, *kv_kg_gs[i, :])
        states[:, 1, i] = v_i
    return states


def apply_model_single(x_0, v_0, xl_0, vl, vehicle_length, kv, kg, g_star):
    x = np.zeros(vl.size)
    v = np.zeros(vl.size)
    x_c, v_c, xl_c = x_0, v_0, xl_0
    for i in range(vl.size):
        g_t = xl_c - x_c - vehicle_length
        a_t = kv * (vl[i] - v_c) + kg * (g_t - g_star)
        xl_c += vl[i] * DT
        v[i] = v_c + a_t * DT
        x[i] = x_c + v_c * DT + 0.5 * a_t * DT **2
        x_c, v_c = x[i], v[i]
    return x, v


def apply_model_v1(x_0, v_0, xl_0, vl_0, vehicle_length, kv_kg_gs, n_steps):
    # kv_kg_gs: n_samples, 3
    # ---
    # B = N,n,p
    # x = p,N
    # np.einsum('ijk,ij->ij',B,x.T).T  == np.einsum('ijk,ji->ji',B,x)
    # same as
    # for each i=0...N-1, B[i, ...].dot(x[:, i])
    n_samples = kv_kg_gs.shape[0]
    xc = np.array([x_0, v_0])
    xc = np.repeat(xc[:, np.newaxis], n_samples, axis=1)  # 2, n_samples
    wc = np.array([xl_0, vl_0])
    wc = np.repeat(wc[:, np.newaxis], n_samples, axis=1)
    xs = np.zeros((n_steps, 2, n_samples))
    A_base = np.array([[1, DT], [0, 1]])
    S = np.array([[1, DT], [0, 1]])
    for i in range(n_steps):
        a_ego = -kv_kg_gs[:, 1] * xc[0, :] - kv_kg_gs[:, 0] * xc[1, :]  # n_samples,
        Axc_ego = np.array([DT**2/2 * a_ego, DT * a_ego])  # 2, n_samples
        a_lead = kv_kg_gs[:, 1] * wc[0, :] + kv_kg_gs[:, 0] * wc[1, :] +\
                 -kv_kg_gs[:, 1] * (vehicle_length + kv_kg_gs[:, 2])
        Bwc = np.array([DT**2/2 * a_lead, DT * a_lead])
        xs[i, ...] = A_base.dot(xc) + Axc_ego + Bwc
        xc = xs[i, ...]
        wc = S.dot(wc)
    return xs


def solve_ls(A, b):
    x0 = np.array([0.5, 0.5, 0.5 * 10])
    AtA = A.T.dot(A)
    Atb = A.T.dot(b)
    bounds = [(EPS, K_UB), (EPS, K_UB), (EPS, None)]
    res = opt.minimize(
        lambda x: 0.5 * np.linalg.norm(A.dot(x) - b) ** 2,
        x0,
        jac=(lambda x, *args: AtA.dot(x) - Atb),
        bounds=bounds,
        options=dict(maxiter=10),
    )
    return res.x


def solve_sdp(A, b):
    # x = [kv, kg, u, g]
    # kg*g - u = 0
    m = A.shape[1]
    B = np.zeros((m, m))
    B[1, 3] = B[3, 1] = 1
    q = np.array([0, 0, -1, 0])
    A, b, B, q = [cvx.matrix(arr.copy()) for arr in [A, b, B, q]]

    prob = pic.Problem()
    x = prob.add_variable('x', m)
    X = prob.add_variable('X', (m, m), vtype='symmetric')
    prob.add_constraint(x >= EPS)
    prob.add_constraint(x[:2] <= K_UB)
    # [[1, x']; [x, X]] > 0
    prob.add_constraint(((1 & x.T) // (x & X)) >> 0)
    prob.add_constraint(0.5*(X | B) + q.T*x  == 0)

    prob.set_objective('min', (X | A.T * A) - 2 * b.T * A * x)
    # from pprint import pprint
    # print(prob)
    try:
        prob.solve(verbose=0, solver='cvxopt', solve_via_dual=False, feastol=EPS/2)
    except ValueError:
        print(A)
        print('retrying')
        prob.solve(verbose=0, solver='cvxopt', solve_via_dual=False, feastol=0.1)
    x_hat = np.array(x.value).T[0]
    assert (x_hat >= EPS-1e-2).all() and (x_hat[:2] <= K_UB+1e-2).all(), '{}'.format(x)
    return x_hat


def post_weight_samples(y_hats, p, weight_fcn, **kwargs):
    for i in range(p.size):
        p[i] *= weight_fcn(y_hats[:, i], **kwargs)
    assert p.sum() > 0
    p /= p.sum()
    return p


def weight0(y_hat, v_mean=np.nan):
    # inf norm on velocity
    beta = 0.5
    inf_norm = np.max(np.abs((y_hat[1:] - y_hat[:-1])/DT - v_mean))
    pw = np.exp(-beta*inf_norm)
    return pw


def weight1(y_hat, v_0=np.nan):
    # inf norm on accel
    beta = 2.0
    if y_hat.size < 2:
        return 1
    v_hat = np.hstack((v_0, (y_hat[1:] - y_hat[:-1])/DT))
    inf_norm = np.max(np.abs(v_hat[1:] - v_hat[:-1]))/DT
    pw = np.exp(-beta * inf_norm)
    return pw


def post_weight1_samples(y_hats, p, v_0):
    beta = 15.0
    if y_hats.shape[0] < 2:
        return p, 0*p + 1
    v_hat = np.vstack((v_0 + np.zeros(y_hats.shape[1]),
                       (y_hats[1:, :] - y_hats[:-1, :]) / DT))
    inf_norm = np.max(np.abs(v_hat[1:, :] - v_hat[:-1, :]), axis=0) / DT
    w_f2 = beta * inf_norm
    up_w = np.exp(-w_f2)
    v_check_mask = np.any(v_hat < 0, axis=0)
    w_f2[v_check_mask] += 1e12
    up_w[v_check_mask] *= 1e-12  # effectively 0
    p *= up_w
    assert p.sum() > 0
    p /= p.sum()
    return p, w_f2


def post_weight1_samples_w(y_hats, w, v_0):
    beta = 15.0
    if y_hats.shape[0] < 2:
        return w
    v_hat = np.vstack((v_0 + np.zeros(y_hats.shape[1]),
                       (y_hats[1:, :] - y_hats[:-1, :]) / DT))
    inf_norm = np.max(np.abs(v_hat[1:, :] - v_hat[:-1, :]), axis=0) / DT
    w += beta * inf_norm
    v_check_mask = np.any(v_hat < 0, axis=0)
    w[v_check_mask] += 1e12  # p = 0
    assert np.any(1-v_check_mask)
    return w


def post_weight2_samples(y_hats, p, v_0, a_fit):
    # weight all, including fit terms
    beta = 15.0
    if y_hats.shape[0] < 2:
        a_all = a_fit
    else:
        v_hat = np.vstack((v_0 + np.zeros(y_hats.shape[1]),
                           (y_hats[1:, :] - y_hats[:-1, :]) / DT))
        a_pred = (v_hat[1:, :] - v_hat[:-1, :]) / DT
        a_all = np.vstack((a_fit, a_pred))
        v_check_mask = np.any(v_hat < 0, axis=0)
        p[v_check_mask] *= 1e-12  # effectively 0
    inf_norm = np.max(np.abs(a_all), axis=0)
    pw = np.exp(-beta * inf_norm)
    p *= pw
    assert p.sum() > 0
    p /= p.sum()
    return p


def nll2p(w):
    w = -w  # log(-nll) = p
    w_stable = w - w.max()
    p = np.exp(w_stable)
    p /= p.sum()
    return p
