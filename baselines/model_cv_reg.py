import numpy as np
from baselines import baseline_utils as ut
import scipy.stats as ss
import cvxopt as cvx
import picos as pic

DT = 0.1
K_UB = 20
EPS = 1e-5


def predict_sampling_gprior(df, lag_vid, lead_vid, t0, t1, n_steps, **kwargs):
    n_samples = 1000
    alpha = 1.
    v = ut.get_velocities(df, lag_vid, t0, t1)
    vl = ut.get_velocities(df, lead_vid, t0, t1)
    g = ut.get_positions(df, lead_vid, t0, t1) - ut.get_positions(df, lag_vid, t0, t1)
    g0 = g.mean()
    A = np.array([vl - v, g, 0*g - 1]).T[:-1, :]  # k-1, 3
    Ap = np.vstack((np.hstack((A, 0*A[:, [0]])), np.array([0, 0, 0, np.sqrt(alpha)])))  # n, 4
    b = (v[1:] - v[:-1])/DT
    bp = np.hstack((b, np.sqrt(alpha) * g0))

    # ------ CV trajectory regularization ------
    beta = 1.
    Ap = np.vstack((Ap, np.zeros((2, 4))))
    Ap[-1, 0] = np.sqrt(beta) * g0
    Ap[-1, 1] = np.sqrt(beta) * g0
    bp = np.hstack((bp, 0, 0))

    if np.linalg.matrix_rank(A) < A.shape[1]:
        # print('bad rank')
        params = np.array([0., 0, 0, g0])
    else:
        params = solve_sdp(Ap, bp)
    kv, kg, g_star = params[0], params[1], params[3]
    # print(kv, kg, g_star, np.linalg.cond(A))
    # print(g.mean())

    theta_hat = np.array([kv, kg, g_star])
    sigmas = np.array([.5, .5, 5]) * 2  # doesn't change much with this (if var enough, >=*1) - ie sampled well
    kv_kg_gs, p, w_f1 = importance_sample_thetas(theta_hat, sigmas, Ap, bp, n_samples)
    y = ut.get_positions(df, lag_vid, t0, t1)
    yl = ut.get_positions(df, lead_vid, t0, t1)
    vehicle_length = yl[-1] - y[-1] - g[-1]
    xv = apply_model_v1(y[-1], v[-1], yl[-1], np.mean(vl), vehicle_length, kv_kg_gs, n_steps)
    y_hats = xv[:, 0, :]
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
    # prob.add_constraint(x[:2] <= K_UB)
    # [[1, x']; [x, X]] > 0
    prob.add_constraint(((1 & x.T) // (x & X)) >> 0)
    prob.add_constraint(0.5*(X | B) + q.T*x  == 0)

    prob.set_objective('min', (X | A.T * A) - 2 * b.T * A * x)
    # from pprint import pprint
    # print(prob)
    try:
        prob.solve(verbose=0, solver='cvxopt', solve_via_dual=False, tol=EPS/2)
        # prob.solve(verbose=0, solver='cvxopt', solve_via_dual=False, rel_prim_fsb_tol=EPS/2)
    except ValueError:
        print(A)
        print('retrying')
        prob.solve(verbose=0, solver='cvxopt', solve_via_dual=False, tol=0.1)
        # prob.solve(verbose=0, solver='cvxopt', solve_via_dual=False, rel_prim_fsb_tol=0.1)
    x_hat = np.array(x.value).T[0]
    assert (x_hat >= EPS-1e-2).all() and (x_hat[:2] <= K_UB+1e-2).all(), '{}'.format(x)
    return x_hat


def post_weight1_samples(y_hats, p, v_0):
    if y_hats.shape[0] < 2:
        return p, 0*p + 1
    v_hat = np.vstack((v_0 + np.zeros(y_hats.shape[1]),
                       (y_hats[1:, :] - y_hats[:-1, :]) / DT))
    v_check_mask = np.any(v_hat < 0, axis=0)
    w_f2 = v_check_mask * 0.
    up_w = w_f2 + 1.
    w_f2[v_check_mask] += 1e12
    up_w[v_check_mask] *= 1e-12  # effectively 0
    p *= up_w
    assert p.sum() > 0
    p /= p.sum()
    return p, w_f2
