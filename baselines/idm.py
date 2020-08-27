import numpy as np
from baselines import baseline_utils as ut

DT = 0.1


def predict_idm(df, lag_vid, lead_vid, t0, t1, n_steps, is_merge=True, onramp_vid=np.nan, **kwargs):
    front_vid = onramp_vid if is_merge else lead_vid
    y = ut.get_positions(df, lag_vid, t0, t1)
    yl = ut.get_positions(df, front_vid, t0, t1)
    v = ut.get_velocities(df, lag_vid, t0, t1)
    vl_oracle = ut.get_velocities(df, front_vid, t1-1, t1+n_steps-1)
    if yl.size == 0:
        # vid entered at t1
        yl = ut.get_positions(df, front_vid, t1, t1+1)
        vl_oracle = np.hstack((vl_oracle[0], vl_oracle))
        assert vl_oracle.size == n_steps
    # if is_merge:
    vehicle_length = ut.get_length(df, front_vid)
    g_0 = yl[-1] - y[-1] - vehicle_length
    # else:
    #     g_0 = ut.get_spacing(df, lag_vid, t0, t1)[-1]
    y_hat, _ = apply_model(y[-1], v[-1], g_0, yl[-1], vl_oracle)
    return (*ut.single_prediction2probabilistic_format(y_hat)), {}


def apply_model(x_0, v_0, g_0, xl_0, vl):
    T = 0.5
    a = 1.75
    delta = 4.
    b = 0.8  # -m/s^2 for deceleration
    d0 = 2.
    v_des = 29  # m/s ~ 65 mph
    x = np.zeros(vl.size)
    v = np.zeros(vl.size)
    x_t = x_0
    v_t = v_0
    g_t = g_0
    vehicle_length = xl_0 - x_0 - g_0
    assert vehicle_length > 0
    for i in range(vl.size):
        phi_t = d0 + v_t*T + v_t*(v_t - vl[i])/(2*np.sqrt(a*b))
        a_t = a * (1 - (v_t/v_des)**delta - (phi_t/g_t)**2)
        a_t = np.clip(a_t, -4, 4)  # realistic accel
        a_t = max(a_t, -v_t/DT)  # accelerations keep speed >= 0
        xl_0 += vl[i] * DT
        v[i] = v_t + a_t * DT
        x[i] = x_t + v_t * DT + 0.5 * a_t * DT ** 2
        v_t, x_t = v[i], x[i]
        g_t = xl_0 - x_t - vehicle_length
    return x, v
