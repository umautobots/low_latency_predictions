import numpy as np


def visualize_trajectories(axs, y_true, y_hat_p_list, is_color_cf=False,
                           y_hat_name_list=(), fig=None, data_title='', prediction_t0=0):
    t = np.arange(prediction_t0, y_true.size)
    for i, ax in enumerate(axs):
        y_hat, p = y_hat_p_list[i]
        if is_color_cf and p.size > 2:
            # also set minimum cf for display
            p_min = 0.01
            p_min_mask = p_min <= p
            p_quantiles = np.quantile(p, [0.9])
            if y_hat[:, (p_quantiles[0] <= p) & p_min_mask].size > 0:
                ax.plot(t, y_hat[:, (p_quantiles[0] <= p) & p_min_mask], c='blue', alpha=0.1)
            if y_hat[:, (p < p_quantiles[0]) & p_min_mask].size > 0:
                ax.plot(t, y_hat[:, (p < p_quantiles[0]) & p_min_mask], c='blue', alpha=0.001)
        else:
            # also handle sparse predictions - zeros
            mask = (y_hat > 0).any(axis=1)
            ax.plot(t[mask], y_hat[mask, :], c='blue', alpha=0.4)
        ax.plot(y_true, c='black', ls='', marker='+', alpha=0.8)
        if len(y_hat_name_list) > 0:
            ax.set_title(y_hat_name_list[i])
    if fig:
        fig.suptitle(data_title)
    return axs
