import numpy as np


def visualize_probability_of_inclusion(ax, bins, name2kwargs):
    for name, kwargs in name2kwargs.items():
        if 'poi' not in kwargs:
            continue
        visualize_method_probability_of_inclusion(ax, bins, name, **kwargs)
    ax.set_xlabel('distance (m)')
    ax.set_ylabel('PoI(t = 4s, distance)')
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def visualize_method_probability_of_inclusion(ax, bins, name, poi=(), color='k', ls='-', **kwargs):
    # bins = 1st edge, bin right edges = distances [0, ..., inf]
    ax.plot(bins[1:-1], poi[:-1], c=color, ls=ls, label=name)


def visualize_rmse(ax, name2kwargs):
    for name, kwargs in name2kwargs.items():
        if 'rmse' not in kwargs:
            continue
        visualize_method_rmse(ax, name, **kwargs)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('RMSE (m)')
    ax.legend(loc='upper left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def visualize_method_rmse(ax, name, rmse=(), mask=(), color='k', ls='-', **kwargs):
    t = np.arange(rmse.size) / 10 + 1/10.
    ax.plot(t[mask], rmse[mask], c=color, ls=ls, label=name)


def visualize_f_theta_gaps(ax, gaps, name2kwargs):
    for name, kwargs in name2kwargs.items():
        if 'f_theta_gap_dist' not in kwargs:
            continue
        visualize_method_f_theta_gaps(ax, gaps, name, **kwargs)
    ax.set_xlabel('Gap')
    ax.set_ylabel(r'$\displaystyle p(f(\theta_i) - f(\theta^*) \leq \mathrm{gap})$')
    ax.legend(loc='upper left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def visualize_method_f_theta_gaps(ax, gaps, name, f_theta_gap_dist=(), color='k', ls='-', **kwargs):
    ax.plot(gaps, f_theta_gap_dist, c=color, ls=ls, label=name)


def get_figure_init_name2kwargs():
    sgan_mask = np.arange(3, 48, 4)
    rest_mask = np.sort(np.hstack([sgan_mask, np.arange(9, 80, 10)]))
    name2kwargs = {
        'CV': dict(mask=rest_mask, color='g', ls=':'),
        # 'HMM': dict(mask=rest_mask, color='cyan', ls='-.'),
        'IDM': dict(mask=rest_mask, color='orange', ls='-.'),
        'SGAN': dict(mask=sgan_mask, color='red', ls='--'),
        'MATF': dict(mask=sgan_mask, color='purple', ls='--'),
        'Proposed-NR': dict(mask=rest_mask, color='magenta', ls='--'),
        'Proposed': dict(mask=rest_mask, color='blue', ls='-'),
    }
    return name2kwargs


# =====


def vis_test_rmse():
    import matplotlib  # for mac
    matplotlib.use('TkAgg')  # for mac
    import matplotlib.pyplot as plt
    n7 = np.zeros(7)
    name2kwargs = {
        'method_1': dict(
            rmse=np.hstack([n7, 2, n7, 4, n7, 7, n7, 8,
                            n7, 10., n7, 13, np.zeros(100-1-48)]),
            mask=np.arange(7, 48, 8),
            color='blue', ls='-',
        ),
        'method_2': dict(
            rmse=np.arange(100.) / 8,
            mask=np.sort(np.hstack([np.arange(7, 48, 8), np.arange(9, 100, 10)])),
            color='magenta', ls='--',
        )
    }
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(dpi=200, frameon=False)
    visualize_rmse(ax, name2kwargs)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    vis_test_rmse()
