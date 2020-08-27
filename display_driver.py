import numpy as np
from matplotlib.patches import Rectangle
import utils as ut
import query as qu


def visualize_position_from_frame(ax, df, frame0=-1, frame_last=-1, frame_skip=1, artists=None, **kwargs):
    frame0 = frame0 if frame0 >= 0 else df['Frame_ID'].min()
    frame_last = frame_last if frame_last > 0 else df['Frame_ID'].max()
    artists = artists or {}
    frame_i = frame0
    while frame_i < frame_last:
        df_i = df[df['Frame_ID'] == frame_i]
        artists, maintain_keys = visualize_position(ax, df_i, artists=artists, **kwargs)
        artist_keys = list(artists.keys())
        for k in artist_keys:
            if k not in maintain_keys and 'patch' not in k:
                artists[k].remove()
                artists.pop(k)
        info_str = 'frame={:3.0f}'.format(frame_i)
        ax.set_title(info_str)
        frame_i += frame_skip
        yield None


def visualize_position(ax, df, artists=None, vids2rect_kwargs=(),
                       is_scroll_x=False, is_scroll_y=False, **kwargs):
    vids2rect_kwargs = vids2rect_kwargs or {}
    artists = artists or {}
    maintain_keys = []
    dot_kwargs = dict(
        marker='o', color='black', alpha=0.5, linestyle=''
    )
    rect_kwargs = dict(alpha=0.5, facecolor='blue')
    agent_ids = df['Vehicle_ID'].unique()
    for agent_id in agent_ids:
        tag = '{:4.0f}'.format(agent_id)
        tag_patch = tag + 'patch'
        tag_text = tag + 'text'
        df_i = df[df['Vehicle_ID'] == agent_id]
        xy = df_i[['Local_X', 'Local_Y']].values[0]
        lw = df_i[['v_Length', 'v_Width']].values[0]
        if tag in artists:
            # artists[tag].set_data(xy[0], xy[1])
            artists[tag_patch].set_xy(xy - np.array([0, lw[0]]))
            artists[tag_text].set_position(xy + .2)
            artists[tag_text].set_text(tag)
        else:
            # artists[tag], = ax.plot(xy[0], xy[1], **dot_kwargs)
            rk = rect_kwargs if agent_id not in vids2rect_kwargs else vids2rect_kwargs[agent_id]
            artists[tag_patch] = Rectangle(xy - np.array([0, lw[0]]), lw[1], lw[0], **rk)
            artists[tag] = ax.add_artist(artists[tag_patch])
            artists[tag_text] = ax.text(xy[0] + .2, xy[1] + .2, tag, alpha=0.5, fontsize=8)
        maintain_keys.extend((tag, tag_patch, tag_text))
    if is_scroll_x:
        x = df[df['Vehicle_ID'].isin(agent_ids)]['Local_X'].values
        ax.set_xlim(x.min()-5, max(x.min() + 20, x.max() + 5))
    if is_scroll_y:
        print('foo')
        y = df[df['Vehicle_ID'].isin(agent_ids)]['Local_Y'].values
        ax.set_ylim(y.min()-5, max(y.min() + 20, y.max() + 5))
    return artists, maintain_keys


def visualize_speed_plot(ax, df, artists=None, vid2kwargs=None, label_order=()):
    vid2kwargs = vid2kwargs or {}
    default_kw = dict(alpha=0.5)
    vids = df['Vehicle_ID'].unique()
    for vid in vids:
        df_i = df[df['Vehicle_ID'] == vid]
        p = df_i['Local_Y'].values
        v = p[1:] - p[:-1]
        kw = vid2kwargs[vid] if vid in vid2kwargs else default_kw
        ax.plot(df_i['Frame_ID'].values[1:], v, label=str(vid), **kw)
    add_legend_by_sorted_labels(ax, label_order)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Speed (m/s /10)')
    return artists


def visualize_position_plot(ax, df, vid2kwargs=None, label_order=()):
    vid2kwargs = vid2kwargs or {}
    default_kw = dict(alpha=0.5)
    vids = df['Vehicle_ID'].unique()
    for vid in vids:
        df_i = df[df['Vehicle_ID'] == vid]
        kw = vid2kwargs[vid] if vid in vid2kwargs else default_kw
        ax.plot(df_i['Frame_ID'].values, df_i['Local_Y'].values, label=str(vid), **kw)
    add_legend_by_sorted_labels(ax, label_order)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Position (m)')


def add_legend_by_sorted_labels(ax, label_order):
    if label_order:
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        permute_d = {label: ind for ind, label in enumerate(labels)}
        permute_handles = [handles[permute_d[label]] for label in label_order]
        ax.legend(permute_handles, label_order)
    else:
        ax.legend()


def visualize_pair_distance_plot(ax, df, pairs, pair2kwargs=None):
    pair2kwargs = pair2kwargs or {}
    default_kw = dict(alpha=0.5)
    for pair in pairs:
        df_i = qu.get_shared_frames_df(df, pair)
        kw = pair2kwargs[pair] if pair in pair2kwargs else default_kw
        dif = df_i[df_i['Vehicle_ID'] == pair[0]]['Local_Y'].values -\
            df_i[df_i['Vehicle_ID'] == pair[1]]['Local_Y'].values
        ax.plot(df_i['Frame_ID'].unique(), dif, label=str(pair), **kw)
    ax.legend()
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance (m)')


def draw_lane_guides(ax, lane_guides_x):
    for lane_guide in lane_guides_x:
        ax.axvline(lane_guide, 0, 1, color='black', alpha=0.4)


def main_display_positions():
    tag = ut.DatasetTag.i80
    df = ut.load_df(ut.get_dataset_path(tag, 0))
    print(df.head())

    import matplotlib  # for mac
    matplotlib.use('TkAgg')  # for mac
    import matplotlib.pyplot as plt
    plt.ion()

    fig, ax = plt.subplots()
    ax.set_xlim([df['Local_X'].min()-10, df['Local_X'].max()+10])
    ax.set_ylim([df['Local_Y'].min(), df['Local_Y'].max()])
    plt.gca().set_aspect('equal', adjustable='box')
    ax.grid()
    plt.show()
    draw_lane_guides(ax, ut.get_lane_guides(tag))
    for _ in visualize_position_from_frame(ax, df, frame_skip=1, frame0=600):
        plt.pause(0.01)
    plt.close()


def main_display_pairs():
    import matplotlib  # for mac
    matplotlib.use('TkAgg')  # for mac
    import matplotlib.pyplot as plt
    plt.ion()
    tag = ut.DatasetTag.us101
    tau_dist = 10.0  # m
    onramp_rect_kw = dict(alpha=0.5, facecolor='springgreen')
    mainline_rect_kw = dict(alpha=0.5, facecolor='red')
    frame_skip = 5
    lane_guides_x = ut.get_lane_guides(tag)

    df = ut.load_df(ut.get_dataset_path(tag, 0))
    mainline_id, onramp_id = ut.get_mainline_onramp_lane_ids(tag)
    onramp_vids = qu.get_ids_entering_vehicles(df, onramp_id, mainline_id)
    for onramp_vid in onramp_vids:
        mainline_vids = qu.get_close_ids_in_lane(df, onramp_vid, onramp_id, mainline_id, tau_dist)
        if len(mainline_vids) == 0:
            continue
        frames = df[(df['Vehicle_ID'] == onramp_vid) & (df['Lane_ID'] == onramp_id)]['Frame_ID'].values
        for mainline_vid in mainline_vids:
            vids2rect_kwargs = {onramp_vid: onramp_rect_kw, mainline_vid: mainline_rect_kw}

            fig_speed, axs = plt.subplots(3, 1, sharex='all')
            lead_vid = df[(df['Vehicle_ID'] == mainline_vid) & (df['Frame_ID'] == frames[-5])]['Preceding'].values[0]
            pair2kwargs = {
                (lead_vid, onramp_vid): dict(alpha=0.8, color='black'),
                (lead_vid, mainline_vid): dict(alpha=0.5, color='orange'),
            }
            vid2kwargs = {
                onramp_vid: dict(alpha=0.8, color='black'),
                lead_vid: dict(alpha=0.5, color='orange'),
            }
            visualize_pair_distance_plot(
                axs[0], df[(df['Vehicle_ID'].isin([onramp_vid, mainline_vid, lead_vid])) &
                           (df['Frame_ID'].isin(frames))],
                [(lead_vid, onramp_vid), (lead_vid, mainline_vid)], pair2kwargs=pair2kwargs)
            visualize_position_plot(axs[1], df[(df['Vehicle_ID'].isin([onramp_vid, mainline_vid, lead_vid])) &
                                               (df['Frame_ID'].isin(frames))], vid2kwargs=vid2kwargs)
            visualize_speed_plot(axs[2], df[(df['Vehicle_ID'].isin([onramp_vid, mainline_vid, lead_vid])) &
                                            (df['Frame_ID'].isin(frames))], vid2kwargs=vid2kwargs)
            vision_clearance_frame_id = qu.get_first_vision_clearance(df, mainline_vid, onramp_vid)
            if vision_clearance_frame_id > -1:
                for ax in axs.ravel():
                    ax.axvline(vision_clearance_frame_id, 0, 1, alpha=0.1, color='black')
            left_edge_merge_frame_id = qu.get_first_lane_clearance(df, onramp_vid, lane_guides_x[-1], crossing_pt='Local_X')
            midpoint_merge_frame_id = qu.get_first_lane_clearance(df, onramp_vid, lane_guides_x[-1])
            if midpoint_merge_frame_id > -1:
                for ax in axs.ravel():
                    ax.axvline(left_edge_merge_frame_id, 0, 1, alpha=0.1, color='black')
                    ax.axvline(midpoint_merge_frame_id, 0, 1, alpha=0.1, color='black')
            plt.show()

            lag_vid = qu.get_lag_vid(df, mainline_id, onramp_vid, after_frame_id=vision_clearance_frame_id)
            print(lag_vid)

            fig, ax = plt.subplots(figsize=(6, 12))
            ax.set_xlim([df['Local_X'].min()-10, df['Local_X'].max()+10])
            ax.set_ylim([df['Local_Y'].min()+5, min(df['Local_Y'].max(), 200)])
            plt.gca().set_aspect('equal', adjustable='box')
            ax.grid(b=False)
            plt.show()
            draw_lane_guides(ax, lane_guides_x)
            for _ in visualize_position_from_frame(
                    ax, df, frame_skip=frame_skip, frame0=frames.min(),
                    frame_last=frames.max()+5*frame_skip,
                    vids2rect_kwargs=vids2rect_kwargs):
                plt.pause(0.5)
            input('_: ')
            plt.close(fig)
            plt.close(fig_speed)


def main_get_merge_pairs(is_display=True, tag=ut.DatasetTag.i80, dataset_split=0, frames_before_obs=4,
                         frames_before_min=32-3, is_verbose=False):
    verbose_print = print if is_verbose else lambda *a, **k: None

    # frames_before_min = 32-3  # example must have this
    # frames_before_obs = 4  # actually used as observations
    if is_display:
        import matplotlib  # for mac
        matplotlib.use('TkAgg')  # for mac
        import matplotlib.pyplot as plt
        plt.ion()
        lead_rect_kw = dict(alpha=0.5, facecolor='springgreen')
        lag_rect_kw = dict(alpha=0.5, facecolor='red')
        frame_skip = 5
    lane_guides_x = ut.get_lane_guides(tag)
    official_merge_vs_left_edge_frame_leeway = 40

    df = ut.load_df(ut.get_dataset_path(tag, dataset_split))
    mainline_id, onramp_id = ut.get_mainline_onramp_lane_ids(tag)
    onramp_vids = qu.get_ids_entering_vehicles(df, onramp_id, mainline_id)[0:20000]
    for onramp_vid in onramp_vids:
        # get pairs:
        # lag, lead, t in [t1 - k, left edge]
        # laglag, lag, t in [t0 - k, t1]
        official_merge_frame_id = qu.get_official_merge_frame_id(df, onramp_id, mainline_id, onramp_vid)
        if official_merge_frame_id == -1:
            # This vid does not actually merge onto mainline (eg in us101 was in aux lane to exit to off ramp)
            continue
        left_edge_merge_frame_id = qu.get_first_lane_clearance(
            df, onramp_vid, lane_guides_x[-1], crossing_pt='Local_X',
            before_frame_id=official_merge_frame_id+official_merge_vs_left_edge_frame_leeway)
        if left_edge_merge_frame_id == -1:
            verbose_print('Error for vid {}'.format(onramp_vid))
            verbose_print('- Official merge is too far before first entering the lane (more than leeway frames)')
            continue
        lead_vid, lag_vid = qu.get_leadlag_vid_from_closest_midpoint(
            df, mainline_id, onramp_vid, left_edge_merge_frame_id)
        if np.isnan(lead_vid) or np.isnan(lag_vid):
            verbose_print('Error for vid {}'.format(onramp_vid))
            verbose_print('- One of lead/lag pair missing')
            continue
        t1_frame_id = qu.get_first_vision_clearance(df, lag_vid, onramp_vid, before_frame_id=left_edge_merge_frame_id)
        if t1_frame_id == -1:
            verbose_print('Error for vid {}'.format(onramp_vid))
            verbose_print('- Merge in front of {} occurred before vision clearance'.format(lag_vid))
            continue
        laglag_vid = qu.get_lag_vid(df, mainline_id, lag_vid, before_frame=t1_frame_id)
        if np.isnan(laglag_vid):
            verbose_print('Error for vid {}`s laglag pair ({} as its lead)'.format(onramp_vid, lag_vid))
            verbose_print('- {} was not in lane before clearance time'.format(lag_vid))
            continue
        t0_frame_id = qu.get_first_vision_clearance(df, laglag_vid, onramp_vid, before_frame_id=t1_frame_id)
        is_using_lag_laglag_pair = True
        is_using_lead_lag_pair = True
        is_lead_nonmerge = qu.is_vid_in_single_lane(df, mainline_id, lead_vid, [t1_frame_id - frames_before_min, t1_frame_id])
        is_lag_nonmerge = qu.is_vid_in_single_lane(df, mainline_id, lag_vid, [t1_frame_id - frames_before_min, t1_frame_id])
        is_laglag_nonmerge = qu.is_vid_in_single_lane(df, mainline_id, laglag_vid, [t0_frame_id - frames_before_min, t0_frame_id])
        if not is_lag_nonmerge:
            verbose_print('Error for vid {}'.format(onramp_vid))
            verbose_print('- Lag merged out of lane')
            continue
        if not is_lead_nonmerge:
            verbose_print('Error for vid {}'.format(onramp_vid))
            verbose_print('- Lead merged out of lane')
            is_using_lead_lag_pair = False
        if not is_laglag_nonmerge:
            verbose_print('Error for vid {}'.format(onramp_vid))
            verbose_print('- Laglag merged out of lane')
            is_using_lag_laglag_pair = False
        is_lag_laglag_paired = qu.is_leadlag_pair(df, lead_vid, lag_vid, [t0_frame_id - frames_before_min, t0_frame_id])
        if not is_lag_laglag_paired:
            verbose_print('Error for vid {}'.format(onramp_vid))
            verbose_print('- Another vehicle is between lag-laglag')
            is_using_lag_laglag_pair = False
        if not is_using_lead_lag_pair and not is_using_lag_laglag_pair:
            continue
        if left_edge_merge_frame_id - t1_frame_id < 1:
            is_using_lead_lag_pair = False
        if t1_frame_id - t0_frame_id < 1:
            is_using_lag_laglag_pair = False
        # extra info
        leadlead_vid = qu.get_lead_vid(df, mainline_id, lead_vid, before_frame=t1_frame_id)
        is_leadlead_nonmerge = qu.is_vid_in_single_lane(
            df, mainline_id, leadlead_vid, [t1_frame_id - frames_before_min, left_edge_merge_frame_id])
        leadlead_vid = leadlead_vid if is_leadlead_nonmerge else np.nan
        is_lead_nonmerge_t01 = qu.is_vid_in_single_lane(
            df, mainline_id, lead_vid, [t0_frame_id-frames_before_min, t1_frame_id])
        lead_t01_vid = lead_vid if is_lead_nonmerge_t01 else np.nan
        egolead_vid = qu.get_lead_vid(df, onramp_id, onramp_vid, before_frame=t1_frame_id)

        # display
        if is_display:
            display_data = []
            if is_using_lead_lag_pair:
                display_data.append((lag_vid, lead_vid, t1_frame_id, left_edge_merge_frame_id))
            if is_using_lag_laglag_pair:
                display_data.append((laglag_vid, lag_vid, t0_frame_id, t1_frame_id))
            for (lag_vid, lead_vid, t0, t1) in display_data:
                frames = np.arange(t0 - frames_before_min, t1+50)
                fig_speed, axs = plt.subplots(3, 1, sharex='all')
                pair2kwargs = {
                    (lead_vid, onramp_vid): dict(alpha=0.8, color='black'),
                    (lead_vid, lag_vid): dict(alpha=0.5, color='orange'),
                }
                vid2kwargs = {
                    lead_vid: dict(alpha=0.5, color='orange'),
                    onramp_vid: dict(alpha=0.8, color='black'),
                }
                label_order = [str(_l) for _l in [lead_vid, onramp_vid, lag_vid]]
                visualize_pair_distance_plot(
                    axs[0], df[(df['Vehicle_ID'].isin([onramp_vid, lag_vid, lead_vid])) &
                               (df['Frame_ID'].isin(frames))],
                    [(lead_vid, onramp_vid), (lead_vid, lag_vid)], pair2kwargs=pair2kwargs)
                visualize_position_plot(
                    axs[1], df[(df['Vehicle_ID'].isin([lag_vid, onramp_vid, lead_vid])) &
                               (df['Frame_ID'].isin(frames))], vid2kwargs=vid2kwargs, label_order=label_order)
                visualize_speed_plot(
                    axs[2], df[(df['Vehicle_ID'].isin([lag_vid, onramp_vid, lead_vid])) &
                               (df['Frame_ID'].isin(frames))], vid2kwargs=vid2kwargs, label_order=label_order)
                for ax in axs.ravel():
                    ax.axvline(t0, 0, 1, alpha=0.1, color='black')
                    ax.axvline(t1, 0, 1, alpha=0.1, color='black')
                plt.show()

                vids2rect_kwargs = {lag_vid: lag_rect_kw, lead_vid: lead_rect_kw}
                fig, ax = plt.subplots(figsize=(6, 12))
                ax.set_xlim([df['Local_X'].min() - 10, df['Local_X'].max() + 10])
                ax.set_ylim([df['Local_Y'].min() + 5, min(df['Local_Y'].max(), 200)])
                plt.gca().set_aspect('equal', adjustable='box')
                ax.grid(b=False)
                plt.show()
                draw_lane_guides(ax, lane_guides_x)
                for _ in visualize_position_from_frame(
                        ax, df, frame_skip=frame_skip, frame0=frames.min(),
                        frame_last=frames.max() + 5 * frame_skip,
                        vids2rect_kwargs=vids2rect_kwargs):
                    plt.pause(0.5)
                plt.pause(0.5)
                input('_: ')
                plt.close(fig)
                plt.close(fig_speed)
        if is_using_lead_lag_pair:
            yield_dict = dict(is_merge=True, leadlead_vid=leadlead_vid,
                              onramp_vid=onramp_vid, egolead_vid=egolead_vid)
            yield df, lag_vid, lead_vid, t1_frame_id - frames_before_obs, t1_frame_id, left_edge_merge_frame_id, \
                  yield_dict
        if is_using_lag_laglag_pair:
            yield_dict = dict(is_merge=False, leadlead_vid=lead_t01_vid,
                              onramp_vid=onramp_vid, egolead_vid=egolead_vid)
            yield df, laglag_vid, lag_vid, t0_frame_id - frames_before_obs, t0_frame_id, t1_frame_id, \
                  yield_dict


def get_all_merge_pairs(frames_before_obs, frames_before_obs_min=29):
    for tag in [ut.DatasetTag.i80, ut.DatasetTag.us101]:
        for dataset_split in [0, 1, 2]:
            for pair_data in main_get_merge_pairs(
                    is_display=False, tag=tag, dataset_split=dataset_split,
                    frames_before_obs=frames_before_obs, frames_before_min=frames_before_obs_min):
                yield tag, pair_data


def main_display_predictions():
    is_display = False
    if is_display:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from display.predictions import visualize_trajectories
    from baselines import velocity_model, idm, markov_model, baseline_utils
    from evaluation import metrics

    # frames_before_obs = 4
    frames_before_obs = 32-3
    max_n_steps = 100
    np.random.seed(0)
    # difP_yield, difP_noyield = markov_model.trained_difP()
    from baselines import model_0, model_cv_reg
    y_hat_name_list = [
        'CV',
        'IDM',
        # 'HMM',
        'Proposed-NR',
        'Proposed',
    ]
    name2eval_results_list = {name: [] for name in y_hat_name_list}
    select_fcn_kwargs_list = []
    df_ref = ()
    n_eval = 0
    # tag = ut.DatasetTag.us101
    # for (df, lag_vid, lead_vid, t0, t1, t2, kwargs) in main_get_merge_pairs(
    #         is_display=False, tag=tag, frames_before_obs=frames_before_obs):
    for tag, (df, lag_vid, lead_vid, t0, t1, t2, kwargs) in get_all_merge_pairs(frames_before_obs):
        # if lag_vid > 1000:
        #     break
        df_ref = df
        is_merge = kwargs['is_merge']
        y_true = baseline_utils.get_positions(df, lag_vid, t0, t2)
        n_steps = t2 - t1
        y_hat_p_dict_list = [
            velocity_model.predict_constant_velocity(df, lag_vid, lead_vid, t0, t1, n_steps, **kwargs),
            idm.predict_idm(df, lag_vid, lead_vid, t0, t1, n_steps, **kwargs),
            # markov_model.predict_mm(df, lag_vid, lead_vid, t0, t1, n_steps, difP_yield, difP_noyield, **kwargs),
            model_0.predict_sampling(df, lag_vid, lead_vid, t0, t1, n_steps, **kwargs),
            model_cv_reg.predict_sampling_gprior(df, lag_vid, lead_vid, t0, t1, n_steps, **kwargs),
        ]
        for i in range(len(y_hat_name_list)):
            name2eval_results_list[y_hat_name_list[i]].append(
                (y_hat_p_dict_list[i][0][:max_n_steps, :], y_hat_p_dict_list[i][1],
                 y_true[-n_steps:][:max_n_steps], y_hat_p_dict_list[i][2]
                 ))
        select_fcn_kwargs = dict(is_merge=is_merge, lag_vid=lag_vid, lead_vid=lead_vid, t0=t0, t1=t1)
        select_fcn_kwargs_list.append(select_fcn_kwargs)
        n_eval += 1

        if is_display:
            plt.rcParams['axes.grid'] = True
            fig_pred, ax = plt.subplots(len(y_hat_p_dict_list), 1, sharex='all', sharey='all', figsize=(6, 8))
            visualize_trajectories(
                ax, y_true, [_[:2] for _ in y_hat_p_dict_list], y_hat_name_list=y_hat_name_list,
                data_title='lead {}, lag {}, is merge: {}'.format(lead_vid, lag_vid, is_merge),
                is_color_cf=True, fig=fig_pred, prediction_t0=t1-t0,
            )
            plt.ioff()
            plt.show()
            plt.close(fig_pred)

        if n_eval % 50 == 0:
            print('{} evaluated'.format(n_eval))
    print('\n\n\n')

    metric_fcn_list = [
        ['E[d_t]', metrics.get_expected_dist_by_time_fcns(
            n_steps=100, select_inds=np.arange(7, 48, 8)), None],
        ['rmse[d_t]', metrics.get_rmse_dist_by_time_fcns(
            n_steps=100, select_inds=np.arange(7, 48, 8)), None],
    ]
    for i in range(len(y_hat_name_list)):
        print('Results for {}'.format(y_hat_name_list[i]))
        for metric_fcns in metric_fcn_list:
            metric_name = metric_fcns[0]
            metric_val, n_eval = metrics.evaluate_metric_on_eval_results_list(
                name2eval_results_list[y_hat_name_list[i]], *metric_fcns[1], metric_fcns[2],
                select_fcn_kwargs_list=select_fcn_kwargs_list, df=df_ref)
            print('{}: {}'.format(metric_name, np.round(metric_val, decimals=2)))
            print('  {} evaluated'.format(n_eval))
        print('------------------')


if __name__ == '__main__':

    # main_display_positions()
    # main_display_pairs()
    main_display_predictions()
