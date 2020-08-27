from display_driver import get_all_merge_pairs
from utils import DatasetTag


def get_dataset_scenario_counts():
    tag2count = {DatasetTag.i80: 0, DatasetTag.us101: 0}
    frames_before_obs = 32 - 3
    for tag, (df, lag_vid, lead_vid, t0, t1, t2, kwargs) in get_all_merge_pairs(frames_before_obs):
        tag2count[tag] += 1

    for name, tag in [('i80', DatasetTag.i80), ('us101', DatasetTag.us101)]:
        print('{} dataset has {} scenarios'.format(name, tag2count[tag]))


def get_not_included_short_obs_scenario_counts():
    frames_before_long_obs = 32 - 3
    long_obs_count = 0
    # for tag, (df, lag_vid, lead_vid, t0, t1, t2, kwargs) in get_all_merge_pairs(
    #         frames_before_long_obs, frames_before_obs_min=frames_before_long_obs):
    #     long_obs_count += 1
    frames_before_short_obs = 4
    short_obs_count = 0
    for tag, (df, lag_vid, lead_vid, t0, t1, t2, kwargs) in get_all_merge_pairs(
            frames_before_short_obs, frames_before_obs_min=frames_before_short_obs):
        short_obs_count += 1
    print('long scenarios: {} | short scenarios: {}'.format(long_obs_count, short_obs_count))


if __name__ == '__main__':
    get_dataset_scenario_counts()
    get_not_included_short_obs_scenario_counts()
