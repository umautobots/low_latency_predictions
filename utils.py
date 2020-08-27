import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'ngsim')

LANE_WIDTH_M = 3.7


class DatasetTag:
    i80 = 0
    us101 = 1


class Lanes:
    far_right = 0
    onramp = 1
    offramp = 2


class DatasetI80:
    FOLDER = 'i-80'
    SPLITS = [
        '0400pm-0415pm/trajectories-0400-0415.csv',
        '0500pm-0515pm/trajectories-0500-0515.csv',
        '0515pm-0530pm/trajectories-0515-0530.csv',
    ]
    LANES2ID = {
        Lanes.far_right: 6,
        Lanes.onramp: 7,
        Lanes.offramp: 8,
    }


class DatasetUS101:
    FOLDER = 'us-101'
    SPLITS = [
        '0750am-0805am/trajectories-0750-0805.csv',
        '0805am-0820am/trajectories-0805-0820.csv',
        '0820am-0835am/trajectories-0820-0835.csv',
    ]
    LANES2ID = {
        Lanes.far_right: 5,
        Lanes.onramp: 6,  # really the auxiliary lane, but this is the one merged from. 7 is the real onramp
        Lanes.offramp: 8,
    }

DATASET_TAG2INFO = {DatasetTag.i80: DatasetI80, DatasetTag.us101: DatasetUS101}


def get_dataset_path(dataset_tag, split):
    dataset = DATASET_TAG2INFO[dataset_tag]
    return os.path.join(DATASETS_ROOT, dataset.FOLDER, 'vehicle-trajectory-data', dataset.SPLITS[split])


def get_mainline_onramp_lane_ids(dataset_tag):
    lane2id = DATASET_TAG2INFO[dataset_tag].LANES2ID
    return lane2id[Lanes.far_right], lane2id[Lanes.onramp]


def get_lane_guides(dataset_tag):
    far_right_id = DATASET_TAG2INFO[dataset_tag].LANES2ID[Lanes.far_right]
    return [i * LANE_WIDTH_M + 1. for i in range(far_right_id+1)]


def load_df(path, is_meters=True, dataset_tag=-1):
    df = pd.read_csv(path, header=0, sep=',')
    if is_meters:
        df = convert_ft2meters(df)
    if dataset_tag == DatasetTag.us101:
        df[df['Lane_ID'].isin([6, 7, 8])] = Lanes.onramp
    df['Local_Y_center'] = df['Local_Y'] - df['v_Length'] / 2
    df['Local_X_center'] = df['Local_X'] + df['v_Width'] / 2
    return df


def convert_ft2meters(df):
    df[['Local_X', 'Local_Y', 'v_Length', 'v_Width', 'v_Vel', 'v_Acc', 'Space_Headway']] *= 0.3048
    return df
