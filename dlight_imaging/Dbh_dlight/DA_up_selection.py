# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 23:35:49 2025

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.trial_selection import seperate_valid_trial
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst    

def classify_roi(df_stat):
    df_stat = df_stat.copy()
    # df_stat['dlight_valid'] = df_stat['mean_profile'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
    # df_stat['red_valid'] = df_stat['mean_profile_red'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
    # df_stat = df_stat.loc[(df_stat['dlight_valid'])&(df_stat['red_valid'])]
    df_stat['mean_dlight'] = (df_stat['mean_profile'].apply(np.nanmean))
    df_stat['mean_red'] = (df_stat['mean_profile_red'].apply(np.nanmean))
    df_stat['dlight_valid'] = df_stat['mean_dlight'].apply(lambda x: 0<x<1.5)
    df_stat['red_valid'] = df_stat['mean_red'].apply(lambda x: 0<x<1.5)
    df_stat = df_stat.loc[(df_stat['dlight_valid'])&(df_stat['red_valid'])]

    edges = [0, 31]
    df_stat['edge'] = df_stat['roi_id'].apply(lambda rc: any(v in edges for v in rc))
    df_stat['shuffle_amps_thresh_up'] = df_stat['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_up))
    df_stat['shuffle_amps_thresh_down'] = df_stat['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_down))
    
    df_stat['Up'] = np.where(
                                ~(df_stat['edge'])&
                                (df_stat['response_amplitude']>df_stat['shuffle_amps_thresh_up'])&
                                (df_stat['effect_size']>0.05),
                                True, False)

    df_stat['Down'] = np.where(
                                ~(df_stat['edge'])&
                                (df_stat['response_amplitude']<df_stat['shuffle_amps_thresh_down'])&
                                (df_stat['effect_size']< -0.05),
                                True, False)
    return df_stat
#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_Dbh_dlight")

baseline_window=(-1, 0)
response_window=(0, 1.5)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
regression_name ='single_trial_regression'
#%% MAIN
rec_lst = ['AC969-20250326-04', ] # for example ROI
for rec in tqdm(rec_lst):
    print(rec)
    anm, date, ss = rec.split('-')
    
    # load run-onset event frames
    p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
    beh = pd.read_pickle(p_beh_file)
    run_onset_frames = np.array(beh['run_onset_frames'])
    valid_trials = (seperate_valid_trial(beh))&(run_onset_frames!=-1)
    
    k_size=0
    p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                    / r'dilation_k={}'.format(k_size))
    # load stat    
    p_stats = p_regression / f'{rec}_profile_stat.parquet'
    p_stats_red = p_regression / f'{rec}_profile_stat_red.parquet'
    roi_stats = pd.read_parquet(p_stats)
    mean_profile_red = pd.read_parquet(p_stats_red)['mean_profile']
    roi_stats['mean_profile_red'] = mean_profile_red
    roi_stats = classify_roi(roi_stats)
    roi_stats.to_parquet(p_regression / f'{rec}_profile_stat_ES={effect_size_thresh}_shuff={amp_shuff_thresh_up}.parquet')
    