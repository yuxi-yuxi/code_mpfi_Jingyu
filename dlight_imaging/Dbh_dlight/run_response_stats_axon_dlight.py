# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:38:52 2026

@author: Jingyu Cao
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d

# sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")
# sys.path.append(r"Z:\Jingyu\code_mpfi_Jingyu")
from common.trial_selection import seperate_valid_trial
from common.utils_imaging import percentile_dff
from common.utils_basic import trace_filter
from common.event_response_quantification import quantify_event_response
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst    
import dlight_imaging.regression.utils_regression as utl

#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
# OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_grid_free_dilation'

OUT_DIR_FIG = ''
regression_name ='single_trial_regression'
# DILATION_STEPS = (0, 2, 4, 6, 8, 10)
DILATION_STEPS = (0, ) # for testing
# DILATION_STEPS = ( 1, 3, 5, 7, 9)
# DILATION_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
dlight_pre  = (-1, 0)
dlight_post = (0, 1)
#%%
# rec_lst = ['AC969-20250319-04', ] # for testing
# rec_lst = ['AC964-20250131-02', ] # for testing
# rec_lst = ['AC969-20250326-04', ] # for example ROI
for rec in tqdm(rec_lst):
    print(f'\nprocessing {rec}...')
    # load run-onset event frames
    p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
    beh = pd.read_pickle(p_beh_file)
    run_onset_frames = np.array(beh['run_onset_frames'])
    valid_trials = (seperate_valid_trial(beh))&(run_onset_frames!=-1)
    run_onset_frames_valid = run_onset_frames[valid_trials]
    
    for k_size in tqdm(DILATION_STEPS):
        print(f'\ndilation={k_size}...')
        p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                        / r'dilation_k={}'.format(k_size))
        # if not (p_regression / f'{rec}_profile_stat.parquet').exists():
            
        # load raw traces 
        raw_traces = np.load(OUR_DIR_REGRESS/rec/f'{rec}_raw_traces_k={k_size}.npz')
        red_trace  = raw_traces['red_trace']
        dlight_neuropil_trace = raw_traces['neuropil_dlight']
        # lode regression result traces
        corrected_dlight = np.load(p_regression/'corrected_dlight_trace.npy')
        # list of grids with signals extracted
        valid_grids = utl.get_valid_grids(corrected_dlight)
        
        # example_roi = valid_grids[10]
        
        # loading dFF traces
        H, W, T = corrected_dlight.shape  # (32, 32, nframes)
        if not (p_regression / 'dff_corrected_dlight.npy').exists():
            print('calculating dlight dFF trace...') 
            dff_dlight, baseline_dlight = percentile_dff(corrected_dlight.reshape(H * W, T), q=20, return_baseline=True)
            dff_dlight = dff_dlight.reshape(H, W, T)
            baseline_dlight = baseline_dlight.reshape(H, W, T)
            np.save(p_regression / 'dff_corrected_dlight.npy', dff_dlight)
            np.save(p_regression / 'baseline_corrected_dlight.npy', baseline_dlight)
        else:
            print('loading dlight dFF trace...') 
            dff_dlight = np.load(p_regression / 'dff_corrected_dlight.npy')
            
        if not (p_regression / 'dff_red.npy').exists():
            print('calculating red dFF trace...') 
            dff_red, baseline_red = percentile_dff(red_trace.reshape(H * W, T), q=20, return_baseline=True)
            dff_red = dff_red.reshape(H, W, T) 
            baseline_red = baseline_red.reshape(H, W, T)
            np.save(p_regression / 'baseline_red.npy', baseline_red)
            np.save(p_regression / 'dff_red.npy', dff_red)
        else:
            print('loading red dFF trace...')
            dff_red = np.load(p_regression / 'dff_red.npy')
        
        thresh_dlight = np.nanmean(dff_dlight) + 5*np.nanstd(dff_dlight)
        thresh_red = np.nanmean(dff_red) + 5*np.nanstd(dff_red)
        dff_dlight_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_dlight, fix_thresh=thresh_dlight)
        dff_red_safe   = np.apply_along_axis(trace_filter, axis=-1, arr=dff_red, fix_thresh=thresh_red)
        
        # dff_dlight_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_dlight, n_sd=5)
        # dff_red_safe    = np.apply_along_axis(trace_filter, axis=-1, arr=dff_red, n_sd=5)

        dff_dlight_sm = cp_gaussian_filter1d(cp.array(dff_dlight_safe), 
                                                   sigma=1).get()
        dff_red_sm = cp_gaussian_filter1d(cp.array(dff_red_safe), 
                                                   sigma=1).get()
        
        out_dir = OUT_DIR_RAW_DATA / 'processed_dataframe_grid_free_dilation_new'
        if not out_dir.exists():
            out_dir.mkdir(exist_ok=False)
        df_roi_stats = quantify_event_response(corrected_traces = dff_dlight_sm, 
                                            event_frames=run_onset_frames_valid,
                                            baseline_window=dlight_pre, 
                                            response_window=dlight_post, # seconds
                                            dilation_k = k_size,
                                            imaging_rate=30.0, shuffle_test=True,
                                            shuffle_params={'times': 1000,
                                                            'pre_event_window':  2, # seconds
                                                            'post_event_window': 4 }
                                            )
        
        df_roi_stats_red = quantify_event_response(corrected_traces = dff_red_sm, 
                                            event_frames=run_onset_frames_valid,
                                            baseline_window=dlight_pre, 
                                            response_window=dlight_post, # seconds
                                            dilation_k = k_size,
                                            imaging_rate=30.0, shuffle_test=True,
                                            shuffle_params={'times': 1000,
                                                            'pre_event_window':  2, # seconds
                                                            'post_event_window': 4 }
                                            )
        df_roi_stats.to_parquet(out_dir / f'{rec}_profile_dilation={k_size}_stat_dlight_pre{dlight_pre}_post{dlight_post}.parquet' )
        df_roi_stats_red.to_parquet(out_dir / f'{rec}_profile_dilation={k_size}_stat_red_pre{dlight_pre}_post{dlight_post}.parquet')
    
        
        
       

        
