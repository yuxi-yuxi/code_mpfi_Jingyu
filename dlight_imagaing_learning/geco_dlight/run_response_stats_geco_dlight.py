# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:38:52 2026

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
# import sys
# sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")
# sys.path.append(r"Z:\Jingyu\code_mpfi_Jingyu")
from common.utils_basic import trace_filter
from common.trial_selection import seperate_valid_trial
from common.utils_imaging import percentile_dff
from common.event_response_quantification import quantify_event_response
#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\geco_dlight")
OUT_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'TEST_PLOTS' / 'regression_res'
regression_name ='single_trial_regression_func_roi'
OUT_DIR = OUT_DIR_RAW_DATA / 'processed_dataframe'


dlight_pre  = (-1, 0)
dlight_post = (0, 1)
geco_pre  = (-1, 0)
geco_post = (0.5, 1.5)

p_session_info = OUT_DIR_RAW_DATA / 'all_animals_learning_classification.parquet'
df_all = pd.read_parquet(p_session_info)
rec_lst = df_all.loc[(df_all['days_from_learned']<=2)&
                     (df_all['animal']=='AC953'), 'rec'].to_list()
#%%
# test session
# rec_lst = ['AC953-20240919-02', ]
# rec_lst = ['AC991-20250710-04',]
error_lst = []
rec_lst=['AC953-20240918-02',]
for rec in tqdm(rec_lst):
    print(f'\nprocessing {rec}...')
    try:
        # load run-onset event frames
        p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
        beh = pd.read_pickle(p_beh_file)
        run_onset_frames = np.array(beh['run_onset_frames'])
        valid_trials = (seperate_valid_trial(beh))&(run_onset_frames!=-1)
        run_onset_frames_valid = run_onset_frames[valid_trials]
        
        # lode regression result traces
        p_regression = (OUT_DIR_REGRESS / rec / regression_name)
        # if not (p_regression / f'{rec}_profile_stat.parquet').exists():
        regress_traces = np.load(p_regression / f'{regression_name}_res_traces.npz')
        corrected_dlight = regress_traces['corrected_dlight']
        # loading dFF traces
        if not (p_regression / 'dff_corrected_dlight_new.npy').exists():
            print('calculating dlight dFF trace...') 
            dff_dlight, baseline_dlight = percentile_dff(corrected_dlight, q=20, return_baseline=True)
            np.save(p_regression / 'dff_corrected_dlight.npy', dff_dlight)
            np.save(p_regression / 'baseline_corrected_dlight.npy', baseline_dlight)
        else:
            print('loading dlight dFF trace...') 
            dff_dlight = np.load(p_regression / 'dff_corrected_dlight.npy')
            
            
        if not (p_regression / 'dff_geco_new.npy').exists():
            print('calculating geco dFF trace...') 
            anm, date, ss = rec.split('-')
            p_suite2p_geco = Path(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\dLight+GECO\GECO")
            geco_trace = np.load(p_suite2p_geco / 'F.npy')
            geco_trace_neu = np.load(p_suite2p_geco / 'Fneu.npy')
            # correct calcium trace with neuropil signal
            geco_trace_corr = geco_trace - 0.7*geco_trace_neu
            dff_geco, baseline_geco = percentile_dff(geco_trace_corr, q=20, return_baseline=True) 
            np.save(p_regression / 'dff_geco.npy', dff_geco)
            np.save(p_regression / 'baseline_geco.npy', baseline_geco)
        else:
            print('loading soam geco dFF trace...')
            dff_geco_soma = np.load(p_regression / 'dff_geco.npy')
        
        #%%
        # thresh_dlight = np.nanmean(dff_dlight) + 5*np.nanstd(dff_dlight)
        # thresh_geco = np.nanmean(dff_geco_soma) + 5*np.nanstd(dff_geco_soma)
        # dff_dlight_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_dlight, fix_thresh=thresh_dlight)
        # dff_geco_safe   = np.apply_along_axis(trace_filter, axis=-1, arr=dff_geco_soma, fix_thresh=thresh_geco)
        
        # dff_dlight_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_dlight, n_sd=5)
        # dff_geco_safe   = np.apply_along_axis(trace_filter, axis=-1, arr=dff_geco_soma, n_sd=5)
    
    
        dff_dlight_sm = cp_gaussian_filter1d(cp.array(dff_dlight), 
                                                   sigma=1).get()
        dff_geco_sm = cp_gaussian_filter1d(cp.array(dff_geco_soma), 
                                                   sigma=1).get()
        
        if not (OUT_DIR / f'{rec}_profile_stat_dlight_pre{dlight_pre}_post{dlight_post}_new.parquet').exists():
            df_roi_stats = quantify_event_response(corrected_traces = dff_dlight_sm, 
                                                event_frames=run_onset_frames_valid,
                                                baseline_window=dlight_pre, 
                                                response_window=dlight_post, # seconds
                                                dilation_k = 0,
                                                imaging_rate=30.0, shuffle_test=True,
                                                shuffle_params={'times': 1000,
                                                                'pre_event_window':  2, # seconds
                                                                'post_event_window': 4 }
                                                )
            # df_roi_stats.to_parquet(p_regression / f'{rec}_profile_stat_dlight_pre{dlight_pre}_post{dlight_post}.parquet' )
            df_roi_stats.to_parquet(OUT_DIR / f'{rec}_profile_stat_dlight_pre{dlight_pre}_post{dlight_post}.parquet' )
            
            
        if not (OUT_DIR / f'{rec}_profile_stat_geco_pre{geco_pre}_post{geco_post}_new.parquet').exists():
            df_roi_stats_red = quantify_event_response(corrected_traces = dff_geco_sm, 
                                                event_frames=run_onset_frames_valid,
                                                baseline_window=geco_pre, 
                                                response_window=geco_post, # seconds
                                                dilation_k = 0,
                                                imaging_rate=30.0, shuffle_test=True,
                                                shuffle_params={'times': 1000,
                                                                'pre_event_window':  2, # seconds
                                                                'post_event_window': 4 }
                                                )
            
            # df_roi_stats_red.to_parquet(p_regression / f'{rec}_profile_stat_geco_pre{geco_pre}_post{geco_post}.parquet')
            df_roi_stats_red.to_parquet(OUT_DIR / f'{rec}_profile_stat_geco_pre{geco_pre}_post{geco_post}.parquet')
    except:
        error_lst.append(rec)
        
        
       

        
