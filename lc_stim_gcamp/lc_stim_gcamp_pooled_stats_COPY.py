# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:50:02 2026

@author: Jingyu Cao
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
# import matplotlib.pyplot as plt

# Spyder's %runfile --wdir runs this file from the lc_stim_gcamp directory.
# Add the repository root so sibling packages such as common remain importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lc_stim_gcamp.calculate_dff import process_F_trace
from common.utils_imaging import align_trials
from common.trial_selection import seperate_valid_trial
from common.utils_behaviour import speed_match, extract_first_licks
# from place_cell_analysis import place_cell_functions as pcf
from common.robust_sd_filter import robust_filter_along_axis
import common.plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()
from common.event_response_quantification import quantify_event_response
from help_func import align_pulses, classify_pyrs

rec_exp = [

# 'AC333-20260724-02',
# 'AC333-20260724-04', 

# 'AC333-20260725-02',
# 'AC333-20260725-04', 

# 'AC333-20260726-02',
'AC333-20260726-04', 
    
'AC333-20260728-02',
'AC333-20260728-04',  

'AC333-20260729-02',
'AC333-20260729-04',    
    
    ]

rec_ctrl = [
# 'AC334-20260729-02',
# 'AC334-20260729-04',    
    
    ]


def process_session(rec,
                    rsd_factor=3,
                    baseline_window=(-1, 0),
                    response_window=(1, 1.5), # seconds
                    profile_max_thresh=3,
                    profile_min_thresh=0.1,
                    ctrl_trail='stim+2',
                    path_res=None,
                    overwrite=False
                    ):
    
    if path_res is None:
        process_ctrl = 1
        process_stim = 1
        process_lick = 1
    else:
        path_res = Path(path_res)
        
        if ((path_res/'df_response_ctrl_pyr.parquet').exists()&
            (overwrite==False)):
            df_response_ctrl_sorted = pd.read_parquet(path_res/'df_response_ctrl_pyr.parquet')
        else:
            process_ctrl = 1
            
        if ((path_res/'df_response_stim_pyr.parquet').exists()&
            (overwrite==False)):
            df_response_stim_sorted = pd.read_parquet(path_res/'df_response_stim_pyr.parquet')
        else:
            process_stim = 1
            
        if ((path_res / f'{rec}_df_first_lick.npy').exists()&
            (overwrite==False)):
            lick_stat = np.load(path_res / f'{rec}_df_first_lick.npy', allow_pickle=True).item()
        else:
            process_lick = 1
        
            
        if (process_ctrl)or(process_stim)or(process_lick):
            
            anm, date, ss = rec.split('-')
            df_beh = pd.read_pickle(rf"Z:\Jingyu\raw_data\lc_stim_gcamp\processed_data\{rec}\{rec}.pkl")

            # pulse alignment
            pulse_method = [trial_stat[15] for trial_stat in df_beh['trial_statements']]
            pulse_method = max(pulse_method)
            if pulse_method == '2': # run-onset pulse
                df_pulse = align_pulses(df_beh, max_pulse_delay=500)
            elif pulse_method == '7': # pulse after 1500 ms post run onset
                df_pulse = align_pulses(df_beh, max_pulse_delay=2000)
            else:
                print(f'{rec}: pulse method not valid\npulse method = {pulse_method}')
                df_response_ctrl_sorted=pd.DataFrame()
                df_response_stim_sorted=pd.DataFrame() 
                lick_stat={}
                return df_response_ctrl_sorted, df_response_stim_sorted, lick_stat

            # select stim and ctrl trials
            stim_trials = df_pulse['trials_with_stim']
            stim_valid_trials = df_pulse['valid_trials']&stim_trials
            beh_valid_trials = seperate_valid_trial(df_beh, time_thresh=10000)
            stim_valid_trials = np.array(beh_valid_trials)&(stim_valid_trials)
    
    
            if ctrl_trail == 'stim+2':
                stim_plus_two_trials = np.zeros_like(stim_valid_trials)
                stim_plus_two_trials[2:] = stim_valid_trials[:-2]
                ctrl_valid_trials = np.array(beh_valid_trials)&np.array(stim_plus_two_trials)
            elif ctrl_trail == 'stim+1':
                stim_plus_one_trials = np.zeros_like(stim_valid_trials)
                stim_plus_one_trials[1:] = stim_valid_trials[:-1]
                ctrl_valid_trials = np.array(beh_valid_trials)&np.array(stim_plus_one_trials)
            elif  ctrl_trail == 'baseline_block':
                block_num = np.array(df_beh['block_numbers'])
                ctrl_valid_trials = np.array(beh_valid_trials)&(block_num == 1)
            
            
            # behaviour: first licks
            first_lick_distance = extract_first_licks(df_beh, align_by='distance')
            first_lick_time = extract_first_licks(df_beh, align_by='time')

            stim_matched, ctrl_matched, pvalue = speed_match(df_beh, stim_valid_trials, ctrl_valid_trials,
                                                             align_by='distance', 
                                                             tolerance=1.5, 
                                                             plot_validation=1)
            stim_lick_distance = first_lick_distance[stim_valid_trials]
            ctrl_lick_distance = first_lick_distance[ctrl_valid_trials]
            stim_lick_time = first_lick_time[stim_valid_trials]/1000
            ctrl_lick_time = first_lick_time[ctrl_valid_trials]/1000
            
            lick_stat = {'rec_id': rec,
                         'pulse_method': pulse_method,
                         'stim_lick_distance': stim_lick_distance,
                         'ctrl_lick_distance': ctrl_lick_distance,
                         'stim_lick_time': stim_lick_time,
                         'ctrl_lick_time': ctrl_lick_time,
                         }
            
            if path_res is not None:
                np.save(path_res / f'{rec}_df_first_lick.npy', lick_stat)
            
            if (process_ctrl)or(process_stim):
                # quantify run-onset response
                dff, is_active_soma, shutter_masks = process_F_trace(rec,
                                                                     active_soma_only=True,
                                                                     overwrite={"shutter_mask": False},
                                                                     )  
                # filter for extreme values
                # rsd_factor=3
                # thresh =pcf.dff_thresh(dff, hard_thresh=100, factor=5)
                kept_frames = ~shutter_masks
                dff_sd_kept = robust_filter_along_axis(
                    dff[:, kept_frames],
                    factor=rsd_factor,
                )
                dff_sd = np.full_like(dff, np.nan)
                dff_sd[:, kept_frames] = dff_sd_kept
                # dff_sd[abs(dff_sd)>thresh]=np.nan
                dff = dff_sd
                
                # black frames
                dff_stim_masked = dff.copy()
                # dff_stim_masked[:, train_covered_frames] = np.nan
                dff_stim_masked[:, shutter_masks] = np.nan
                run_onset_frames = np.array(df_beh['run_onset_frames'])
                
                if (process_ctrl):
                    # ctrl
                    df_response_ctrl = quantify_event_response(corrected_traces = dff_stim_masked, 
                                                        event_frames=run_onset_frames[ctrl_valid_trials],
                                                        baseline_window=baseline_window, 
                                                        response_window=response_window, # seconds
                                                        dilation_k = 0,
                                                        imaging_rate=30.0, shuffle_test=0,
                                                        shuffle_params={'times': 1000,
                                                                        'pre_event_window':  2, # seconds
                                                                        'post_event_window': 4 }
                                                        )
                    df_response_ctrl['profile_mean'] = df_response_ctrl['mean_profile'].apply(lambda x: np.nanmean(x))
                    df_response_ctrl['profile_exm']  = df_response_ctrl['mean_profile'].apply(
                        lambda x: np.nanmax(np.abs(x))
                        if np.any(np.isfinite(x)) else np.nan
                    )
                    df_response_ctrl['is_soma'] = df_response_ctrl['profile_exm'].apply(lambda x: x<profile_max_thresh)
                    df_response_ctrl['is_soma'] = (df_response_ctrl['is_soma']
                                                   &(df_response_ctrl['profile_mean']>profile_min_thresh))
                    
                                
                if (process_stim):
    
                    # stim
                    df_response_stim = quantify_event_response(corrected_traces = dff_stim_masked, 
                                                        event_frames=run_onset_frames[stim_valid_trials],
                                                        baseline_window=baseline_window, 
                                                        response_window=response_window, 
                                                        dilation_k = 0,
                                                        imaging_rate=30.0, shuffle_test=0,
                                                        shuffle_params={'times': 1000,
                                                                        'pre_event_window':  2, # seconds
                                                                        'post_event_window': 4 },
                                                        stim_window=shutter_masks
                                                        )
                    # df_response_stim['is_soma'] = (df_response_stim['valid'])&(df_response_stim['response_ratio']>0) 
                    df_response_stim['profile_mean'] = df_response_stim['mean_profile'].apply(lambda x: np.nanmean(x))
                    df_response_stim['profile_exm'] = df_response_stim['mean_profile'].apply(
                        lambda x: np.nanmax(np.abs(x))
                        if np.any(np.isfinite(x)) else np.nan
                    )
                    df_response_stim['is_soma'] = df_response_stim['profile_exm'].apply(lambda x: x<profile_max_thresh)
                    df_response_stim['is_soma'] = (df_response_stim['is_soma']
                                                   &(df_response_stim['profile_mean']>profile_min_thresh))
    
        
                assert df_response_ctrl['roi_id'].equals(df_response_stim['roi_id'])
                both_soma = (
                    df_response_ctrl['is_soma'].astype(bool)
                    & df_response_stim['is_soma'].astype(bool)
                )
    
                df_response_ctrl_sorted = df_response_ctrl.loc[both_soma]
                df_response_stim_sorted = df_response_stim.loc[both_soma]

                df_response_ctrl_sorted['rec_id'] = rec
                df_response_stim_sorted['rec_id'] = rec
                df_response_ctrl_sorted['pulse_method'] = pulse_method
                df_response_stim_sorted['pulse_method'] = pulse_method
            
                if path_res is not None:
                    df_response_ctrl_sorted.to_parquet(path_res/f'{rec}_df_response_ctrl_pyr.parquet')
                    df_response_stim_sorted.to_parquet(path_res/f'{rec}_df_response_stim_pyr.parquet')
        
    return df_response_ctrl_sorted, df_response_stim_sorted, lick_stat

#%% MAIN
if __name__ == "__main__":
    path_res = r"Z:\Jingyu\raw_data\lc_stim_gcamp\test_analysis\processed_dataframe"
    # containers
    ctrl_group_ctrl = pd.DataFrame()
    ctrl_group_stim = pd.DataFrame()
    exp_group_ctrl  = pd.DataFrame()
    exp_group_stim  = pd.DataFrame()
    lick_stat_ctrl = []
    lick_stat_exp = []
    
    for rec in rec_exp:
        df_ctrl, df_stim, lick_stat = process_session(rec, path_res=path_res)
        exp_group_ctrl   = pd.concat((exp_group_ctrl, df_ctrl))
        exp_group_stim   = pd.concat((exp_group_ctrl, df_stim))
        lick_stat_exp.append(lick_stat)
    
    for rec in rec_ctrl:
        df_ctrl, df_stim, lick_stat = process_session(rec, path_res=path_res)
        ctrl_group_ctrl  = pd.concat((exp_group_ctrl, df_ctrl))
        ctrl_group_stim  = pd.concat((exp_group_ctrl, df_stim))
        lick_stat_ctrl.append(lick_stat)

        
        
        
        