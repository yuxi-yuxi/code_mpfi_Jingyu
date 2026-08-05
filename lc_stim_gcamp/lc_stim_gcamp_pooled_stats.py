# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:50:02 2026

@author: Jingyu Cao
"""
import pandas as pd
import numpy as np
import sys
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

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

# 'AC322-20260506-02',
# 'AC322-20260506-04', # issue session
# 'AC322-20260507-02',

'AC333-20260724-02',
'AC333-20260724-04', 

'AC333-20260725-02',
'AC333-20260725-04', 

'AC333-20260726-02',
'AC333-20260726-04', 
    
'AC333-20260728-02',
'AC333-20260728-04',  

'AC333-20260729-02',
'AC333-20260729-04', 

'AC333-20260730-02',
'AC333-20260730-04',     
    
    ]

rec_ctrl = [

# 'AC334-20260728-02',
# 'AC334-20260728-04', 

# 'AC334-20260729-02',
# 'AC334-20260729-04',  

# 'AC334-20260729-02',    
    ]


def process_session(rec,
                    rsd_factor=3,
                    baseline_window=(-1, 0),
                    response_window=(1, 1.5), # seconds
                    profile_max_thresh=3,
                    profile_min_thresh=0.1,
                    ctrl_trial='stim_2',
                    path_res=None,
                    overwrite=False
                    ):
    process_ctrl = False
    process_stim = False
    process_lick = False
    temporary_cache = None
    if path_res is None:
        # Reuse the normal processing path without retaining cache files.
        temporary_cache = tempfile.TemporaryDirectory()
        path_res = Path(temporary_cache.name)
        overwrite = True

    if path_res is None:
        process_ctrl = True
        process_stim = True
        process_lick = True
    else:
        path_res = Path(path_res)
        path_res.mkdir(parents=True, exist_ok=True)
        ctrl_path = path_res / f'{rec}_df_response_ctrl_{ctrl_trial}_pyr.parquet'
        stim_path = path_res / f'{rec}_df_response_stim_pyr.parquet'
        lick_path = path_res / f'{rec}_df_first_lick_{ctrl_trial}.npy'

        if ctrl_path.exists() and not overwrite:
            df_response_ctrl_sorted = pd.read_parquet(ctrl_path)
        else:
            process_ctrl = True

        if stim_path.exists() and not overwrite:
            df_response_stim_sorted = pd.read_parquet(stim_path)
        else:
            process_stim = True

        if lick_path.exists() and not overwrite:
            lick_stat = np.load(lick_path, allow_pickle=True).item()
        else:
            process_lick = True

        # Ctrl/stim must be recomputed together because both_soma is a
        # paired mask derived from both conditions.
        if process_ctrl or process_stim:
            process_ctrl = True
            process_stim = True
        
            
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
    
    
            if ctrl_trial == 'stim_2':
                stim_plus_two_trials = np.zeros_like(stim_valid_trials)
                stim_plus_two_trials[2:] = stim_valid_trials[:-2]
                ctrl_valid_trials = np.array(beh_valid_trials)&np.array(stim_plus_two_trials)
            elif ctrl_trial == 'stim_1':
                stim_plus_one_trials = np.zeros_like(stim_valid_trials)
                stim_plus_one_trials[1:] = stim_valid_trials[:-1]
                ctrl_valid_trials = np.array(beh_valid_trials)&np.array(stim_plus_one_trials)
            elif  ctrl_trial == 'baseline_block':
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
                np.save(lick_path, lick_stat)
            
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
    
                df_response_ctrl_sorted = df_response_ctrl.loc[both_soma].copy()
                df_response_stim_sorted = df_response_stim.loc[both_soma].copy()

                df_response_ctrl_sorted['rec_id'] = rec
                df_response_stim_sorted['rec_id'] = rec
                df_response_ctrl_sorted['pulse_method'] = pulse_method
                df_response_stim_sorted['pulse_method'] = pulse_method
            
                if path_res is not None:
                    df_response_ctrl_sorted.to_parquet(ctrl_path)
                    df_response_stim_sorted.to_parquet(stim_path)
        
    return df_response_ctrl_sorted, df_response_stim_sorted, lick_stat

def _session_mean_profiles(df, pulse_method, cell_class='pyrUp'):
    if df.empty:
        return np.empty((0, 0)), []
    required = {'rec_id', 'pulse_method', 'mean_profile', cell_class}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing pooled-data columns: {sorted(missing)}')
    means, counts = [], []
    rows = df.loc[(df['pulse_method'].astype(str) == str(pulse_method))
                  & df[cell_class].astype(bool)]
    for _, session_df in rows.groupby(['rec_id', 'pulse_method'], sort=True):
        profiles = [np.asarray(x, dtype=float)
                    for x in session_df['mean_profile']]
        if len({x.size for x in profiles}) > 1:
            raise ValueError('mean_profile lengths differ within a session')
        if profiles:
            means.append(np.nanmean(np.stack(profiles), axis=0))
            counts.append(len(profiles))
    if not means:
        return np.empty((0, 0)), counts
    if len({x.size for x in means}) != 1:
        raise ValueError('mean_profile lengths differ between sessions')
    return np.stack(means), counts

def plot_pooled_roi_traces(df_ctrl, df_stim, group_name, bef=2, aft=4,
                           cell_class='pyrUp'):
    # Average ROIs within each rec_id/pulse_method before plotting.
    if df_ctrl.empty and df_stim.empty:
        return {}
    ctrl_methods = df_ctrl.get('pulse_method', pd.Series(dtype=str))
    stim_methods = df_stim.get('pulse_method', pd.Series(dtype=str))
    pulse_methods = sorted(set(ctrl_methods.astype(str))
                           | set(stim_methods.astype(str)))
    figures = {}
    for method in pulse_methods:
        ctrl, n_ctrl_roi = _session_mean_profiles(df_ctrl, method, cell_class)
        stim, n_stim_roi = _session_mean_profiles(df_stim, method, cell_class)
        if not ctrl.size and not stim.size:
            continue
        available = ctrl if ctrl.size else stim
        xaxis = np.arange(available.shape[1]) / 30 - bef
        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
        if stim.size:
            pf.plot_mean_trace(stim, ax, xaxis, color='blue', label='stim')
        if ctrl.size:
            pf.plot_mean_trace(ctrl, ax, xaxis, color='green', label='ctrl')
        ax.axvline(0, color='0.5', lw=0.8, ls='--')
        title = (f'{group_name}, pulse method {method}\n'
                 f'stim: {len(n_stim_roi)} sessions/{sum(n_stim_roi)} ROIs; '
                 f'ctrl: {len(n_ctrl_roi)} sessions/{sum(n_ctrl_roi)} ROIs')
        ax.set(title=title, xlabel='Time from run onset (s)', ylabel='dF/F',
               xlim=(-1, min(aft, xaxis[-1])))
        ax.legend(frameon=False)
        fig.tight_layout()
        figures[method] = (fig, ax)
    return figures

def plot_pooled_heatmaps(df_ctrl, df_stim, group_name, bef=2, aft=4):
    if df_ctrl.empty and df_stim.empty:
        return {}
    figures = {}
    ctrl_methods = df_ctrl.get('pulse_method', pd.Series(dtype=str))
    stim_methods = df_stim.get('pulse_method', pd.Series(dtype=str))
    methods = sorted(set(ctrl_methods.astype(str))
                     | set(stim_methods.astype(str)))
    for method in methods:
        figures[method] = {}
        for condition, df in [('ctrl', df_ctrl), ('stim', df_stim)]:
            if df.empty:
                continue
            rows = df.loc[df['pulse_method'].astype(str) == method].copy()
            if rows.empty:
                continue
            label = f'{group_name}, pulse method {method}, {condition}'
            fig, ax = pf.plot_pyr_sorted_heatmap(
                rows, label, bef, aft, 'pooled', prefix='pooled',
                activity_profile='mean_profile', ratio='response_ratio',
                plot_mean=0)
            ax.set(xlim=(-1, aft), title=label)
            fig.tight_layout()
            figures[method][condition] = (fig, ax)
    return figures

def plot_first_lick_comparisons(lick_stats, group_name):
    rows = []
    for stat in lick_stats:
        if not stat:
            continue
        row = {'rec_id': stat['rec_id'],
               'pulse_method': str(stat['pulse_method'])}
        for metric in ('distance', 'time'):
            row[f'ctrl_{metric}'] = np.nanmean(stat[f'ctrl_lick_{metric}'])
            row[f'stim_{metric}'] = np.nanmean(stat[f'stim_lick_{metric}'])
        rows.append(row)
    if not rows:
        return {}
    session_stats = pd.DataFrame(rows).groupby(
        ['rec_id', 'pulse_method'], as_index=False).mean(numeric_only=True)
    figures = {}
    for method, method_df in session_stats.groupby('pulse_method', sort=True):
        figures[method] = {}
        for metric, ylabel in [('distance', '1st lick distance'),
                               ('time', '1st lick time (s)')]:
            finite = (np.isfinite(method_df[f'ctrl_{metric}'])
                      & np.isfinite(method_df[f'stim_{metric}']))
            paired_df = method_df.loc[finite]
            if paired_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(1.6, 2.5), dpi=300)
            stats = pf.plot_bar_with_paired_scatter(
                ax, paired_df[f'ctrl_{metric}'], paired_df[f'stim_{metric}'],
                title=f'{group_name}, pulse method {method}', ylabel=ylabel,
                xticklabels=('ctrl', 'stim'))
            fig.tight_layout()
            figures[method][metric] = (fig, ax, stats)
    return figures

def plot_pyrup_percent(df_ctrl, df_stim, group_name):
    if df_ctrl.empty or df_stim.empty:
        return {}
    keys = ['rec_id', 'pulse_method']
    ctrl_pct = (df_ctrl.groupby(keys)['pyrUp'].mean().mul(100)
                .rename('ctrl_pct').reset_index())
    stim_pct = (df_stim.groupby(keys)['pyrUp'].mean().mul(100)
                .rename('stim_pct').reset_index())
    paired = ctrl_pct.merge(stim_pct, on=keys, validate='one_to_one')
    figures = {}
    for method, method_df in paired.groupby('pulse_method', sort=True):
        fig, ax = plt.subplots(figsize=(1.6, 2.5), dpi=300)
        stats = pf.plot_bar_with_paired_scatter(
            ax, method_df['ctrl_pct'], method_df['stim_pct'],
            title=f'{group_name}, pulse method {method}',
            ylabel='% pyrUp', xticklabels=('ctrl', 'stim'), ylim=(0, 100))
        fig.tight_layout()
        figures[str(method)] = (fig, ax, stats)
    return figures

#%% MAIN
if __name__ == "__main__":
    path_res = r"Z:\Jingyu\raw_data\lc_stim_gcamp\test_analysis\processed_dataframe"
    pyrUp_thresh = 2.4
    pyrDown_thresh = 1/pyrUp_thresh
    # containers
    ctrl_group_ctrl = pd.DataFrame()
    ctrl_group_stim = pd.DataFrame()
    exp_group_ctrl  = pd.DataFrame()
    exp_group_stim  = pd.DataFrame()
    lick_stat_ctrl = []
    lick_stat_exp = []
    
    ctrl_trial = 'stim_2'
    # ctrl_trial = 'baseline_block'
    
    for rec in rec_exp:
        df_ctrl, df_stim, lick_stat = process_session(rec, path_res=path_res,
                                                      ctrl_trial = ctrl_trial)
        exp_group_ctrl = pd.concat((exp_group_ctrl, df_ctrl), ignore_index=True)
        exp_group_stim = pd.concat((exp_group_stim, df_stim), ignore_index=True)
        lick_stat_exp.append(lick_stat)
    
    for rec in rec_ctrl:
        df_ctrl, df_stim, lick_stat = process_session(rec, path_res=path_res,
                                                      ctrl_trial = ctrl_trial)
        ctrl_group_ctrl = pd.concat((ctrl_group_ctrl, df_ctrl), ignore_index=True)
        ctrl_group_stim = pd.concat((ctrl_group_stim, df_stim), ignore_index=True)
        lick_stat_ctrl.append(lick_stat)
    
    # assign up and down cells
    exp_group_ctrl = classify_pyrs(exp_group_ctrl, pyrUp_thresh, pyrDown_thresh)
    exp_group_stim = classify_pyrs(exp_group_stim, pyrUp_thresh, pyrDown_thresh)
    if not ctrl_group_ctrl.empty:
        ctrl_group_ctrl = classify_pyrs(ctrl_group_ctrl, pyrUp_thresh,
                                        pyrDown_thresh)
        ctrl_group_stim = classify_pyrs(ctrl_group_stim, pyrUp_thresh,
                                        pyrDown_thresh)

    # Each dictionary contains a separate figure for every pulse_method.
    exp_trace_figures = plot_pooled_roi_traces(
        exp_group_ctrl, exp_group_stim, 'exp_group')
    ctrl_trace_figures = plot_pooled_roi_traces(
        ctrl_group_ctrl, ctrl_group_stim, 'ctrl_group')
    exp_heatmap_figures = plot_pooled_heatmaps(
        exp_group_ctrl, exp_group_stim, 'exp_group')
    ctrl_heatmap_figures = plot_pooled_heatmaps(
        ctrl_group_ctrl, ctrl_group_stim, 'ctrl_group')
    exp_lick_figures = plot_first_lick_comparisons(lick_stat_exp, 'exp_group')
    ctrl_lick_figures = plot_first_lick_comparisons(lick_stat_ctrl, 'ctrl_group')
    exp_pyrup_figures = plot_pyrup_percent(
        exp_group_ctrl, exp_group_stim, 'exp_group')
    ctrl_pyrup_figures = plot_pyrup_percent(
        ctrl_group_ctrl, ctrl_group_stim, 'ctrl_group')
    plt.show()
