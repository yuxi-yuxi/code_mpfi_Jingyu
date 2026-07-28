# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 23:35:49 2025

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d

from common.trial_selection import seperate_valid_trial
from common.utils_imaging import align_trials
from common.utils_basic import trace_filter
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst    

from common import plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()
#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_Dbh_dlight")

dlight_pre  = (-1, 0)
dlight_post = (0, 1)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
regression_name ='single_trial_regression'
time_windows=[(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4)] # time windows used to quantify dFF differecne 
save_plot = 0
#%% MAIN

# initiate containers
# short_trials_mean_all_rois = []
# long_trials_mean_all_rois = []
short_trials_mean_up_rois = []
long_trials_mean_up_rois = []
short_trials_mean_zscore_up_rois = []
long_trials_mean_zscore_up_rois = []
last_reward_time_all = [] # store last reward time for all trials from all sessions to plot distribution

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

    # loading dFF traces
    print('loading dlight dFF trace...') 
    dlight_dff_all    = np.load(p_regression / 'dff_corrected_dlight.npy')
    dlight_zscore_all = np.load(p_regression / 'zscored_corrected_dlight.npy')    
    # load stat    
    # p_stats = p_regression / f'{rec}_profile_stat.parquet'
    p_stats = (OUT_DIR_RAW_DATA/'processed_dataframe' / 
               rf"{rec}_profile_combined_dilation=0_pre{dlight_pre}_post{dlight_post}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.parquet")
    roi_stats = pd.read_parquet(p_stats)

    # select_valid_rois
    roi_stats = roi_stats.loc[(roi_stats['dlight_valid'])&(roi_stats['red_valid'])&(~roi_stats['edge'])]

    up_grids = roi_stats.loc[roi_stats['Up'], 'roi_id']
    if len(up_grids) <2:
        continue
    
    # calculate last reward time
    last_reward_time = [np.nan]
    reward_time = beh['reward_times']
    run_onset_time = beh['run_onsets']
    tot_trials = len(run_onset_time)
    for t in range(1, tot_trials):
        if (reward_time[t-1] is not np.nan) and (run_onset_time is not np.nan):
            last_reward_time.append(run_onset_time[t]-reward_time[t-1])
        else:
            last_reward_time.append(np.nan)
    last_reward_time = np.array(last_reward_time)  
    last_reward_time_all.append(last_reward_time)
    bef, aft = 2, 4
    

    # last_reward_time_sorted_idx = np.argsort(last_reward_time)
    # dlight_dff_aligned_sorted = dlight_dff_aligned[:,last_reward_time_sorted_idx,:]
    # dlight_dff_aligned_sorted_valid = dlight_dff_aligned_sorted[:,valid_trials[last_reward_time_sorted_idx],:]
    
    thresh_time = np.nanpercentile(last_reward_time, 50)
    
    short_trials_indx = np.where((last_reward_time<thresh_time)&
                                 (valid_trials), True, False)
    long_trials_indx = np.where((last_reward_time>thresh_time)&
                                 (valid_trials), True, False)
    
    
    
    # all rois
    # dlight_dff = dlight_dff_all.reshape(-1, dlight_dff_all.shape[2])
    # dff_dlight_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dlight_dff, n_sd=5)
    # dff_dlight_sm = cp_gaussian_filter1d(cp.array(dff_dlight_safe), 
    #                                            sigma=1).get()
    # dlight_dff_aligned = align_trials(dff_dlight_sm, alignment='run', beh=beh,
    #                                       bef=bef, aft=aft)
    
    # DA-up rois
    up_grids_idx = np.vstack(up_grids)
    # up_grids_idx: (n_up_rois, 2) with columns [y, x]
    y = up_grids_idx[:, 0].astype(int)
    x = up_grids_idx[:, 1].astype(int)
    dlight_dff_up = dlight_dff_all[y, x, :]
    dlight_zscore_up = dlight_zscore_all[y, x, :]
    
    dff_dlight_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dlight_dff_up, n_sd=5)
    dff_dlight_sm = cp_gaussian_filter1d(cp.array(dff_dlight_safe), 
                                               sigma=1).get()
    dlight_zscore_sm = cp_gaussian_filter1d(cp.array(dlight_zscore_up), 
                                               sigma=1).get()
    dlight_dff_aligned = align_trials(dff_dlight_sm, alignment='run', beh=beh,
                                          bef=bef, aft=aft)
    dlight_zscore_aligned = align_trials(dlight_zscore_sm, alignment='run', beh=beh,
                                          bef=bef, aft=aft)
    
    short_trials_mean = np.nanmean(dlight_dff_aligned[:, short_trials_indx, :], axis=1)
    long_trials_mean = np.nanmean(dlight_dff_aligned[:, long_trials_indx, :], axis=1)
    short_trials_mean_up_rois.append(short_trials_mean)
    long_trials_mean_up_rois.append(long_trials_mean)
    
    # short_trials_mean_zscore = np.nanmean(dlight_zscore_aligned[:, short_trials_indx, :], axis=1)
    # long_trials_mean_zscore = np.nanmean(dlight_zscore_aligned[:, long_trials_indx, :], axis=1)
    # short_trials_mean_zscore_up_rois.append(short_trials_mean_zscore)
    # long_trials_mean_zscore_up_rois.append(long_trials_mean_zscore)

    
    # up rois only
    # dlight_dff = dlight_dff_all[np.vstack(up_grids)[:, 0], :, :][:,np.vstack(up_grids)[:, 1], :]
    # dlight_dff = dlight_dff.reshape(-1, dlight_dff.shape[2])
    # dlight_dff_aligned = align_trials(dlight_dff, alignment='run', beh=beh,
    #                                       bef=bef, aft=aft)
    # short_trials_indx = np.where((last_reward_time<thresh_time)&
    #                              (valid_trials), True, False)
    # long_trials_indx = np.where((last_reward_time>thresh_time)&
    #                              (valid_trials), True, False)
    # short_trials_mean = np.nanmean(dlight_dff_aligned[:, short_trials_indx, :], axis=1)
    # long_trials_mean = np.nanmean(dlight_dff_aligned[:, long_trials_indx, :], axis=1)
    
    # short_trials_mean_up_rois.append(short_trials_mean)
    # long_trials_mean_up_rois.append(long_trials_mean)
    
# short_trials_mean_all_rois = np.vstack(short_trials_mean_all_rois)
# long_trials_mean_all_rois = np.vstack(long_trials_mean_all_rois)
short_trials_mean_up_rois = np.vstack(short_trials_mean_up_rois)
long_trials_mean_up_rois = np.vstack(long_trials_mean_up_rois)
# short_trials_mean_zscore_up_rois = np.vstack(short_trials_mean_zscore_up_rois)
# long_trials_mean_zscore_up_rois = np.vstack(long_trials_mean_zscore_up_rois)
#%%
fig, ax = plt.subplots(dpi=200)
xaxis = np.arange(30*(bef+aft))/30-2
pf.plot_mean_trace(short_trials_mean_up_rois,
                   ax, xaxis, color='grey')
pf.plot_mean_trace(long_trials_mean_up_rois,
                   ax, xaxis, color='green')
plt.show()

#%%
# profile_a = 100*short_trials_mean_up_rois
# profile_b = 100*long_trials_mean_up_rois

# fig, ax = plt.subplots(figsize=(2,2), dpi=300)
# fig, ax = pf.plot_two_traces_with_scalebars(profile_b, profile_a, xaxis, ax,
#                                             colors=("tab:green", "gray"),
#                                             labels=("More t. since last rew.", "Less t. since last rew."),
#                                             timebar=0.5, dffbar=1, 
#                                             show_xaxis=1, xlabel='time from run (s)',
#                                             baseline_correct=False,
#                                             match_centers=False
#                                             )
# ax.set(xlim=(-1, 4))
# ax.legend(frameon=False, prop={'size': 6})
# save_fig(fig, OUT_DIR_FIG, 'last_reward_time_traces', save=0 )

#%%
profile_a = 100*short_trials_mean_up_rois
profile_b = 100*long_trials_mean_up_rois
fig, ax = plt.subplots(dpi=300, figsize=(2.5,2.5))
pf.plot_two_traces_with_binned_stats(profile_b, profile_a,
                                     bef=2, aft=4, ax=ax, 
                                     # baseline_window=(-1, -0),
                                     baseline_window=(-0.5, 0),
                                     time_windows = time_windows,
                                     colors = ["tab:green", "gray"],
                                     labels = ["More t. since last rew.", "Less t. since last rew."],
                                     )
ax.set(xlim=(-1, 4), ylabel='%dF/F')
save_fig(fig, OUT_DIR_FIG, r'last_reward_time_traces_dff.pdf', save=save_plot)

# profile_a = short_trials_mean_zscore_up_rois
# profile_b = long_trials_mean_zscore_up_rois
# fig, ax = plt.subplots(dpi=300, figsize=(2.5,2.5))
# pf.plot_two_traces_with_binned_stats(profile_b, profile_a,
#                                      bef=2, aft=4, ax=ax, 
#                                      baseline_window=None,
#                                      time_windows = time_windows,
#                                      colors = ["tab:green", "gray"],
#                                      labels = ["More t. since last rew.", "Less t. since last rew."],
#                                      )
# ax.set(xlim=(-1, 4), ylabel='zscored F')
# save_fig(fig, OUT_DIR_FIG, r'last_reward_time_traces_zscore.pdf', save=save_plot)
#%%
last_reward_time_all_flat = np.hstack(last_reward_time_all)
hist = plt.hist(last_reward_time_all_flat, bins=100, range=(0,1000))
hist_bins = hist[1]
hist_counts = hist[0]
peak_time = hist_bins[np.argmax(hist_counts)+1]
print(peak_time)
plt.show()