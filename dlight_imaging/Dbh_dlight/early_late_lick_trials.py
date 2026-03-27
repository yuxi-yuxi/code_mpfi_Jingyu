# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 23:35:49 2025

@author: Jingyu Cao
"""
import os 
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from scipy.stats import ttest_ind
if ("Z:\Jingyu\Code\Python" in sys.path) == False:
    sys.path.append("Z:\Jingyu\Code\Python")
import utils_Jingyu as utl
import plotting_functions_Jingyu as pf
#%%
exp = 'dlight_Ai14_Dbh'
f_out_df_selected = r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_corr.pkl".format(exp)
df_selected = pd.read_pickle(f_out_df_selected)
# df_selected = df_selected.loc[df_selected['rg_corr_0_0.5_p']>0.01]
# df_selected = df_selected.loc[df_selected['rg_corr_full_trial_single_trial_r_median']<0.15]
df_selected = df_selected.loc[df_selected['speed_corr_single_trial_r_median']<0.3]
rec_lst = df_selected.index.tolist()
ex_lst = [
'AC963-20250218-04',
'AC964-20250203-04',
'AC964-20250205-04',
'AC964-20250218-02',
'AC966-20250217-04',
'AC966-20250219-02',
]
error_lst = ['AC966-20250226-02',]
rec_lst = [i for i in rec_lst if i not in ex_lst and i not in error_lst]
#%%
early_trials_mean_all_rois = []
late_trials_mean_all_rois = []
early_trials_mean_up_rois = []
late_trials_mean_up_rois = []
first_licks_all = []

baseline_window=(-1, 0)
response_window=(0, 1.5)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 'shuffle_amps_95'
amp_shuff_thresh_down = 'shuffle_amps_5'
regression_method = 'G_dilated_axon_dlight_R_dilated_axon_dlight'
early_thresh = 2500   # ms
late_thresh = 2500 # ms
# early_thresh = '_session_wise'   # ms
# late_thresh = '_session_wise' # ms
early_thresh_all = []
late_thresh_all = []
figure_out = r"Z:\Jingyu\Code\dlight_imgaing\dlight_Ai14_Dbh\ealry_late_licks\early{}_late{}".format(
    early_thresh, late_thresh)
if not os.path.exists(figure_out):
    os.makedirs(figure_out)
    
for rec in tqdm(rec_lst):
    print(rec)
    # rec = 'AC969-20250319-04'
    beh = pd.read_pickle(r"Z:\Jingyu\Code\dlight_imgaing\dlight_Ai14_Dbh\behaviour_profile\{}.pkl".format(rec))
    anm, date, ss = rec.split('-')
    p_data = Path(r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss))
    p_regression = p_data/r"regression_result_dilation_new_fibermask_dlightmask"
    dilation=0
    p_regression_dialtion = p_regression/f'dilation_k={dilation}'/f'{regression_method}'
    p_stats = p_regression/f'dilation_k={dilation}'/ regression_method/ f'roi_stats_pre{baseline_window}_post{response_window}_ES={effect_size_thresh}_{amp_shuff_thresh_up}.pkl'
    roi_stats = pd.read_pickle(p_stats)
    up_grids = roi_stats.loc[roi_stats['Up'], 'roi_id']
    if len(up_grids) <2:
        continue
    # down_grids = roi_stats.loc[roi_stats['Down'], roi_id]
    p_dlight_dff = p_regression_dialtion / 'corrected_dlight_dff.npy'
    p_red_dff = p_regression_dialtion / 'red_dff.npy'
    
    dlight_dff_all = np.load(p_dlight_dff)
    
    valid_trials = np.where((~np.isnan(beh['reward_times']))&
                                        (np.array(beh['non_stop_trials'])==0)&
                                        (np.array(beh['non_fullstop_trials'])==0),
                                        True, False)

    # calculate first lick time
    lick_times = beh['lick_times_aligned']
    first_lick_times = []
    tot_trials = len(lick_times)
    for licks in lick_times:
        if licks is np.nan:
            first_lick_times.append(np.nan)
        else:
            licks = np.array(licks)
            licks_filtered = licks[licks>500] # only include licks after 0.5s
            if len(licks_filtered)>0:
                first_lick_times.append(licks_filtered[0])
            else: # no licks after 0.5s
                first_lick_times.append(np.nan)
    first_lick_times = np.array(first_lick_times)    
    first_licks_all.append(first_lick_times)                      
    # last_reward_time_sorted_idx = np.argsort(last_reward_time)
    # dlight_dff_aligned_sorted = dlight_dff_aligned[:,last_reward_time_sorted_idx,:]
    # dlight_dff_aligned_sorted_valid = dlight_dff_aligned_sorted[:,valid_trials[last_reward_time_sorted_idx],:]
    
    # early_thresh = np.nanmedian(first_lick_times)
    # late_thresh = np.nanmedian(first_lick_times)
    # early_thresh_all.append(early_thresh); late_thresh_all.append(late_thresh)

    early_trials_indx = np.where((first_lick_times<early_thresh)&
                                 (valid_trials), True, False)
    early_trials_indx = np.where(early_trials_indx)[0].tolist()
    late_trials_indx = np.where((late_thresh<first_lick_times)&
                                (first_lick_times<4000)&
                                (valid_trials), True, False)
    late_trials_indx = np.where(late_trials_indx)[0].tolist()
    
    # speed matching
    speeds_aligned = [np.stack(speed)[:, 1] if len(speed)>0 else [] 
                      for speed in beh['speed_times_aligned']]
    # binning
    speeds_binned = []
    speed_trial_idx = []
    for idx, speed in enumerate(speeds_aligned):
        if len(speed)<4000:
            speeds_binned.append(np.nan)
        else: # 0.5s bin, 7 bins
            speed_binned = speed[:4000].reshape(8, 500).mean(axis=1)
            speeds_binned.append(speed_binned)
            speed_trial_idx.append(idx)
    early_trials_indx_speed = [i for i in early_trials_indx if i in speed_trial_idx]
    late_trials_indx_speed = [i for i in late_trials_indx if i in speed_trial_idx]
    if len(early_trials_indx_speed)==0 or len(late_trials_indx_speed)==0:
        continue #skip sessions without out any early or late trials                        
    early_speed_binned = [speeds_binned[i] for i in early_trials_indx_speed]
    late_speed_binned = [speeds_binned[i] for i in late_trials_indx_speed]        
    early_speed_miu = np.mean(early_speed_binned, axis=0)
    early_speed_std = np.std(early_speed_binned, axis=0)
    late_speed_miu = np.mean(late_speed_binned, axis=0)
    late_speed_std = np.std(late_speed_binned, axis=0)
    # how tight the speed match should be 
    tolerance = 1.5
    early_bound_low = early_speed_miu-tolerance*early_speed_std 
    early_bound_high = early_speed_miu+tolerance*early_speed_std
    late_bound_low = late_speed_miu-tolerance*late_speed_std 
    late_bound_high = late_speed_miu+tolerance*late_speed_std
    
    early_in_bound = np.all((early_speed_binned>=late_bound_low)&
                            (early_speed_binned<=late_bound_high),
                            axis=1)
    late_in_bound = np.all((late_speed_binned>=early_bound_low)&
                            (late_speed_binned<=early_bound_high),
                            axis=1)
    early_in_bound_idx = [early_trials_indx_speed[i] for i in np.where(early_in_bound)[0]]
    late_in_bound_idx = [late_trials_indx_speed[i] for i in np.where(late_in_bound)[0]]
    if len(early_in_bound_idx)<10 or len(late_in_bound_idx)<10:
        continue # skip sessions without enough trials
    # check speed match using test
    early_speed_binned_inbound = np.vstack([speeds_binned[i] for i in early_in_bound_idx])
    late_speed_binned_inbound = np.vstack([speeds_binned[i] for i in late_in_bound_idx])
    p_values = []
    for i in range(early_speed_binned_inbound.shape[1]): #n_bins
        res = ttest_ind(early_speed_binned_inbound[:, i], late_speed_binned_inbound[:, i])
        p_values.append(res[1])
    p_values = np.array(p_values)
    if np.any(p_values<0.05):
        continue # skip session with different speed even after matching
    # plot speed match validation
    fig, axs = plt.subplots(1, 2, figsize=(4,2), dpi=300); fig.tight_layout()
    plt.suptitle=(f'{rec}')
    early_speeds = [utl.zero_padding(speeds_aligned[i], 4000) for i in early_trials_indx]
    late_speeds = [utl.zero_padding(speeds_aligned[i], 4000) for i in late_trials_indx]
    early_speeds_matched = [utl.zero_padding(speeds_aligned[i], 4000) for i in early_in_bound_idx]
    late_speeds_matched = [utl.zero_padding(speeds_aligned[i], 4000) for i in late_in_bound_idx]
    ax=axs[0]
    pf.plot_mean_trace(early_speeds, ax, color='grey')
    pf.plot_mean_trace(late_speeds, ax, color='steelblue')
    ax.set(title='before_match', ylabel='speed', xlabel='time')
    ax=axs[1]
    pf.plot_mean_trace(early_speeds_matched, ax, color='grey')
    pf.plot_mean_trace(late_speeds_matched, ax, color='steelblue')
    ax.set(title='after_match', ylabel='speed', xlabel='time')
    plt.savefig(figure_out+r'\speed_match_{}.png'.format(rec))
    # plt.show()
    plt.close()
    
    # all rois profile
    dlight_dff = dlight_dff_all.reshape(-1, dlight_dff_all.shape[2])
    dlight_dff_aligned = utl.align_trials(dlight_dff, alignment='run', beh=beh,
                                          bef=2, aft=4)
    early_trials_mean = np.nanmean(dlight_dff_aligned[:, early_in_bound_idx, :], axis=1)
    late_trials_mean = np.nanmean(dlight_dff_aligned[:, late_in_bound_idx, :], axis=1)
    
    early_trials_mean_all_rois.append(early_trials_mean)
    late_trials_mean_all_rois.append(late_trials_mean) 
    
    # up rois only profile
    coords = np.vstack(up_grids.values)
    ys = coords[:, 0]   # all y indices
    xs = coords[:, 1]   # all x indices
    dlight_dff_up = dlight_dff_all[ys, xs, :] 
    # dlight_dff = dlight_dff_all[np.vstack(up_grids)[:, 0], :, :][:,np.vstack(up_grids)[:, 1], :]
    # dlight_dff = dlight_dff.reshape(-1, dlight_dff.shape[2])
    dlight_dff_up_aligned = utl.align_trials(dlight_dff_up, alignment='run', beh=beh,
                                          bef=2, aft=4)
        
    early_trials_mean = np.nanmean(dlight_dff_up_aligned[:, early_in_bound_idx, :], axis=1)
    late_trials_mean = np.nanmean(dlight_dff_up_aligned[:, late_in_bound_idx, :], axis=1)
    
    sm_sigma = 2
    
    fig, ax = plt.subplots(figsize=(3,3), dpi=300)
    xaxis = np.arange(30*(2+4))/30-2
    pf.plot_mean_trace(gaussian_filter1d(early_trials_mean, sigma=sm_sigma, axis=-1),
                       ax, xaxis, color='grey', label=f'early_trials (n={len(early_in_bound_idx)})')
    pf.plot_mean_trace(gaussian_filter1d(late_trials_mean, sigma=sm_sigma, axis=-1),
                       ax, xaxis, color='green', label=f'late_trials (n={len(late_in_bound_idx)})')
    ax.legend(frameon=False)
    ax.set(title=f'{rec}')
    fig.tight_layout()
    plt.savefig(figure_out+r'\late_vs_early_dlight_{}.png'.format(rec))
    # plt.show()
    plt.close()
    
    early_trials_mean_up_rois.append(early_trials_mean)
    late_trials_mean_up_rois.append(late_trials_mean)
    
early_trials_mean_all_rois = np.vstack(early_trials_mean_all_rois)
late_trials_mean_all_rois = np.vstack(late_trials_mean_all_rois)
early_trials_mean_up_rois = np.vstack(early_trials_mean_up_rois)
late_trials_mean_up_rois = np.vstack(late_trials_mean_up_rois)
#%%
sm_sigma = 2
fig, ax = plt.subplots(figsize=(2,2), dpi=200)
xaxis = np.arange(30*(2+4))/30-2
pf.plot_mean_trace(gaussian_filter1d(early_trials_mean_all_rois, sigma=sm_sigma, axis=-1),
                   ax, xaxis, color='grey')
pf.plot_mean_trace(gaussian_filter1d(late_trials_mean_all_rois, sigma=sm_sigma, axis=-1),
                   ax, xaxis, color='green')
plt.show()

fig, ax = plt.subplots(figsize=(3,3), dpi=300)
xaxis = np.arange(30*(2+4))/30-2
pf.plot_mean_trace(gaussian_filter1d(early_trials_mean_up_rois, sigma=sm_sigma, axis=-1),
                   ax, xaxis, color='grey', label='early_trials')
pf.plot_mean_trace(gaussian_filter1d(late_trials_mean_up_rois, sigma=sm_sigma, axis=-1),
                   ax, xaxis, color='green', label='late_trials')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig(figure_out+r'\late_vs_early_dlight.png')
plt.show()

#%%
first_licks_flat = np.hstack(first_licks_all)/1000
fig, ax = plt.subplots()
ax.hist(first_licks_flat, bins=100, range=(0, 6))
ax.set(title = 'first_lick_time_distribution_Dbh-dlight\nmedian={:.2f}'.format(np.nanmedian(first_licks_flat)),
       xlabel='time from run (s)',
       ylabel='trial count')
ax.axvline(np.nanmedian(first_licks_flat), lw=1, color='grey', ls='--')
plt.show()