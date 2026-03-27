# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 22:43:24 2026

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d

from common.utils_imaging import align_trials
from common.utils_basic import normalize
from common.trial_selection import seperate_valid_trial
from common import plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()

def select_n_trials(data_array, n):
    n_trails, len_trial = data_array.shape
    # sort trials by trial response
    trial_idx = np.arange(n_trails)
    trial_response = np.nanmean(data_array[:, bef*30:bef*30+45], axis=-1) - np.nanmean(data_array[:, bef*30-30:bef*30], axis=-1)
    response_sorted_arg = np.argsort(trial_response)
    # trial_response_sorted = trial_response[response_sorted_arg]
    trial_idx_sorted = trial_idx[response_sorted_arg]
    trial_idx_sorted_to_plot = trial_idx_sorted[-n:]
    trial_idx_to_plot = np.sort(trial_idx_sorted_to_plot)
    array_to_plot = data_array[trial_idx_to_plot]
    
    return array_to_plot

def get_valid_grids(res_traces):
    """
    Get list of all valid grids (those without NaN traces).

    Args:
        res_traces: Loaded npz file with trace arrays

    Returns:
        list: List of tuples (grid_y, grid_x) for valid grids
    """
    # corrected_dlight = res_traces['corrected_dlight']
    n_blocks_y, n_blocks_x, _ = res_traces.shape

    valid_grids = []
    for gy in range(n_blocks_y):
        for gx in range(n_blocks_x):
            if not np.all(np.isnan(res_traces[gy, gx, :])):
                valid_grids.append((gy, gx))

    print(f"Found {len(valid_grids)} valid grids out of {n_blocks_y * n_blocks_x} total grids")
    return valid_grids
#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\TEST_PLOTS\example_rois")
regression_name ='single_trial_regression'

effect_size_thresh = 0.05
amp_shuff_thresh_up = 95

rec ='AC967-20250226-04'
# rec = 'AC969-20250326-04'
anm, date, ss = rec.split('-')
bef, aft = 1, 4
p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
beh = pd.read_pickle(p_beh_file)
run_onset_frames = np.array(beh['run_onset_frames'])
valid_trials = (seperate_valid_trial(beh))&(run_onset_frames!=-1)
n_trial = len(valid_trials)

p_rec = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\RegOnly\suite2p\plane0"
# load suite2p ops
suite2p_ops = np.load(p_rec+r'\ops.npy', allow_pickle=True).item()
ref_img_ch1 = suite2p_ops['meanImg']
ref_img_ch2 = suite2p_ops['meanImg_chan2']
 
p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                / r'dilation_k=0')
dff_dlight = np.load(p_regression / 'dff_corrected_dlight.npy')
dff_red = np.load(p_regression / 'dff_red.npy')
dff_dlight = cp_gaussian_filter1d(cp.array(dff_dlight), sigma=1).get()
dff_red = cp_gaussian_filter1d(cp.array(dff_red), sigma=1).get()
x, y, T = dff_dlight.shape
dff_dlight_aligned =  align_trials(dff_dlight.reshape(x*y, T), 'run', beh, bef, aft).reshape(x, y, n_trial, 150)
dff_red_aligned    =  align_trials(dff_red.reshape(x*y, T), 'run', beh, bef, aft).reshape(x, y, n_trial, 150)

valid_grids = get_valid_grids(dff_dlight)

grid_stats = pd.read_parquet(p_regression / f'{rec}_profile_stat_ES={effect_size_thresh}_shuff={amp_shuff_thresh_up}.parquet')
p_mask = (OUR_DIR_REGRESS / rec / 'masks')
axon_mask = np.load(p_mask / 'dilated_global_axon_k=0.npy')
#%%
fiber_cmap = pf.single_color_cmap('#f7d0d3', low_strong=True)
# roi = (10, 3)
save_plot = 1
# for roi in [(6, 28), ]:
for roi in [(10, 3), ]:
# for roi in [(10, 3), (1, 9), (3, 2), (12, 3), (18, 5)]:
    out_dir = (OUT_DIR_FIG/f'{rec}_grid{roi}')
    if not out_dir.exists():
        out_dir.mkdir()
    roi_dlight_aligned = 100*dff_dlight_aligned[roi]
    roi_red_aligned = 100*dff_red_aligned[roi]
    fig, ax = plt.subplots(figsize=(2, 2), dpi=200)
    xaxis = np.arange(30*(bef+aft))/30-bef
    pf.plot_two_traces_with_scalebars(roi_dlight_aligned, roi_red_aligned, xaxis, ax,
                                   timebar=0.5, dffbar=10,
                                   show_xaxis=1)
    ax.set(title='{}_grid{}'.format(rec, roi))
    save_fig(fig, out_dir, f'{rec}_grid{roi}_mean_trace', save=save_plot)
    
    # heatmap of 100 trials with highest response
    roi_dlight_aligned_clean = roi_dlight_aligned[~np.isnan(roi_dlight_aligned).any(axis=1)]
    roi_dlight_aligned_clean = normalize(roi_dlight_aligned_clean)
    ch1_trial_response_to_plot = select_n_trials(roi_dlight_aligned_clean, 100)
    fig, ax = plt.subplots(figsize=(3,3), dpi=300)
    ax.imshow(ch1_trial_response_to_plot,
              aspect='auto', interpolation='none',
              extent=[-1, 4, 100, 0],
              cmap='Greys')
    ax.set(xlabel='time_from_run', ylabel='trail#')
    save_fig(fig, out_dir, f'{rec}_grid{roi}_trial_heatmap', save=save_plot)

    
    # plot grid
    ref_img_clipped_ch1 = np.clip(ref_img_ch1, 
                              np.percentile(ref_img_ch1, 1),
                              np.percentile(ref_img_ch1, 99.8))
    
    ref_img_clipped_ch2 = np.clip(ref_img_ch2-0.05*ref_img_ch1, 
                              np.percentile(ref_img_ch2, 5),
                              np.percentile(ref_img_ch2, 99.5))
    
    grid_start = (5,25)
    grid_end = (7,29)
    
    grid_start = (roi[0] - 1,  roi[1]-2)#(0, 27)
    grid_end = (roi[0] +0,  roi[1]+2)#(3, 31)
    
    ref_img = ref_img_clipped_ch1
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    img_height, img_width = ref_img.shape
    ax.imshow(ref_img, cmap='gray', alpha=1)
    
    # plot all grid tiling FOV
    pf.plot_tiled_grid(
        ax,
        img_height, img_width,
        grid_size=16,
        grid_color='grey',
        grid_alpha=0.8,
        linewidth=3, 
        show_coordinates=0,
        grid_start=grid_start,   # comment these two lines to tile the whole FOV
        grid_end=grid_end,
        crop_to_window=1,
        add_scalebar=1,
        scalebar_len_um=10,
    )
    ax.axis('off')
    save_fig(fig, out_dir, f'{rec}_grid{roi}_dlight_img', save=save_plot)
    
    ref_img = ref_img_clipped_ch2
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    img_height, img_width = ref_img.shape
    ax.imshow(ref_img, cmap='gray', alpha=1)
    # ax.imshow(np.where(axon_mask<1, np.nan, 1), cmap=fiber_cmap, alpha=0.8)
    
    

    # plot all grid tiling FOV
    pf.plot_tiled_grid(
        ax,
        img_height, img_width,
        grid_size=16,
        grid_color='grey',
        grid_alpha=0.8,
        linewidth=1, 
        show_coordinates=0,
        grid_start=grid_start,   # comment these two lines to tile the whole FOV
        grid_end=grid_end,
        crop_to_window=1,
        add_scalebar=0,
        scalebar_len_um=10,
    )
        
    # plot axon containing grid
    mask = grid_stats['roi_id'].apply(lambda v: tuple(v) == roi)
    pf.plot_grid(
        ax, 
        # grid_stats.loc[grid_stats['roi_id']==roi],
        # grid_stats.loc[grid_stats['Up']],
        grid_stats,
        img_height, img_width,
        grid_size=16, grid_color='#EC8F3F', #'#EC8F3F'
        grid_alpha=1,
        grid_lw = 1,
        show_coordinates=0,
        grid_start=grid_start,     # set both start/end to enable windowing
        grid_end=grid_end,
        crop_to_window=1,   # crop the view to that window
        add_scalebar=0,     # turn on scalebar
    )
    
    pf.plot_grid(
        ax, 
        grid_stats.loc[mask],
        # grid_stats.loc[grid_stats['Up']],
        img_height, img_width,
        grid_size=16, grid_color='tab:red', #'#EC8F3F'
        grid_alpha=1,
        grid_lw = 4,
        show_coordinates=0,
        grid_start=grid_start,     
        grid_end=grid_end,
        crop_to_window=True,   
        add_scalebar=0,     
    )
    
    pf.plot_grid_with_mask(
        ax, 
        grid_stats.loc[mask],
        # grid_stats.loc[grid_stats['Up']],
        # grid_stats,
        img_height, img_width,
        show_grid=0,
        grid_color='tab:red', #'#EC8F3F'
        grid_alpha=0,
        grid_lw = 4,
        show_coordinates=False,
        grid_start=grid_start,     # set both start/end to enable windowing
        grid_end=grid_end,
        crop_to_window=True,   # crop the view to that window
        add_scalebar=1,     # turn on scalebar
        scalebar_len_um=10,    # length label
        
        axon_mask=axon_mask,           # (H, W) bool
        mask_cmap='Set1',         # e.g., 'Set1'
        mask_alpha=0.5,           # imshow alpha    
    )
    

    ax.axis('off')
    save_fig(fig, out_dir, f'{rec}_grid{roi}_axon_img', save=save_plot)



