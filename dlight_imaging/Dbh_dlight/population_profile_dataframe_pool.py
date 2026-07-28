# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:52:50 2026

@author: Jingyu Cao
"""
#%% imports
import numpy as np
import pandas as pd
from pathlib import Path

from common import plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()

# Load recording list
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst

def profile_is_valid(x):
    if x is None:
        return False
    a = np.asarray(x)
    if a.size == 0:
        return False
    return np.isfinite(a).all()   # True only if no NaN/inf inside

def classify_rois(df_profile_ori, 
                  amp_shuff_thresh_up, amp_shuff_thresh_down,
                  effect_size_thresh):
    
    df_profile = df_profile_ori.copy()
    df_profile['shuffle_amps_thresh_up'] = df_profile['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_up))
    df_profile['shuffle_amps_thresh_down'] = df_profile['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_down))

    df_profile['Up'] = np.where(
                                ~(df_profile['edge'])&
                                (df_profile['response_amplitude']>df_profile['shuffle_amps_thresh_up'])&
                                (df_profile['effect_size']>effect_size_thresh),
                                True, False)

    df_profile['Down'] = np.where(
                                ~(df_profile['edge'])&
                                (df_profile['response_amplitude']<df_profile['shuffle_amps_thresh_down'])&
                                (df_profile['effect_size']< -effect_size_thresh),
                                True, False)

    df_profile.loc[df_profile['Up'], 'roi_type'] = 'Up'
    df_profile.loc[df_profile['Down'], 'roi_type'] = 'Down'
    df_profile.loc[(df_profile['Up']==0)&
                       (df_profile['Down']==0)
                       , 'roi_type'] = 'Stable'
    
    return df_profile
#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
# OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_grid_free_dilation'
OUT_DIR_DF = OUT_DIR_RAW_DATA/'processed_dataframe_grid_free_dilation'
dlight_pre  = (-1, 0)
dlight_post = (0, 1)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5

thresh_baseline_dlight = 2
thresh_baseline_red    = 1

regression_name ='single_trial_regression'

# DILATION_STEPS = (0, 2, 4, 6, 8, 10)
# DILATION_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
DILATION_STEPS = (0, )
#%% collect roi profile for all sessions
for k in DILATION_STEPS:
    df_pool_all = pd.DataFrame()
    for rec in rec_lst:
        print(f'loading: {rec}--------------------------------------------------')
        anm, date, ss = rec.split('-')
        # data path
        p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
        p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                        / f'dilation_k={k}')
        p_stats = OUT_DIR_DF / f'{rec}_profile_dilation={k}_stat_dlight_pre{dlight_pre}_post{dlight_post}.parquet'
        p_stats_red = OUT_DIR_DF / f'{rec}_profile_dilation={k}_stat_red_pre{dlight_pre}_post{dlight_post}.parquet'
        p_masks = OUR_DIR_REGRESS / rec / 'masks'
        
        # load dlight dataframe
        roi_stats = pd.read_parquet(p_stats)
        roi_stats['rec_id'] = rec
        roi_stats['anm'] = anm
        
        # load red dataframe
        mean_profile_red = pd.read_parquet(p_stats_red)['mean_profile']
        roi_stats['mean_profile_red'] = mean_profile_red
    
        # calculate F baseline info
        baseline_dlight = np.load(p_regression/'baseline_corrected_dlight.npy')
        baseline_red = np.load(p_regression/'baseline_red.npy')
        coords = np.array(roi_stats['roi_id'].tolist())   # shape (n_rois, 2)
        ys = coords[:, 0]
        xs = coords[:, 1]
        roi_dlight_baseline = baseline_dlight[ys, xs, :]   # shape (n_rois, frames)
        roi_red_baseline = baseline_red[ys, xs, :]   # shape (n_rois, frames)
        roi_stats['baseline_red_min'] = np.nanmin(roi_red_baseline, axis=-1)
        roi_stats['baseline_dlight_min'] = np.nanmin(roi_dlight_baseline, axis=-1)
        roi_stats['baseline_red_mean'] = np.nanmean(roi_red_baseline, axis=-1)
        roi_stats['baseline_dlight_mean'] = np.nanmean(roi_dlight_baseline, axis=-1)

        # load masks
        global_axon_mask = np.load(p_masks / f'dilated_global_axon_k={k}.npy')
        global_dlight_mask = np.load(p_masks / 'global_dlight_mask_enhanced.npy')
        dilated_global_axon_and_dlight_mask = (global_axon_mask)&(global_dlight_mask)
    
        # count ROI's pixels
        if global_axon_mask.ndim == 2:
            # ---- sanity checks ----
            H, W = global_axon_mask.shape
            n_grid = 32
            assert H % n_grid == 0 and W % n_grid == 0, f"Mask shape {global_axon_mask.shape} not divisible by n_grid={n_grid}"
            grid_h = H // n_grid
            grid_w = W // n_grid
            assert grid_h == grid_w, f"Non-square tiles? grid_h={grid_h}, grid_w={grid_w}"
            
            ys = coords[:, 0].astype(int)  # gy
            xs = coords[:, 1].astype(int)  # gx
            assert ys.min() >= 0 and ys.max() < n_grid and xs.min() >= 0 and xs.max() < n_grid, "ROI coords out of range"
            
            # reshape into (gy, tile_y, gx, tile_x), then sum over tile dims -> (gy, gx) -> select rois (shape (n_rois,))
            roi_stats['n_pixels_dlight'] = global_dlight_mask.reshape(n_grid, grid_h, n_grid, grid_w).sum(axis=(1, 3))[ys, xs]
            roi_stats['n_pixels_axon'] = global_axon_mask.reshape(n_grid, grid_h, n_grid, grid_w).sum(axis=(1, 3))[ys, xs]
            roi_stats['n_pixels_axon_and_dlight'] = dilated_global_axon_and_dlight_mask.reshape(n_grid, grid_h, n_grid, grid_w).sum(axis=(1, 3))[ys, xs]
        elif global_axon_mask.ndim == 4:
            grid_axon_mask = global_axon_mask[ys, xs]
            # grid_dlight_mask = global_dlight_mask[ys, xs]
            grid_axon_and_dlight_mask = dilated_global_axon_and_dlight_mask[ys, xs]
            # roi_stats['n_pixels_dlight'] = np.sum(grid_dlight_mask, axis=(1, 2))
            roi_stats['n_pixels_axon'] = np.sum(grid_axon_mask, axis=(1, 2))
            roi_stats['n_pixels_axon_and_dlight'] = np.sum(grid_axon_and_dlight_mask, axis=(1, 2))
        # check if edge ROIs
        edges = [0, 31]
        roi_stats['edge'] = roi_stats['roi_id'].apply(lambda rc: any(v in edges for v in rc))
        
        # identify valid ROIs based on baseline min
        roi_stats['dlight_valid'] = (
            (roi_stats['baseline_dlight_min'] > thresh_baseline_dlight) &
            roi_stats['mean_profile'].apply(profile_is_valid)
            )
        roi_stats['red_valid'] = (
            (roi_stats['baseline_red_min'] > thresh_baseline_red) &
            roi_stats['mean_profile_red'].apply(profile_is_valid)
            )
            
        # assign DA-Up ROIs
        roi_stats = classify_rois(roi_stats,
                                  amp_shuff_thresh_up, amp_shuff_thresh_down,
                                  effect_size_thresh)
        
        # save per session dataframe
        p_df_out = OUT_DIR_DF / rf"{rec}_profile_combined_dilation={k}_pre{dlight_pre}_post{dlight_post}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}_0621.parquet"
        roi_stats.to_parquet(p_df_out)
        
        # add to dataframe pool
        df_pool_all = pd.concat((df_pool_all, roi_stats))
        
    # save pooled dataframes
    p_pooled_df = OUT_DIR_DF/ rf"df_population_profile_pooled_dilation={k}_pre{dlight_pre}_post{dlight_post}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}_0621.parquet"
    df_pool_all.to_parquet(p_pooled_df)