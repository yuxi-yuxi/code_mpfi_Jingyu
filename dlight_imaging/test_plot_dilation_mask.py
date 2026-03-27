# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:21:33 2026

@author: CaoJ
"""
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
def profile_is_valid(x):
    if x is None:
        return False
    a = np.asarray(x)
    if a.size == 0:
        return False
    return np.isfinite(a).all()   # True only if no NaN/inf inside
DILATION_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
output_dir = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\TEST_PLOTS\mask_test")
filebase = 'AC963-20250122-04'
for k in DILATION_STEPS:
    mask = np.load(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\regression_res_grid_free_dilation\{filebase}\masks\dilated_global_axon_k={k}.npy")
    dlight_mask = np.load(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\regression_res_grid_free_dilation\{filebase}\masks\global_dlight_mask_enhanced.npy")
    dlight_regressor = np.load(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\regression_res_grid_free_dilation\{filebase}\masks\dlight_regressor_fiber_dilation_k={k}.npy")
    dlight_trace = np.load(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\regression_res_grid_free_dilation\{filebase}\single_trial_regression\dilation_k={k}\dff_corrected_dlight.npy")
    p_df = rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\processed_dataframe_grid_free_dilation\{filebase}_profile_combined_dilation=0_pre(-1, 0)_post(0, 1)_ES=0.05_shuff95.parquet"
    df = pd.read_parquet(p_df)
    p_df_dilated = rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\processed_dataframe_grid_free_dilation\{filebase}_profile_combined_dilation={k}_pre(-1, 0)_post(0, 1)_ES=0.05_shuff95.parquet"
    df_dilated = pd.read_parquet(p_df_dilated)
    thresh_baseline_dlight = 2        
    thresh_baseline_red    = 1
    thresh_npix = 0
    df['dlight_valid'] = (
        (df['baseline_dlight_min'] > thresh_baseline_dlight) &
        (df['mean_profile'].apply(profile_is_valid))&
        (df['n_pixels_axon_and_dlight']>thresh_npix)
        )
    df['red_valid'] = (
        (df['baseline_red_min'] > thresh_baseline_red)&
        (df['mean_profile_red'].apply(profile_is_valid))& 
        (df['n_pixels_axon_and_dlight']>thresh_npix)
        )
    up_roi = df.loc[(df['Up'])&(df['dlight_valid'])&(df['red_valid'])&(~df['edge'])]
    # up_roi_id = up_roi['roi_id']
    up_roi_id = up_roi['roi_id'].astype(str)
    up_roi_dilated = df_dilated.loc[df_dilated['roi_id'].astype(str).isin(up_roi_id)]
    # up_roi_dilated = df_dilated.loc[df_dilated['roi_id'].isin(up_roi_id)]
    idx = np.array(up_roi['roi_id'].to_list())   # shape (n_roi, 2)
    roi_masks = mask[idx[:, 0], idx[:, 1]]   # shape (n_roi, 512, 512)
    roi_dlight_trace = dlight_trace[idx[:, 0], idx[:, 1]] 
    roi_regressor = dlight_regressor[idx[:, 0], idx[:, 1]]
    roi_and_dlight = roi_masks & dlight_mask[None, :, :]
    roi_regressor_and_dlight = roi_regressor & dlight_mask[None, :, :]
    n_pixs_roi_and_dlight = np.sum(roi_and_dlight, axis=(1, 2))
    n_roi_regressor_and_dlight = np.sum(roi_regressor_and_dlight, axis=(1, 2))
    tifffile.imwrite(output_dir / f'{filebase}_roi_masks_k={k}.tiff', roi_masks .astype('uint8') * 255)
    tifffile.imwrite(output_dir / f'{filebase}_roi_and_dlight_k={k}.tiff', roi_and_dlight.astype('uint8') * 255)


