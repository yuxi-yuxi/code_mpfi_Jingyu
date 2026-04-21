# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:18:18 2026

@author: Jingyu Cao
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst    
import dlight_imaging.regression.utils_regression as utl
from common.plotting_functions_Jingyu import save_fig
#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\Dbh_dlight") # OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_grid_free_dilation'

OUT_DIR_FIG = Path(r"\\mpfi.org\Public\Wang lab\Jingyu\dlight_learning\Dbh_dlight\TEST_PLOTS\dlight_map_traces")
regression_name ='single_trial_regression'
DILATION_STEPS = (0, )
dlight_pre  = (-1, 0)
dlight_post = (0, 1)

p_session_info = OUT_DIR_RAW_DATA / 'all_animals_learning_classification.parquet'
df_all = pd.read_parquet(p_session_info)
rec_lst = df_all.loc[(df_all['days_from_learned']<=0)&
                     (df_all['animal']=='AC964'), 'rec'].to_list()
#%% First pass: collect all response amplitudes to find global symmetric scale
all_response_amplitudes = []
for rec in rec_lst:
    df_dlight = pd.read_parquet(OUT_DIR_RAW_DATA / 'processed_dataframe_grid_free_dilation'/
                                f'{rec}_profile_combined_dilation=0_pre(-1, 0)_post(0, 1)_ES=0.05_shuff95_test.parquet',
                                )
    df_valid = df_dlight[(~df_dlight['edge'])].copy()
    all_response_amplitudes.extend(df_valid['response_amplitude'].values)

all_response_amplitudes = np.array(all_response_amplitudes)
# Calculate symmetric scale: use the max absolute value (with percentile to handle outliers)
global_max_abs = np.nanpercentile(np.abs(all_response_amplitudes), 95)
vmin_global = -global_max_abs
vmax_global = global_max_abs
print(f"Global symmetric colorbar range: [{vmin_global:.4f}, {vmax_global:.4f}]")

#%% Second pass: plot with fixed symmetric scale
for rec in rec_lst:
    # load reference image and masks for plotting
    p_ref_img = Path(r"Z:\Jingyu\Dbh_fiber_detecion")
    mean_img_ch1 = np.load(p_ref_img/'dlight_Ai14_Dbh'/rec/'ch1_mean.npy')
    mean_img_ch2 = np.load(p_ref_img/'dlight_Ai14_Dbh'/rec/'ch2_mean.npy')
    p_masks = OUR_DIR_REGRESS/rec/'masks'
    fiber_mask = np.load(p_masks/'dilated_global_axon_k=0.npy')
    fiber_mask = np.any(fiber_mask, axis=(0,1))
    dlight_mask = np.load(p_masks/'global_dlight_mask_enhanced.npy')

    # load dataframe
    df_dlight = pd.read_parquet(OUT_DIR_RAW_DATA / 'processed_dataframe_grid_free_dilation'/
                                f'{rec}_profile_combined_dilation=0_pre(-1, 0)_post(0, 1)_ES=0.05_shuff95_test.parquet',
                                )

    # Filter for valid dlight ROIs
    df_valid = df_dlight[
                         # (df_dlight['dlight_valid'])&
                         (~df_dlight['edge'])].copy()

    # Get response amplitude values and roi_ids
    response_amplitude = df_valid['response_amplitude'].values
    roi_ids = df_valid['roi_id'].values  # Each roi_id is (grid_y, grid_x)

    # Identify up_rois and calculate percentage (valid = non-NaN response_amplitude)
    up_rois = df_valid.loc[df_valid['Up'], 'roi_id'].values
    n_valid = df_valid['response_amplitude'].notna().sum()
    n_up = len(up_rois)
    pct_up = 100 * n_up / n_valid if n_valid > 0 else 0

    # Create response map image (grid-based)
    grid_size = 16  # typical grid size
    img_h, img_w = mean_img_ch2.shape
    n_grids_y = img_h // grid_size
    n_grids_x = img_w // grid_size

    # Initialize response map with NaN
    response_map = np.full((img_h, img_w), np.nan)

    # Fill in response values for each valid ROI grid
    for roi_id, amp in zip(roi_ids, response_amplitude):
        gy, gx = roi_id
        y0, y1 = gy * grid_size, (gy + 1) * grid_size
        x0, x1 = gx * grid_size, (gx + 1) * grid_size
        response_map[y0:y1, x0:x1] = amp

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Plot background reference image
    ax.imshow(mean_img_ch2,
              vmin=np.percentile(mean_img_ch2, 1),
              vmax=np.percentile(mean_img_ch2, 99),
              cmap='gray')

    # Overlay response map with global symmetric scale
    im = ax.imshow(response_map,
                   vmin=vmin_global, vmax=vmax_global,
                   cmap='coolwarm', alpha=0.6,
                   interpolation='none')

    ax.imshow(np.where(fiber_mask, 1, np.nan),
              cmap='Set1', alpha=.5)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label='Response Amplitude')

    # Highlight up_rois with edge rectangles
    # from matplotlib.patches import Rectangle
    # for roi_id in up_rois:
    #     gy, gx = roi_id
    #     y0, x0 = gy * grid_size, gx * grid_size
    #     rect = Rectangle((x0, y0), grid_size, grid_size,
    #                       linewidth=2, edgecolor='lime', facecolor='none')
    #     ax.add_patch(rect)

    ax.set_title(f'{rec}\nUp ROIs: {n_up}/{n_valid} ({pct_up:.1f}%)')
    ax.axis('off')
    save_fig(fig, OUT_DIR_FIG, f'{rec}_run_response_map')
    # plt.tight_layout()
    # plt.show()

