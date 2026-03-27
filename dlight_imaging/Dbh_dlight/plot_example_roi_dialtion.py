# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:56:35 2026

@author: Jingyu Cao
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from common import plotting_functions_Jingyu as pf
#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
# OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_grid_free_dilation'
OUT_DIR_DF = OUT_DIR_RAW_DATA/'processed_dataframe_grid_free_dilation'

OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_Dbh_dlight")
save_plot = 0

dlight_pre  = (-1, 0)
dlight_post = (0, 1)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
regression_name ='single_trial_regression'
# DILATION_STEPS = (0, 2, 4, 6, 8, 10)
DILATION_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

#%%
rec = 'AC964-20250120-04'
print(f'loading: {rec}-------------------------------------------')
anm, date, ss = rec.split('-')
p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
p_masks = OUR_DIR_REGRESS / rec / 'masks'
suite2p_ops = np.load(Path(p_data)/'suite2p'/'plane0'/'ops.npy', allow_pickle=True).item()
ref_img_ch1 = suite2p_ops['meanImg']
ref_img_ch2 = suite2p_ops['meanImg_chan2']
ref_img_clipped_ch1 = np.clip(ref_img_ch1, 
                          np.percentile(ref_img_ch1, 1),
                          np.percentile(ref_img_ch1, 99.8))

ref_img_clipped_ch2 = np.clip(ref_img_ch2-0.05*ref_img_ch1, 
                          np.percentile(ref_img_ch2, 0.5),
                          np.percentile(ref_img_ch2, 99))


ref_img = ref_img_clipped_ch2
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
img_height, img_width = ref_img.shape
ax.imshow(ref_img, cmap='gray', alpha=1)

# plot all grid tiling FOV
pf.plot_tiled_grid(
    ax,
    img_height, img_width,
    grid_size=16,
    grid_color='grey',
    grid_alpha=0.8,
    linewidth=1, 
    show_coordinates=1,
    # grid_start=grid_start,   # comment these two lines to tile the whole FOV
    # grid_end=grid_end,
    crop_to_window=1,
    add_scalebar=1,
    scalebar_len_um=10,
)
plt.show()

#%%
roi = (5, 6)
ry, rx = roi[0], roi[1]
global_dlight_mask = np.load(p_masks / 'global_dlight_mask_enhanced.npy')[ry, rx]
global_axon_mask_0 = np.load(p_masks / f'dilated_global_axon_k=0.npy')[ry, rx]
for k in DILATION_STEPS:
    # load masks
    global_axon_mask = np.load(p_masks / f'dilated_global_axon_k={k}.npy')[ry, rx]
    dilated_global_axon_and_dlight_mask = (global_axon_mask)&(global_dlight_mask)
    
    # grid_start = (5,25)
    # grid_end = (7,29)
    
    # grid_start = (roi[0] - 1,  roi[1]-2)#(0, 27)
    # grid_end = (roi[0] +0,  roi[1]+2)#(3, 31)
    
    grid_start = (roi[0] - 1,  roi[1]-1)#(0, 27)
    grid_end = (roi[0] +1,  roi[1]+1)#(3, 31)
    
    ref_img = ref_img_clipped_ch2
    img_height, img_width = ref_img.shape
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.imshow(ref_img, cmap='gray', alpha=1)
    
    ax.imshow(np.where(global_axon_mask_0>0, 1, np.nan), 
              cmap='Set1', alpha=0.6)
    
    if k==0:
        # cmap = 'Set1'
        lw=4
        scalebar=1
        
    else:
        cmap = 'Accent'
        lw=0
        scalebar=0
        ax.imshow(np.where(dilated_global_axon_and_dlight_mask>0, 1, np.nan), 
                  cmap=cmap, alpha=0.9)
        
    # plot all grid tiling FOV
    pf.plot_tiled_grid(
        ax,
        img_height, img_width,
        grid_size=16,
        grid_color='grey',
        grid_alpha=0.8,
        linewidth=lw, 
        # show_coordinates=1,
        grid_start=grid_start,   # comment these two lines to tile the whole FOV
        grid_end=grid_end,
        crop_to_window=1,
        add_scalebar=scalebar,
        scalebar_len_um=10,
        
    )
    
    # if k ==0:
    #     ax.imshow(np.where(dilated_global_axon_and_dlight_mask>0, 1, np.nan), 
    #               cmap='ocean', alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    
    OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_dbh_dlight\dilation_example")
    pf.save_fig(fig, OUT_DIR_FIG, f'{rec}_{roi}_k={k}', save=1)
    

    
    