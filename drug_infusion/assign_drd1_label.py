# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:37:44 2026

@author: Jingyu Cao
"""

#%% imports 
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import center_of_mass
import xarray as xr

from common import plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()

from drd1_detection import drd1_cell_match

def warp_rois_rigid(roi_map, sh, fill=0):
    """
    roi_map: (T, H, W) or (H, W)
    sh: (2,) rigid shift [dy, dx] (most common) OR [dx, dy] if you swap
    """
    sh = np.asarray(sh).astype(float).ravel()
    dy, dx = sh[0], sh[1]

    if roi_map.ndim == 2:
        shift_vec = (dy, dx)
    elif roi_map.ndim == 3:
        shift_vec = (0.0, dy, dx)  # no shift along first axis
    else:
        raise ValueError(f"roi_map must be 2D or 3D, got shape {roi_map.shape}")

    return ndi_shift(roi_map, shift=shift_vec, order=0, mode="constant", cval=fill)
#%% PATHS AND PARAMS
import drug_infusion.rec_lst_infusion as rec_info
# rec_drug, rec_drug_ctrl = rec_info.rec_SCH, rec_info.rec_SCH_ctrl
# for (rec_drug, rec_drug_ctrl) in [(rec_info.rec_SCH, rec_SCH_ctrl)
recs = rec_info.rec_lst

pre_window=(-1, 0)
post_window=(0.5, 1.5)
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
#%% 
error_lst = []
for rec_idx, rec in tqdm(recs.iterrows(), total=len(recs), desc="Processing sessions"):
    anm    = rec['anm']
    date   = rec['date']
    rec_id = anm +'-'+date
    
    if rec_id != 'AC310-20250828':
        continue
    data_path = OUT_DIR_RAW_DATA/'raw_signals'/f'{anm}-{date}'
    print(f'\n{anm}-{date}')

    # always use the first session for reference mean image
    p_suite2p_ss1 = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\02\suite2p_func_detec\plane0"
    suite2p_ss1_ops = np.load(p_suite2p_ss1+r'\ops.npy', allow_pickle=True).item()
    mean_img_ch1 = suite2p_ss1_ops['meanImg']
    mean_img_ch2 = suite2p_ss1_ops['meanImg_chan2_corrected']
    
    if not (data_path/r'drd1_label_new.npy').exists():
        
        A_master = xr.open_dataarray(data_path/"A_master.nc")
        roi_map = A_master.values.squeeze()
        shift_ds = xr.open_dataset(data_path/"shift_ds.nc")
        sh = shift_ds["shifts"].sel(animal=anm, session=f'{date}_02')
        roi_map_shifted = warp_rois_rigid(roi_map, (-sh).values)
        
        gcamp_rois = roi_map_shifted
        ref_mean = mean_img_ch2
        
        coms = np.array([center_of_mass(roi) for roi in gcamp_rois])
        ys = coms[:, 0]
        xs = coms[:, 1]
        
        com_vals = np.array([
            ref_mean[int(y), int(x)] if np.isfinite(y) and np.isfinite(x) else np.nan
            for y, x in zip(ys, xs)
            ])

        thresh_ch2 = np.nanpercentile(ref_mean, 90)
        matches, unmatched_A, unmatched_B, scores = drd1_cell_match(ref_mean, gcamp_rois)
        drd1_gcamp_idx = [match[1] for match in matches]
        drd1_match_gcamp = np.zeros(gcamp_rois.shape[0]).astype(bool)
        drd1_match_gcamp[drd1_gcamp_idx] = True
        drd1_label = (drd1_match_gcamp) & (com_vals>thresh_ch2)
        
        np.save(data_path/r'drd1_label.npy', drd1_label)
        
    else:
        drd1_label = np.load(data_path/r'drd1_label.npy')
        
    p_profile  = OUTPUT_RES/f'{anm}-{date}_raw_dff_profile_pre{pre_window}_post{post_window}.parquet'
    p_zscore_profile = OUTPUT_RES/f'{anm}-{date}_zscored_profile_pre{pre_window}_post{post_window}.parquet'
    df_profile =  pd.read_parquet( p_profile)
    df_zscored_profile =  pd.read_parquet(p_zscore_profile)

    df_profile['drd1+'] = drd1_label[df_profile['unit_id']]
    df_zscored_profile['drd1+'] = drd1_label[df_profile['unit_id']]
    df_profile.to_parquet(p_profile)
    df_zscored_profile.to_parquet(p_zscore_profile)
    # plot validation
    fig, ax = plt.subplots(figsize=(3,3), dpi=150)
    ax.imshow(mean_img_ch2,
              vmin=np.percentile(mean_img_ch2, 0.5),
              vmax=np.percentile(mean_img_ch2, 99.5),
              cmap = 'grey')
    drd1_cells = df_profile.loc[df_profile['drd1+'], 'unit_id'].to_list()
    for roi in drd1_cells:
        ax.imshow(np.where(roi_map_shifted[roi], 1, np.nan), 'Set1', alpha=.5)
    ax.set_axis_off()  
    save_fig(fig, Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\TEST_PLOTS\drd1_validation"),
             f'{anm}_{date}', save=1, forms=['png',])

