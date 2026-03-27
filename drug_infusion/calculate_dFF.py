# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:45:49 2026

@author: Jingyu Cao
"""

#%% imports 
from pathlib import Path
import sys 

import numpy as np
import cupy as cp
import pandas as pd
from tqdm import tqdm
import xarray as xr
from scipy.ndimage import shift as ndi_shift
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d

# add parent directories to path
directories = [
    'Z:/Jingyu/code_mpfi_Jingyu/common', 
    'Z:/Jingyu/code_mpfi_Jingyu/drug_infusion',
    'Z:/Dinghao/code_mpfi_dinghao/utils'
    ]
sys.path.extend(directories)

from common.utils_basic import trace_filter
from common.mask import generate_masks
from common.robust_sd_filter import robust_filter_along_axis
from common.trial_selection import select_good_trials, seperate_valid_trial
from common.event_response_quantification import quantify_event_response


#%%
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


def roi_map_to_list(roi_map):
    """
    Convert ROI map of shape (n_roi, H, W) into a list of dicts
    [{'npix': int, 'ypix': array, 'xpix': array}, ...]
    """
    roi_list = []
    n_roi = roi_map.shape[0]

    for i in range(n_roi):
        ypix, xpix = np.where(roi_map[i] > 0)  # coordinates where roi is active
        roi_list.append({
            'npix': len(ypix),
            'ypix': ypix,
            'xpix': xpix
        })
    return roi_list

#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"

pre_window=(-1.5, -0.5)
post_window=(0.5, 1.5)
thresh_pyrUp = 1.3
thresh_pyrDown = 1/thresh_pyrUp
active_thresh = 0
bef, aft = 2, 4
plot_single_session = 0

df_drug_pool = pd.DataFrame()
df_ctrl_pool = pd.DataFrame()
#%%
# import recording list
import drug_infusion.rec_lst_infusion as rec_info
# rec_drug, rec_drug_ctrl = rec_info.rec_SCH, rec_info.rec_SCH_ctrl
# for (rec_drug, rec_drug_ctrl) in [(rec_info.rec_SCH, rec_SCH_ctrl)
rec_drug = rec_info.rec_lst
error_lst = []
# Process each recording
for rec_idx, rec in tqdm(rec_drug.iterrows(), total=len(rec_drug), desc="Processing sessions"):
    anm = rec['anm']
    date = rec['date']
    data_path = OUT_DIR_RAW_DATA/'raw_signals'/f'{anm}-{date}'
    is_active_soma = np.load(data_path/r'soma_class.npz')['is_soma']
    print(f'\n{anm}-{date}')
    for ss in ['02', '04']:
        ops = np.load(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\suite2p_func_detec\plane0\ops.npz")
        beh = pd.read_pickle(OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{anm}-{date}-{ss}.pkl')
        frame_times = beh['frame_times']
        
        
        dff_ss1 = np.load(data_path/f'{anm}-{date}-02_dFF.npy')[is_active_soma]   
        dff_ss2 = np.load(data_path/f'{anm}-{date}-04_dFF.npy')[is_active_soma]
    