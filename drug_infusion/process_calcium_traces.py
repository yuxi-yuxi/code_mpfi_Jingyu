# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 18:00:44 2026
Modified on 3 Feb 2025

process traces for identified somata in drug infusion experiments 

@author: Jingyu Cao
@modifier: Dinghao Luo
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

from drd1_detection import drd1_cell_match

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

pre_window=(-1, 0)
post_window=(0.5, 1.5)
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
    print(f'\n{anm}-{date}')
    data_path = OUT_DIR_RAW_DATA/'raw_signals'/f'{anm}-{date}'

    
    if not(data_path/r'soma_class.npz').exists():
        if not(data_path/r'F_corr.npy').exists():
            sig_master = xr.open_dataarray(data_path/"sig_master_raw.nc")
            F_all =  sig_master.values.squeeze()
            sig_master_neu = xr.open_dataarray(data_path/"sig_master_neu_raw.nc")
            Fneu_all =  sig_master_neu.values.squeeze()
            F_corr = F_all-0.7*Fneu_all
            np.save(data_path/r'F_corr.npy', F_corr)
        else:
            F_corr = np.load(data_path/r'F_corr.npy')
        
        if not (data_path/r'gcamp_stats.npy').exists():
            # always use the first session for reference mean image
            p_suite2p_ss1 = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\02\suite2p_func_detec\plane0"
            suite2p_ss1_ops = np.load(p_suite2p_ss1+r'\ops.npy', allow_pickle=True).item()
            mean_img_ch1 = suite2p_ss1_ops['meanImg']
            # mean_img_ch2 = suite2p_ss1_ops['meanImg_chan2_corrected']
            
            A_master = xr.open_dataarray(data_path/"A_master.nc")
            roi_map = A_master.values.squeeze()
            shift_ds = xr.open_dataset(data_path/"shift_ds.nc")
            sh = shift_ds["shifts"].sel(animal=anm, session=f'{date}_02')
            roi_map_shifted = warp_rois_rigid(roi_map, (-sh).values)
            gcamp_stats = roi_map_to_list(roi_map_shifted)
            gcamp_stats = roi_map_to_list(roi_map)
            np.save(data_path/r'gcamp_stats.npy', np.asarray(gcamp_stats, dtype='object'))
        else:
            gcamp_stats = np.load(data_path/r'gcamp_stats.npy', allow_pickle=True)
            
        is_soma, is_active, is_active_soma = generate_masks.select_gcamp_rois(mean_img_ch1, F_corr,
                                         gcamp_stats, 
                                         path_result=r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\TEST_PLOTS")
        
        np.savez_compressed(
            data_path/r'soma_class.npz',
            is_soma=is_soma,
            is_active=is_active,
            is_active_soma=is_active_soma,
        )
    else:
        is_active_soma = np.load(data_path/r'soma_class.npz')['is_soma']
    

    #%% loading behaviour file
    # try:
    p_beh_ss1 = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{anm}-{date}-02.pkl'
    p_beh_ss2 = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{anm}-{date}-04.pkl'
    beh_ss1   = pd.read_pickle(p_beh_ss1)
    beh_ss2   = pd.read_pickle(p_beh_ss2)

    run_ss1       = np.array(beh_ss1['run_onset_frames'])
    run_valid_ss1 = run_ss1[(seperate_valid_trial(beh_ss1))&(run_ss1!=-1)]
    # run_good_ss1  = run_ss1[(select_good_trials(beh_ss1))&(run_ss1!=-1)]
    run_good_ss1 = run_ss1[(rec['good_trials_ss1'])&(run_ss1!=-1)]
    run_good_ss1[:10] = 0 # exclude first 10 trials due to imaging intensity drifting
    run_ss2       = np.array(beh_ss2['run_onset_frames'])
    run_valid_ss2 = run_ss2[(seperate_valid_trial(beh_ss2))&(run_ss2!=-1)]
    # run_good_ss2  = run_ss2[(select_good_trials(beh_ss2))&(run_ss2!=-1)]
    run_good_ss2 = run_ss2[(rec['good_trials_ss2'])&(run_ss2!=-1)]
    run_good_ss2[:10] = 0 # exclude first 10 trials due to imaging intensity drifting

    has_good_ss1 = np.sum(run_good_ss1) > 0
    has_good_ss2 = np.sum(run_good_ss2) > 0
    if not has_good_ss1:
        print(f'  WARNING: {anm}-{date} ss1 has no good trials after exclusion')
    if not has_good_ss2:
        print(f'  WARNING: {anm}-{date} ss2 has no good trials after exclusion')
    
    # # =============================================================================
    # #  calculate stats for raw dFF traces   
    # # =============================================================================
    # # only include active soma rois
    dff_ss1 = np.load(data_path/f'{anm}-{date}-02_dFF.npy')[is_active_soma]   
    dff_ss2 = np.load(data_path/f'{anm}-{date}-04_dFF.npy')[is_active_soma]
    dff_ss1_baseline = np.load(data_path/f'{anm}-{date}-02_baselines.npy')[is_active_soma] 
    dff_ss2_baseline = np.load(data_path/f'{anm}-{date}-04_baselines.npy')[is_active_soma] 
    f_all = np.load(data_path/'F_corr.npy')[is_active_soma]
    f_all_median = np.nanmedian(f_all, axis=-1)
    dff_ss1_baseline_min = np.nanmin(dff_ss1_baseline, axis=-1)
    dff_ss2_baseline_min = np.nanmin(dff_ss2_baseline, axis=-1)
    
    # dff_ss1_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss1, n_sd=5)
    # dff_ss2_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss2, n_sd=5)
    # thresh_dff_ss1 = np.nanmean(dff_ss1) + 5*np.nanstd(dff_ss1)
    # thresh_dff_ss2 = np.nanmean(dff_ss2) + 5*np.nanstd(dff_ss2)
    # dff_ss1_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss1, fix_thresh=thresh_dff_ss1)
    # dff_ss2_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss2, fix_thresh=thresh_dff_ss2)

    dff_ss1_sm = cp_gaussian_filter1d(cp.array(dff_ss1), 
                                                sigma=1).get()
    dff_ss2_sm = cp_gaussian_filter1d(cp.array(dff_ss2), 
                                                sigma=1).get()
    
    shuffle_params={'times': 1000,
                    'pre_event_window':  2, # seconds
                    'post_event_window': 4 
                    }
    
    # Define configurations for each stats calculation
    configs = {
        'valid_ss1': (dff_ss1_sm, run_valid_ss1),
        'valid_ss2': (dff_ss2_sm, run_valid_ss2),
    }
    if has_good_ss1:
        configs['good_ss1'] = (dff_ss1_sm, run_good_ss1)
    if has_good_ss2:
        configs['good_ss2'] = (dff_ss2_sm, run_good_ss2)

    drop_cols = ['roi_id', 'dilation_k',
                 'shuff_response_amplitude', 'shuff_effect_size', 'shuff_response_ratio']
    stats = {}
    for name, (traces, events) in configs.items():
        result = quantify_event_response(
                corrected_traces=traces,
                event_frames=events,
                baseline_window=pre_window,
                response_window=post_window,
                imaging_rate=30.0,
                shuffle_test=False,
                shuffle_params=shuffle_params
            )
        if result is not None:
            stats[name] = result.drop(columns=drop_cols, errors='ignore')

    # Build list of suffixed DataFrames; for missing good configs, create NaN placeholders
    n_active = int(np.sum(is_active_soma))
    stats_parts = []
    for name in ['valid_ss1', 'valid_ss2', 'good_ss1', 'good_ss2']:
        if name in stats:
            stats_parts.append(stats[name].add_suffix(f'_{name}'))
        else:
            # use columns from any existing stats entry as template
            template = next(iter(stats.values()))
            nan_df = pd.DataFrame(np.nan, index=range(n_active),
                                  columns=[f'{c}_{name}' for c in template.columns])
            stats_parts.append(nan_df)

    # Combine all with suffixes and add unit_id
    stats_combined = pd.concat(stats_parts, axis=1)
    stats_combined.insert(0, 'unit_id', np.where(is_active_soma)[0])
    # append baseline mean and F trace median for later filtering
    stats_combined['dff_baseline_min_ss1']=dff_ss1_baseline_min
    stats_combined['dff_baseline_min_ss2']=dff_ss2_baseline_min
    stats_combined['f_all_median']=f_all_median
    stats_combined.to_parquet(OUTPUT_RES/f'{anm}-{date}_raw_dff_profile_pre{pre_window}_post{post_window}.parquet')
    
    # =============================================================================
    #  calculate stats for zscored F traces   
    # =============================================================================
    # loading zscored traces
        
    zscore_ss1 = np.load(data_path/f'{anm}-{date}-02_zscored.npy')[is_active_soma]
    zscore_ss2 = np.load(data_path/f'{anm}-{date}-04_zscored.npy')[is_active_soma]
    
    # import matplotlib.pyplot as plt
    # from common.utils_imaging import align_trials
    # from common import plotting_functions_Jingyu as pf
    # a = align_trials(zscore_ss1, 'run', beh_ss1)
    # fig, ax = plt.subplots()
    # pf.plot_mean_trace(np.nanmean(a, axis=1), ax)
    
    # thresh_ss1 = np.nanmean(zscore_ss1) + 5*np.nanstd(zscore_ss1)
    # thresh_ss2 = np.nanmean(zscore_ss2) + 5*np.nanstd(zscore_ss2)
    # zscore_ss1_safe = np.apply_along_axis(trace_filter, axis=-1, arr=zscore_ss1, fix_thresh=thresh_ss1)
    # zscore_ss2_safe = np.apply_along_axis(trace_filter, axis=-1, arr=zscore_ss2, fix_thresh=thresh_ss2)
    
    # aa = align_trials(zscore_ss1_safe, 'run', beh_ss1)
    # fig, ax = plt.subplots()
    # pf.plot_mean_trace(np.nanmean(aa, axis=1), ax)
    
    zscore_ss1_sm = cp_gaussian_filter1d(cp.array(zscore_ss1), 
                                                sigma=1).get()
    zscore_ss2_sm = cp_gaussian_filter1d(cp.array(zscore_ss2), 
                                                sigma=1).get()
    
    shuffle_params={'times': 1000,
                    'pre_event_window':  2, # seconds
                    'post_event_window': 4 
                    }
    
    # Define configurations for each stats calculation
    configs_zscore = {
        'valid_ss1': (zscore_ss1_sm, run_valid_ss1),
        'valid_ss2': (zscore_ss2_sm, run_valid_ss2),
    }
    if has_good_ss1:
        configs_zscore['good_ss1'] = (zscore_ss1_sm, run_good_ss1)
    if has_good_ss2:
        configs_zscore['good_ss2'] = (zscore_ss2_sm, run_good_ss2)

    stats_zscore = {}
    for name, (traces, events) in configs_zscore.items():
        result = quantify_event_response(
                corrected_traces=traces,
                event_frames=events,
                baseline_window=pre_window,
                response_window=post_window,
                imaging_rate=30.0,
                shuffle_test=False,
                shuffle_params=shuffle_params
            )
        if result is not None:
            stats_zscore[name] = result.drop(columns=drop_cols, errors='ignore')

    # Build list of suffixed DataFrames; for missing good configs, create NaN placeholders
    stats_zscore_parts = []
    for name in ['valid_ss1', 'valid_ss2', 'good_ss1', 'good_ss2']:
        if name in stats_zscore:
            stats_zscore_parts.append(stats_zscore[name].add_suffix(f'_{name}'))
        else:
            template = next(iter(stats_zscore.values()))
            nan_df = pd.DataFrame(np.nan, index=range(n_active),
                                  columns=[f'{c}_{name}' for c in template.columns])
            stats_zscore_parts.append(nan_df)

    # Combine all with suffixes and add unit_id
    stats_zscore_combined = pd.concat(stats_zscore_parts, axis=1)
    stats_zscore_combined.insert(0, 'unit_id', np.where(is_active_soma)[0])
    # append baseline mean and F trace median for later filtering
    stats_zscore_combined['dff_baseline_min_ss1']=dff_ss1_baseline_min
    stats_zscore_combined['dff_baseline_min_ss2']=dff_ss2_baseline_min
    stats_zscore_combined['f_all_median']=f_all_median
    stats_zscore_combined.to_parquet(OUTPUT_RES/f'{anm}-{date}_zscored_profile_pre{pre_window}_post{post_window}.parquet')
    
    # b = np.stack(stats_zscore_combined['mean_profile_valid_ss1'])
    # fig, ax = plt.subplots()
    # pf.plot_mean_trace(b, ax)

    # # =============================================================================
    # #  calculate stats for robust sd filtered traces   
    # # =============================================================================
    # dff_ss1_rsd = robust_filter_along_axis(dff_ss1, gpu=1).get() # smoothed already, sigma=1
    # dff_ss2_rsd = robust_filter_along_axis(dff_ss2, gpu=1).get() # smoothed already, sigma=1
    
    # thresh_ss1 = np.nanmean(dff_ss1_rsd) + 5*np.nanstd(dff_ss1_rsd)
    # thresh_ss2 = np.nanmean(dff_ss2_rsd) + 5*np.nanstd(dff_ss2_rsd)
    # dff_ss1_rsd_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss1_rsd, fix_thresh=thresh_ss1)
    # dff_ss2_rsd_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss2_rsd, fix_thresh=thresh_ss2)
    
    # shuffle_params={'times': 1000,
    #                 'pre_event_window':  2, # seconds
    #                 'post_event_window': 4 
    #                 }
    
    # # Define configurations for each stats calculation
    # configs_rsd = {
    #     'valid_ss1': (dff_ss1_rsd_safe, run_valid_ss1),
    #     'valid_ss2': (dff_ss2_rsd_safe, run_valid_ss2),
    # }
    # if has_good_ss1:
    #     configs_rsd['good_ss1'] = (dff_ss1_rsd_safe, run_good_ss1)
    # if has_good_ss2:
    #     configs_rsd['good_ss2'] = (dff_ss2_rsd_safe, run_good_ss2)

    # stats_rsd = {}
    # for name, (traces, events) in configs_rsd.items():
    #     result = quantify_event_response(
    #             corrected_traces=traces,
    #             event_frames=events,
    #             baseline_window=pre_window,
    #             response_window=post_window,
    #             imaging_rate=30.0,
    #             shuffle_test=False,
    #             shuffle_params=shuffle_params
    #         )
    #     if result is not None:
    #         stats_rsd[name] = result.drop(columns=drop_cols, errors='ignore')

    # # Build list of suffixed DataFrames; for missing good configs, create NaN placeholders
    # stats_rsd_parts = []
    # for name in ['valid_ss1', 'valid_ss2', 'good_ss1', 'good_ss2']:
    #     if name in stats_rsd:
    #         stats_rsd_parts.append(stats_rsd[name].add_suffix(f'_{name}'))
    #     else:
    #         template = next(iter(stats_rsd.values()))
    #         nan_df = pd.DataFrame(np.nan, index=range(n_active),
    #                               columns=[f'{c}_{name}' for c in template.columns])
    #         stats_rsd_parts.append(nan_df)

    # # Combine all with suffixes and add unit_id
    # stats_rsd_combined = pd.concat(stats_rsd_parts, axis=1)
    # stats_rsd_combined.insert(0, 'unit_id', np.where(is_active_soma)[0])
    # # append baseline mean and F trace median for later filtering
    # stats_rsd_combined['dff_baseline_min_ss1']=dff_ss1_baseline_min
    # stats_rsd_combined['dff_baseline_min_ss2']=dff_ss2_baseline_min
    # stats_rsd_combined['f_all_median']=f_all_median
    # stats_rsd_combined.to_parquet(OUTPUT_RES/f'{anm}-{date}_rsd_dff_profile_pre{pre_window}_post{post_window}.parquet')
    # # except:
    # #     print('!!!ERROR')
    # #     error_lst.append(f'{anm}-{date}')




    
    
    
    