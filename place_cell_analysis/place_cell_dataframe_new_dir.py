# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 14:28:07 2026

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd

from place_cell_analysis import place_cell_functions as pcf
from place_cell_analysis.utils_trial_correlation import calculate_trial_correlations_gpu
from common.utils_basic import nearest_mapping
from common.robust_sd_filter import robust_filter_along_axis
#%% PATHS AND PARAMS
from drug_infusion import rec_lst_infusion as recs
rec_SCH = pd.concat((recs.rec_SCH, recs.rec_SCH_ctrl))
rec_praz = pd.concat((recs.rec_praz, recs.rec_praz_ctrl))
rec_prop = pd.concat((recs.rec_prop, recs.rec_prop_ctrl))
rec_all = pd.concat((rec_SCH, 
                     rec_praz,
                     rec_prop))
# session list 
# drug = 'SCH'
# drug = 'prazosin'
# drug = 'propranolol'
# if drug=='SCH':
#     rec_drug = recs.rec_SCH
#     rec_ctrl = recs.rec_SCH_ctrl
# elif drug=='prazosin':
#     rec_drug = recs.rec_praz
#     rec_ctrl = recs.rec_praz_ctrl
# elif drug=='propranolol':
#     rec_drug = recs.rec_prop
#     rec_ctrl = recs.rec_prop_ctrl

# Parameters
track_length = 180  # cm
bin_size = 4  # cm
n_bins = int(track_length / bin_size)  # 45 bins

# PATHS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
# OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
OUTPUT_RES = Path(r'Z:\Jingyu\GCaMP_drug_infusion\place_cell_dataframe_3rsd')

#%% for dLight-GECO learning animals
# OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\geco_dlight")
# OUTPUT_RES = OUT_DIR_RAW_DATA/'place_cell_dataframe_3rsd'

# rec_lst = [
# 'AC327-20260602-02',     
# 'AC327-20260603-02',     
# 'AC327-20260604-02',     
# 'AC327-20260605-02',     
# 'AC327-20260606-02',     
# 'AC327-20260607-02',     
# 'AC327-20260608-02',     
# 'AC327-20260609-02',     
# 'AC327-20260610-02',     
# 'AC327-20260611-02',
# 'AC327-20260612-02', 

# 'AC330-20260602-02',
# 'AC330-20260603-02', 
# 'AC330-20260604-02', 
# 'AC330-20260605-02', 
# 'AC330-20260606-02', 
# 'AC330-20260607-02', 
# 'AC330-20260608-02',
# 'AC330-20260609-02', 
# 'AC330-20260610-02',          
# 'AC330-20260611-02',          
# 'AC330-20260612-02',     
#     ]

#%%
rec_lst = [
'AC333-20260717-02',    

    ]

rsd_factor = 3
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\raw_data\lc_stim_gcamp\processed_data")
#%% Main
error_list = []

# for _, rec in rec_all.iterrows():
#     anm = rec['anm']
#     date = rec['date']
#     print(f'\n{anm}-{date}')
    
#     if not (anm=='AC989')&(date=='20250711'):
#         continue
#     # if not (anm=='AC310')&(date=='20250822'):
#     #     continue

for rec_id in rec_lst:
    anm, date, ss = rec_id.split('-')
    
    # data_path = OUT_DIR_RAW_DATA/'raw_signals'/f'{anm}-{date}'
    
    # for ss in ['02', '04']:
    # rec_id = f'{anm}-{date}-{ss}'
    print(f'processing {rec_id}----------------------')
    p_beh = OUT_DIR_RAW_DATA / rec_id / f'{rec_id}.pkl'
    data_base = Path(rf'Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\suite2p_func_detec\plane0')
    beh = pd.read_pickle(p_beh)
    
    # is_active_soma = np.load(data_path/r'soma_class.npz')['is_soma']
    dff = np.load(data_base/r"dff_ch1.npy") #[is_active_soma]
    
    # filter for extreme values
    thresh =pcf.dff_thresh(dff, hard_thresh=100, factor=5)
    dff_sd = robust_filter_along_axis(dff, factor=rsd_factor)
    dff_sd[abs(dff_sd)>thresh]=np.nan
    dff = dff_sd
    
    # try:
    # covert time dimension (frames) to distance
    running_time_map_img, running_distance_map_img, running_calcium_map_img, running_time_map_img_abs = pcf.align_run_frame_calcium(dff, beh) 
    if len(running_calcium_map_img) != 0:
        n_running_frames = running_calcium_map_img.shape[-1]
    
        # Spatial binning
        event_count_raw, event_rate_raw, occupancy_raw, per_lap_profile, per_lap_occupancy = pcf.spatial_binning(
            running_calcium_map_img, running_distance_map_img,
            track_length=track_length, bin_size=bin_size
        )

        # Per-cell event rate / occupancy from valid laps only (drop
        # laps with any NaN bin — e.g. dFF thresholded out)
        valid_trials_mask = ~np.any(np.isnan(per_lap_profile), axis=-1)
        event_count_valid, event_rate_valid, occupancy_valid = pcf.compute_valid_event_rate(
            per_lap_profile, per_lap_occupancy, valid_trials_mask
        )

        # Per-cell binned count of significant-transient frames (valid
        # laps only) — drives criterion (iv) of filter_tentative_fields
        # and is saved as a column so the filter can be re-run offline
        # with different thresholds.
        transient_count_per_bin = pcf.compute_transient_count_per_bin(
            running_calcium_map_img, running_distance_map_img,
            valid_trials_mask=valid_trials_mask,
            track_length=track_length, bin_size=bin_size,
        )

        # Detect place fields (place_field_map uses valid laps only)
        df_place_field = pcf.detect_place_field(
            event_count_valid, occupancy_valid,
            per_lap_profile=per_lap_profile,
            valid_trials_mask=valid_trials_mask,
            sigma=1.5, kernel_size=5, bin_size=bin_size,
            transient_count_per_bin=transient_count_per_bin,
        )

        total_active_frames = np.sum(running_calcium_map_img>0, axis=-1)
        df_place_field['total_active_frames'] = total_active_frames
        df_place_field['perc_active_frames'] = total_active_frames/n_running_frames
        active_laps = np.any(per_lap_profile > 0, axis=-1)  # (n_cells, n_laps)
        df_place_field['active_laps'] = list(active_laps)
        df_place_field['total_active_laps'] = np.sum(active_laps, axis=-1)
        df_place_field['perc_active_laps'] = np.sum(active_laps, axis=-1) / active_laps.shape[1]

        # Map running laps to behavioural trial indices
        lap_trial_idx = pcf.map_laps_to_trials(
            running_distance_map_img, running_time_map_img_abs,
            beh, track_length=track_length,
        )
        df_place_field['lap_trial_idx'] = [lap_trial_idx] * len(df_place_field)

        # Compute spatial information from valid trials only
        spatial_information = pcf.calculate_spatial_info(event_rate_valid, occupancy_valid)
        df_place_field['spatial_information_bits'] = spatial_information

        # Only compute shuffled SI for ROIs with SI > threshold
        SI_threshold_for_shuffle = 0.05
        candidate_mask = ((total_active_frames>0.1*n_running_frames)&
                          (spatial_information > SI_threshold_for_shuffle)&
                          (~np.isnan(spatial_information))
                          )
        candidate_indices = np.where(candidate_mask)[0]

        # Initialize shuffled_SI column with None
        df_place_field['shuffled_SI'] = None

        if len(candidate_indices) > 0:
            print(f"Computing shuffled SI for {len(candidate_indices)} ROIs with SI > {SI_threshold_for_shuffle}")
            candidate_indices, shuffled_SI = (
                pcf.calculate_spatial_info_shuffle_by_valid_laps(
                    running_calcium_map_img, running_distance_map_img,
                    valid_trials_mask, candidate_indices,
                    min_valid_laps=80, shuff_times=1000, min_shift=450,
                    gpu=True, track_length=track_length, bin_size=bin_size,
                )
            )
            for i, idx in enumerate(candidate_indices):
                df_place_field.at[idx, 'shuffled_SI'] = shuffled_SI[i]
        # Calculate trial-to-trial correlations (GPU-accelerated)
        print(f"  Computing trial correlations...")
        per_lap_profiles_list = df_place_field['per_lap_profile'].tolist()
        corr_results = calculate_trial_correlations_gpu(
            per_lap_profiles_list,
            methods=['odd_even', 'consecutive'],
            gpu=True
        )
        df_place_field['odd_even_corr'] = corr_results['odd_even']
        df_place_field['consecutive_corr'] = corr_results['consecutive']

        # Convert 2D arrays to lists for parquet compatibility
        if 'per_lap_profile' in df_place_field.columns:
            df_place_field['per_lap_profile'] = df_place_field['per_lap_profile'].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
        for col in ('tentative_field', 'final_field'):
            df_place_field[col] = df_place_field[col].apply(
                lambda fields: [f.tolist() for f in fields]
            )
        df_place_field['lap_trial_idx'] = df_place_field['lap_trial_idx'].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        for col in ('transient_count_per_bin', 'occupancy_per_bin'):
            df_place_field[col] = df_place_field[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
        for col in ('tentative_field_width_cm',
                    'tentative_field_peak_dff',
                    'tentative_field_in_out_ratio',
                    'tentative_field_transient_fraction'):
            df_place_field[col] = df_place_field[col].apply(
                lambda lst: [float(v) for v in lst]
            )
        
        OUTPUT_RES = (OUT_DIR_RAW_DATA / rec_id / 
                      f'{rec_id}_place_cell_dataframe_{rsd_factor}rsd.parquet')
        
        df_place_field.to_parquet(OUTPUT_RES)
                
                # # Extract values from DataFrame for convenience
                # n_cells = len(df_place_field)
                # place_field_position_cm = df_place_field['place_field_position_cm'].values
                # total_events = df_place_field['total_events'].values
        
                # print(f"\nPlace cell analysis complete:")
                # print(f"  Total cells: {n_cells}")
                # print(f"  Cells with >5 events: {np.sum(total_events > 5)}")
                # print(f"  Mean spatial information: {np.nanmean(spatial_information):.3f} bits/event")
        # except:
        #     error_list.append(rec_id)
        