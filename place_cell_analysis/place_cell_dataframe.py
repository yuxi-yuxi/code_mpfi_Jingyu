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

# session list 
drug = 'SCH'
# drug = 'prazosin'
# drug = 'propranolol'

from drug_infusion import rec_lst_infusion as recs
if drug=='SCH':
    rec_drug = recs.rec_SCH
    rec_ctrl = recs.rec_SCH_ctrl
elif drug=='prazosin':
    rec_drug = recs.rec_praz
    rec_ctrl = recs.rec_praz_ctrl
elif drug=='propranolol':
    rec_drug = recs.rec_prop
    rec_ctrl = recs.rec_prop_ctrl

# Parameters
track_length = 180  # cm
bin_size = 4  # cm
n_bins = int(track_length / bin_size)  # 45 bins

# PATHS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
# OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
OUTPUT_RES = Path(r'Z:\Jingyu\GCaMP_drug_infusion\place_cell_dataframe')
#%% Main
error_list = []

for _, rec in rec_ctrl.iterrows():
    anm = rec['anm']
    date = rec['date']
    print(f'\n{anm}-{date}')
    
    data_path = OUT_DIR_RAW_DATA/'raw_signals'/f'{anm}-{date}'
    
    for ss in ['02', '04']:
        rec_id = f'{anm}-{date}-{ss}'
        print(f'processing {rec_id}----------------------')
        p_beh = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec_id}.pkl'
        beh = pd.read_pickle(p_beh)
        
        is_active_soma = np.load(data_path/r'soma_class.npz')['is_soma']
        dff = np.load(data_path/f'{rec_id}_dFF.npy')[is_active_soma]
        dff = robust_filter_along_axis(dff, factor=2.5)
        try:
            # covert time dimension (frames) to distance
            running_time_map_img, running_distance_map_img, running_calcium_map_img = pcf.align_run_frame_calcium(dff, beh) 
            if len(running_calcium_map_img) != 0:
                n_running_frames = running_calcium_map_img.shape[-1]
            
                # Spatial binning
                event_count_raw, event_rate_raw, occupancy_raw, per_lap_profile = pcf.spatial_binning(
                    running_calcium_map_img, running_distance_map_img,
                    track_length=track_length, bin_size=bin_size
                )
            
                # Detect place fields
                df_place_field = pcf.detect_place_field(
                    event_count_raw, event_rate_raw, occupancy_raw,
                    per_lap_profile=per_lap_profile,
                    sigma=1.5, kernel_size=5, bin_size=bin_size
                )

                total_active_frames = np.sum(running_calcium_map_img>0, axis=-1)
                df_place_field['total_active_frames'] = total_active_frames
                # Compute spatial information
                spatial_information = pcf.calculate_spatial_info(event_rate_raw, occupancy_raw)
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
                    shuffled_SI = pcf.calculate_spatial_info_shuffle(
                        running_calcium_map_img[candidate_indices],
                        running_distance_map_img,
                        occupancy_raw, shuff_times=1000, gpu=True
                    )
                    # Assign shuffled SI to candidate ROIs
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
                df_place_field.to_parquet(OUTPUT_RES / f'{rec_id}_place_cell_dataframe.parquet')
                
                # # Extract values from DataFrame for convenience
                # n_cells = len(df_place_field)
                # place_field_position_cm = df_place_field['place_field_position_cm'].values
                # total_events = df_place_field['total_events'].values
        
                # print(f"\nPlace cell analysis complete:")
                # print(f"  Total cells: {n_cells}")
                # print(f"  Cells with >5 events: {np.sum(total_events > 5)}")
                # print(f"  Mean spatial information: {np.nanmean(spatial_information):.3f} bits/event")
        except:
            error_list.append(rec_id)
        