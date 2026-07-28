# -*- coding: utf-8 -*-
"""
Time Cell Analysis - Data Processing Pipeline

Parallel to place_cell_dataframe.py but for temporal tuning analysis.
Processes each recording session and saves time cell dataframes as parquet files.

Time bin: 0.1s (3 frames at 30 Hz)

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd

from place_cell_analysis import time_cell_functions as tcf
from place_cell_analysis.utils_trial_correlation import calculate_trial_correlations_gpu
from common.utils_basic import nearest_mapping
from common.robust_sd_filter import robust_filter_along_axis

#%% PATHS AND PARAMS

# Session list
drug = 'SCH'
# drug = 'prazosin'
# drug = 'propranolol'

from drug_infusion import rec_lst_infusion as recs
if drug == 'SCH':
    rec_drug = recs.rec_SCH
    rec_ctrl = recs.rec_SCH_ctrl
elif drug == 'prazosin':
    rec_drug = recs.rec_praz
    rec_ctrl = recs.rec_praz_ctrl
elif drug == 'propranolol':
    rec_drug = recs.rec_prop
    rec_ctrl = recs.rec_prop_ctrl

# Parameters
time_bin_size = 0.1  # seconds (3 frames at 30Hz)
max_lap_duration_s = 4.0  # fixed time window in seconds (40 bins total)
frame_rate = 30  # Hz
track_length = 180  # cm (for lap detection)

# PATHS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
OUTPUT_RES = Path(r'Z:\Jingyu\GCaMP_drug_infusion\time_cell_dataframe')
OUTPUT_RES.mkdir(parents=True, exist_ok=True)

#%% Main
error_list = []

for _, rec in rec_ctrl.iterrows():
    anm = rec['anm']
    date = rec['date']
    print(f'\n{anm}-{date}')

    data_path = OUT_DIR_RAW_DATA / 'raw_signals' / f'{anm}-{date}'

    for ss in ['02', '04']:
        rec_id = f'{anm}-{date}-{ss}'
        print(f'processing {rec_id}----------------------')
        p_beh = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec_id}.pkl'
        beh = pd.read_pickle(p_beh)

        is_active_soma = np.load(data_path / r'soma_class.npz')['is_soma']
        dff = np.load(data_path / f'{rec_id}_dFF.npy')[is_active_soma]
        dff = robust_filter_along_axis(dff, factor=2.5)

        try:
            # Align calcium to behavior (running frames only)
            running_time_map_img, running_distance_map_img, running_calcium_map_img = tcf.align_run_frame_calcium(dff, beh)

            if len(running_calcium_map_img) != 0:
                n_running_frames = running_calcium_map_img.shape[-1]

                # Temporal binning (time within each lap, fixed 4s window)
                event_count_raw, event_rate_raw, occupancy_raw, per_lap_profile, per_lap_occupancy, n_time_bins, median_lap_duration_s = tcf.temporal_binning(
                    running_calcium_map_img, running_time_map_img, running_distance_map_img,
                    track_length=track_length, time_bin_size=time_bin_size,
                    max_lap_duration_s=max_lap_duration_s, frame_rate=frame_rate
                )

                if len(event_count_raw) == 0:
                    print(f"  Skipping {rec_id}: insufficient laps")
                    continue

                print(f"  n_time_bins: {n_time_bins}, median_lap_duration: {median_lap_duration_s:.2f}s")

                # Per-cell event rate / occupancy from valid laps only.
                # For time cells, only laps with a NaN bin where occupancy > 0
                # (extreme dFF in the middle of a lap) are excluded — trailing
                # NaNs from laps shorter than max_lap_duration_s are tolerated.
                valid_trials_mask = ~np.any(
                    np.isnan(per_lap_profile) & (per_lap_occupancy[None, :, :] > 0),
                    axis=-1,
                )
                _, event_rate_valid, occupancy_valid = tcf.compute_valid_event_rate(
                    per_lap_profile, per_lap_occupancy, valid_trials_mask
                )

                # Detect time fields
                df_time_field = tcf.detect_time_field(
                    event_count_raw, event_rate_raw, occupancy_raw,
                    per_lap_profile=per_lap_profile,
                    n_time_bins=n_time_bins,
                    median_lap_duration_s=median_lap_duration_s,
                    max_lap_duration_s=max_lap_duration_s,
                    sigma=1.5, kernel_size=5, time_bin_size=time_bin_size
                )

                total_active_frames = np.sum(running_calcium_map_img > 0, axis=-1)
                df_time_field['total_active_frames'] = total_active_frames

                # Compute temporal information from valid trials only
                temporal_information = tcf.calculate_temporal_info(event_rate_valid, occupancy_valid)
                df_time_field['temporal_information_bits'] = temporal_information

                # Store metadata
                df_time_field['n_time_bins'] = n_time_bins
                df_time_field['median_lap_duration_s'] = median_lap_duration_s

                # Only compute shuffled TI for ROIs with TI > threshold
                TI_threshold_for_shuffle = 0.05
                candidate_mask = ((total_active_frames > 0.1 * n_running_frames) &
                                  (temporal_information > TI_threshold_for_shuffle) &
                                  (~np.isnan(temporal_information)))
                candidate_indices = np.where(candidate_mask)[0]

                # Initialize shuffled_TI column with None
                df_time_field['shuffled_TI'] = None

                if len(candidate_indices) > 0:
                    print(f"Computing shuffled TI for {len(candidate_indices)} ROIs with TI > {TI_threshold_for_shuffle}")
                    shuffled_TI = tcf.calculate_temporal_info_shuffle(
                        running_calcium_map_img[candidate_indices],
                        running_time_map_img,
                        running_distance_map_img,
                        shuff_times=1000,
                        gpu=True,
                        track_length=track_length,
                        time_bin_size=time_bin_size,
                        max_lap_duration_s=max_lap_duration_s,
                        frame_rate=frame_rate
                    )
                    # Assign shuffled TI to candidate ROIs
                    for i, idx in enumerate(candidate_indices):
                        df_time_field.at[idx, 'shuffled_TI'] = shuffled_TI[i]

                # Calculate trial-to-trial correlations (GPU-accelerated)
                print(f"  Computing trial correlations...")
                per_lap_profiles_list = df_time_field['per_lap_profile'].tolist()
                corr_results = calculate_trial_correlations_gpu(
                    per_lap_profiles_list,
                    methods=['odd_even', 'consecutive'],
                    gpu=True
                )
                df_time_field['odd_even_corr'] = corr_results['odd_even']
                df_time_field['consecutive_corr'] = corr_results['consecutive']

                # Convert 2D arrays to lists for parquet compatibility
                if 'per_lap_profile' in df_time_field.columns:
                    df_time_field['per_lap_profile'] = df_time_field['per_lap_profile'].apply(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                    )

                df_time_field.to_parquet(OUTPUT_RES / f'{rec_id}_time_cell_dataframe.parquet')

                n_cells = len(df_time_field)
                print(f"  Time cell analysis complete: {n_cells} cells, mean TI: {np.nanmean(temporal_information):.3f} bits/event")

        except Exception as e:
            print(f"  Error processing {rec_id}: {e}")
            error_list.append(rec_id)

print(f"\n\nErrors occurred in: {error_list}")
