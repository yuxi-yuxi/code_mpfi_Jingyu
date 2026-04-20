# -*- coding: utf-8 -*-
"""
Time Cell Analysis - Visualization and Analysis

Parallel to plot_place_cell_analysis.py but for temporal tuning analysis.
Loads time cell dataframes and generates visualizations.

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from place_cell_analysis import time_cell_functions as tcf
from common.utils_basic import nearest_mapping, normalize
from common.robust_sd_filter import robust_filter_along_axis
import common.plotting_functions_Jingyu as pf
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

# Display time range (seconds)
TIME_RANGE_S = 6.0  # Plot 0-4 seconds for all sessions

# Use GPU-accelerated batch calculation for specified correlation types
# Options: 'odd_even', 'mean_pairwise', 'consecutive' (or None for all)
corr_methods = ['odd_even', 'consecutive']

# PATHS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\GCaMP_drug_infusion")
OUTPUT_RES = OUT_DIR_RAW_DATA / 'time_cell_dataframe'

#%% Load data for all sessions
df_time_field_drug_ss1 = pd.DataFrame()
df_time_field_drug_ss2 = pd.DataFrame()
# drug = 'ctrl'
for _, rec in rec_drug.iterrows():
    anm = rec['anm']
    date = rec['date']
    print(f'\n{anm}-{date}')

    # Session 1 (baseline)
    ss = '02'
    rec_id = f'{anm}-{date}-{ss}'
    print(f'loading {rec_id}----------------------')
    try:
        parquet_path = OUTPUT_RES / f'{rec_id}_time_cell_dataframe.parquet'
        df_time_field_ss1 = pd.read_parquet(parquet_path)
        df_time_field_ss1['rec_date'] = f'{anm}-{date}'
        df_time_field_ss1['anm'] = anm
        df_time_field_ss1['date'] = date
        df_time_field_ss1[f'{drug}_days'] = rec[f'{drug}_days']
        # Calculate time cell stability if not already computed
        if 'odd_even_corr' not in df_time_field_ss1.columns or 'consecutive_corr' not in df_time_field_ss1.columns:
            print(f"  Computing trial correlations for {rec_id}...")
            per_lap_profiles_ss1 = df_time_field_ss1['per_lap_profile'].tolist()
            corr_results_ss1 = tcf.calculate_all_trial_correlations_gpu(per_lap_profiles_ss1, methods=corr_methods, gpu=True)
            df_time_field_ss1['odd_even_corr'] = corr_results_ss1['odd_even']
            df_time_field_ss1['consecutive_corr'] = corr_results_ss1['consecutive']
            # Save updated dataframe
            df_time_field_ss1.drop(columns=['rec_date', 'anm', 'date', f'{drug}_days']).to_parquet(parquet_path)
            print(f"  Saved updated dataframe to {parquet_path}")
        df_time_field_drug_ss1 = pd.concat((df_time_field_drug_ss1, df_time_field_ss1))
    except Exception as e:
        print(f"  Error loading {rec_id}: {e}")

    # Session 2 (drug)
    ss = '04'
    rec_id = f'{anm}-{date}-{ss}'
    print(f'loading {rec_id}----------------------')
    try:
        parquet_path = OUTPUT_RES / f'{rec_id}_time_cell_dataframe.parquet'
        df_time_field_ss2 = pd.read_parquet(parquet_path)
        df_time_field_ss2['rec_date'] = f'{anm}-{date}'
        df_time_field_ss2['anm'] = anm
        df_time_field_ss2['date'] = date
        df_time_field_ss2[f'{drug}_days'] = rec[f'{drug}_days']
        # Calculate time cell stability if not already computed
        if 'odd_even_corr' not in df_time_field_ss2.columns or 'consecutive_corr' not in df_time_field_ss2.columns:
            print(f"  Computing trial correlations for {rec_id}...")
            per_lap_profiles_ss2 = df_time_field_ss2['per_lap_profile'].tolist()
            corr_results_ss2 = tcf.calculate_all_trial_correlations_gpu(per_lap_profiles_ss2, methods=corr_methods, gpu=True)
            df_time_field_ss2['odd_even_corr'] = corr_results_ss2['odd_even']
            df_time_field_ss2['consecutive_corr'] = corr_results_ss2['consecutive']
            # Save updated dataframe
            df_time_field_ss2.drop(columns=['rec_date', 'anm', 'date', f'{drug}_days']).to_parquet(parquet_path)
            print(f"  Saved updated dataframe to {parquet_path}")
        df_time_field_drug_ss2 = pd.concat((df_time_field_drug_ss2, df_time_field_ss2))
    except Exception as e:
        print(f"  Error loading {rec_id}: {e}")

# Load control sessions
# df_time_field_ctrl_ss1 = pd.DataFrame()
# df_time_field_ctrl_ss2 = pd.DataFrame()

# for _, rec in rec_ctrl.iterrows():
#     anm = rec['anm']
#     date = rec['date']
#     print(f'\n{anm}-{date}')

#     # Session 1 (baseline)
#     ss = '02'
#     rec_id = f'{anm}-{date}-{ss}'
#     print(f'loading {rec_id}----------------------')
#     try:
#         df_time_field_ss1 = pd.read_parquet(OUTPUT_RES / f'{rec_id}_time_cell_dataframe.parquet')
#         df_time_field_ss1['rec_date'] = f'{anm}-{date}'
#         df_time_field_ss1['anm'] = anm
#         df_time_field_ss1['date'] = date
#         df_time_field_ss1['ctrl_days'] = rec['ctrl_days']
#         df_time_field_ctrl_ss1 = pd.concat((df_time_field_ctrl_ss1, df_time_field_ss1))
#     except Exception as e:
#         print(f"  Error loading {rec_id}: {e}")

#     # Session 2 (ctrl)
#     ss = '04'
#     rec_id = f'{anm}-{date}-{ss}'
#     print(f'loading {rec_id}----------------------')
#     try:
#         df_time_field_ss2 = pd.read_parquet(OUTPUT_RES / f'{rec_id}_time_cell_dataframe.parquet')
#         df_time_field_ss2['rec_date'] = f'{anm}-{date}'
#         df_time_field_ss2['anm'] = anm
#         df_time_field_ss2['date'] = date
#         df_time_field_ss2['ctrl_days'] = rec['ctrl_days']
#         df_time_field_ctrl_ss2 = pd.concat((df_time_field_ctrl_ss2, df_time_field_ss2))
#     except Exception as e:
#         print(f"  Error loading {rec_id}: {e}")

#%% Selection recordings for statistics
# Select only the first 3 recording dates per animal for drug
df_time_field_drug_ss1 = df_time_field_drug_ss1.loc[df_time_field_drug_ss1[f'{drug}_days'] < 4]
df_time_field_drug_ss2 = df_time_field_drug_ss2.loc[df_time_field_drug_ss2[f'{drug}_days'] < 4]

# Select only the first 3 recording dates per animal for ctrl, and only animals in drug
# animals_in_drug = df_time_field_drug_ss1['anm'].unique()
# df_time_field_ctrl_ss1 = df_time_field_ctrl_ss1.loc[
#     (df_time_field_ctrl_ss1['ctrl_days'] < 4) &
#     (df_time_field_ctrl_ss1['anm'].isin(animals_in_drug))]
# df_time_field_ctrl_ss2 = df_time_field_ctrl_ss2.loc[
#     (df_time_field_ctrl_ss2['ctrl_days'] < 4) &
#     (df_time_field_ctrl_ss2['anm'].isin(animals_in_drug))]

# Use drug data for analysis (rename for compatibility with rest of script)
df_time_field_all_ss1 = df_time_field_drug_ss1
df_time_field_all_ss2 = df_time_field_drug_ss2

#%% Define time cells and find shared cells

TI_threshold = 0.3
shuff_TI_thresh = 99

key = ['rec_date', 'cell_id']
shared_keys = (
    df_time_field_all_ss1[key]
    .merge(df_time_field_all_ss2[key], on=key, how='inner')
    .drop_duplicates()
)

df_ss1_shared = df_time_field_all_ss1.merge(shared_keys, on=key, how='inner')
df_ss2_shared = df_time_field_all_ss2.merge(shared_keys, on=key, how='inner')


def select_time_cell(df, TI_threshold, shuff_TI_thresh):
    """Select time cells based on temporal information threshold and shuffle significance."""
    df['shuffle_TI_thresh'] = df['shuffled_TI'].apply(
        lambda x: np.nanpercentile(x, shuff_TI_thresh) if x is not None else np.nan)
    df['is_time_cell'] = ((df['temporal_information_bits'] > TI_threshold) &
                          (df['temporal_information_bits'] > df['shuffle_TI_thresh']))
    return df


df_ss1_shared = select_time_cell(df_ss1_shared, TI_threshold, shuff_TI_thresh)
df_ss2_shared = select_time_cell(df_ss2_shared, TI_threshold, shuff_TI_thresh)

df_time_cell_ss1 = df_ss1_shared.loc[df_ss1_shared['is_time_cell']]
df_time_cell_ss2 = df_ss2_shared.loc[df_ss2_shared['is_time_cell']]

print(f"\nTime cells found:")
print(f"  SS1 (baseline): {len(df_time_cell_ss1)}")
print(f"  SS2 (drug): {len(df_time_cell_ss2)}")

#%% Time cell sequence heatmaps - sorted by time field position

cell_order_ss1 = (
    df_time_cell_ss1.sort_values('time_field_position_s')[key]
    .drop_duplicates()
    .apply(tuple, axis=1)
    .to_numpy()
)

cell_order_ss2 = (
    df_time_cell_ss2.sort_values('time_field_position_s')[key]
    .drop_duplicates()
    .apply(tuple, axis=1)
    .to_numpy()
)

df_ss1_ordered = (
    df_time_cell_ss1.set_index(key)
    .reindex(cell_order_ss1)
    .reset_index()
)

df_ss2_ordered = (
    df_time_cell_ss2.set_index(key)
    .reindex(cell_order_ss2)
    .reset_index()
)

# Plot heatmaps with actual time (0-4s)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax, df_ordered, label in [(axes[0], df_ss1_ordered, 'SS1 (baseline)'),
                               (axes[1], df_ss2_ordered, 'SS2 (drug)')]:
    if len(df_ordered) > 0:
        imshow_array = np.stack(df_ordered['time_field_map_norm'])
        n_cells_shown = imshow_array.shape[0]
        n_time_bins = imshow_array.shape[1]

        # Get median lap duration for this session to scale x-axis
        median_lap_dur = df_ordered['median_lap_duration_s'].median()

        im = ax.imshow(imshow_array, aspect='auto', cmap='Greys',
                       extent=[0, TIME_RANGE_S, n_cells_shown, 0],
                       interpolation='nearest')
        ax.set_xlabel('Time in Lap (s)')
        ax.set_ylabel('Cell # (sorted by time field)')
        ax.set_title(f'{label}\n(n={n_cells_shown} time cells)')
        plt.colorbar(im, ax=ax, label='Normalized activity')

plt.suptitle(f'{drug} - Time Cell Sequences', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#%% Time cell stability comparison: SS1 (baseline) vs SS2 (drug)
key_stab = 'odd_even_corr'
# key_stab = 'consecutive_corr'
stability_ss1 = df_time_cell_ss1[key_stab].dropna().values
stability_ss2 = df_time_cell_ss2[key_stab].dropna().values
stability_ss1_mean = df_time_cell_ss1.groupby('rec_date')[key_stab].mean()
stability_ss2_mean = df_time_cell_ss2.groupby('rec_date')[key_stab].mean()
# get common dates
common_dates = stability_ss1_mean.index.intersection(stability_ss2_mean.index)
# subset both
stability_ss1_mean = stability_ss1_mean.loc[common_dates]
stability_ss2_mean = stability_ss2_mean.loc[common_dates]

print(f"\n=== Time Cell Stability Analysis ===")
print(f"SS1 (baseline): n={len(stability_ss1)}, mean={np.mean(stability_ss1):.3f}, median={np.median(stability_ss1):.3f}")
print(f"SS2 (drug):     n={len(stability_ss2)}, mean={np.mean(stability_ss2):.3f}, median={np.median(stability_ss2):.3f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(6, 4))

# 1. Box plot comparison
ax1 = axes[0]

pf.plot_bar_with_paired_scatter(ax1, stability_ss1_mean, stability_ss2_mean)
# ax1.set_ylabel('Stability (odd-even correlation)')


# 2. Histogram comparison
ax2 = axes[1]
bins = np.linspace(-0.5, 1, 20)
if len(stability_ss1) > 0:
    ax2.hist(stability_ss1, bins=bins, alpha=0.5, label=f'SS1 (n={len(stability_ss1)})', color='blue')
    ax2.axvline(np.median(stability_ss1), color='blue', linestyle='--', label=f'SS1 median: {np.median(stability_ss1):.2f}')
if len(stability_ss2) > 0:
    ax2.hist(stability_ss2, bins=bins, alpha=0.5, label=f'SS2 (n={len(stability_ss2)})', color='red')
    ax2.axvline(np.median(stability_ss2), color='red', linestyle='--', label=f'SS2 median: {np.median(stability_ss2):.2f}')
ax2.set_xlabel('Stability (odd-even correlation)')
ax2.set_ylabel('Count')
ax2.set_title('Stability Distribution')
ax2.legend()

plt.suptitle(f'{drug} - Time Cell Stability: Baseline vs Drug', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#%% Detailed visualization for each session

for label, df_raw in [('ss1', df_time_field_all_ss1), ('ss2', df_time_field_all_ss2)]:
    df = df_raw.reset_index(drop=True)

    # Apply time cell selection
    df['shuffle_TI_thresh'] = df['shuffled_TI'].apply(
        lambda x: np.nanpercentile(x, shuff_TI_thresh) if x is not None else np.nan
    )
    df['is_time_cell'] = ((df['temporal_information_bits'] > TI_threshold) &
                          (df['temporal_information_bits'] > df['shuffle_TI_thresh']))

    time_cell_indices = np.where(df['is_time_cell'])[0]
    time_field_pos_s = df['time_field_position_s'].values  # Use actual time in seconds
    temporal_information_all = df['temporal_information_bits'].values

    # Sort by time field position
    sort_order = np.argsort(time_field_pos_s[time_cell_indices])
    sorted_indices = time_cell_indices[sort_order]

    print(f"\n{label}: {len(time_cell_indices)} time cells (TI > {TI_threshold})")

    if len(sorted_indices) == 0:
        print(f"  No time cells found in {label}")
        continue
                                              
    # Build heatmap
    time_fields_sorted = np.vstack(df.loc[sorted_indices, 'time_field_map_norm'].values)
    n_cells_shown = len(sorted_indices)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Time cell sequence heatmap (0-4s)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(time_fields_sorted, aspect='auto', cmap='Greys',
                     extent=[0, TIME_RANGE_S, n_cells_shown, 0],
                     interpolation='nearest')
    ax1.set_xlabel('Time in Lap (s)')
    ax1.set_ylabel('Cell # (sorted by time field)')
    ax1.set_title(f'Time Cell Sequence\n({n_cells_shown} cells)')
    plt.colorbar(im1, ax=ax1, label='Normalized activity')

    # 2. Distribution of time field positions (0-4s)
    ax2 = axes[0, 1]
    ax2.hist(time_field_pos_s[time_cell_indices], bins=20,
             range=(0, TIME_RANGE_S), edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Time field position (s)')
    ax2.set_ylabel('Number of time cells')
    ax2.set_title(f'Time Field Distribution (TI > {TI_threshold})')
    ax2.set_xlim([0, TIME_RANGE_S])

    # 3. Temporal information distribution
    ax3 = axes[1, 0]
    ti_valid = temporal_information_all[~np.isnan(temporal_information_all)]
    ax3.hist(ti_valid, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(TI_threshold, color='g', linestyle='-', linewidth=2, label=f'TI = {TI_threshold} threshold')
    ax3.axvline(np.median(ti_valid), color='r', linestyle='--',
                label=f'Median: {np.median(ti_valid):.2f}')
    ax3.set_xlabel('Temporal information (bits/event)')
    ax3.set_ylabel('Number of cells')
    ax3.set_title('Temporal Information Distribution')
    ax3.legend()

    # 4. TI vs shuffle threshold scatter
    ax4 = axes[1, 1]
    shuffle_thresh_all = df['shuffle_TI_thresh'].values
    valid_mask = ~np.isnan(shuffle_thresh_all)
    ax4.scatter(temporal_information_all[valid_mask], shuffle_thresh_all[valid_mask],
                alpha=0.3, s=10)
    ax4.plot([0, np.nanmax(temporal_information_all)], [0, np.nanmax(temporal_information_all)],
             'r--', label='TI = shuffle threshold')
    ax4.set_xlabel('Temporal information (bits)')
    ax4.set_ylabel('99th percentile shuffle TI')
    ax4.set_title('TI vs Shuffle Threshold')
    ax4.legend()

    plt.suptitle(f'{label} — All recordings (n={len(df)} cells)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#%% Example time cells (top 6 by temporal information)

for label, df_raw in [('ss1', df_time_field_all_ss1), ('ss2', df_time_field_all_ss2)]:
    df = df_raw.reset_index(drop=True)

    df['shuffle_TI_thresh'] = df['shuffled_TI'].apply(
        lambda x: np.nanpercentile(x, shuff_TI_thresh) if x is not None else np.nan
    )
    df['is_time_cell'] = ((df['temporal_information_bits'] > TI_threshold) &
                          (df['temporal_information_bits'] > df['shuffle_TI_thresh']))

    time_cell_indices = np.where(df['is_time_cell'])[0]
    temporal_information_all = df['temporal_information_bits'].values

    if len(time_cell_indices) < 6:
        print(f"  Not enough time cells in {label} for example plots")
        continue

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    # Get top 6 time cells by temporal information
    top_ti_order = np.argsort(temporal_information_all[time_cell_indices])[::-1][:6]
    top_time_cells = time_cell_indices[top_ti_order]

    for i, cell_idx in enumerate(top_time_cells):
        ax = axes[i]
        time_field_map = df.loc[cell_idx, 'time_field_map']
        n_bins = len(time_field_map)
        # Use actual time (0-4s) for x-axis
        time_bins = np.linspace(0, TIME_RANGE_S, n_bins)
        bin_width = TIME_RANGE_S / n_bins * 0.8

        ax.bar(time_bins, time_field_map, width=bin_width, alpha=0.7)
        ax.axvline(df.loc[cell_idx, 'time_field_position_s'], color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Event rate')
        ax.set_title(f'Cell {cell_idx}, TI={temporal_information_all[cell_idx]:.2f} bits')
        ax.set_xlim([0, TIME_RANGE_S])

    plt.suptitle(f'{label} - Top 6 Time Cells (TI > {TI_threshold})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#%% Per-lap temporal profiles for example time cells

for label, df_raw in [('ss1', df_time_field_all_ss1)]:
    df = df_raw.reset_index(drop=True)

    df['shuffle_TI_thresh'] = df['shuffled_TI'].apply(
        lambda x: np.nanpercentile(x, shuff_TI_thresh) if x is not None else np.nan
    )
    df['is_time_cell'] = ((df['temporal_information_bits'] > TI_threshold) &
                          (df['temporal_information_bits'] > df['shuffle_TI_thresh']))

    time_cell_indices = np.where(df['is_time_cell'])[0]

    if len(time_cell_indices) < 4:
        continue

    # Show per-lap profiles for top 4 time cells
    top_ti_order = np.argsort(df.loc[time_cell_indices, 'temporal_information_bits'].values)[::-1][:4]
    top_time_cells = time_cell_indices[top_ti_order]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, cell_idx in enumerate(top_time_cells):
        ax = axes[i]
        per_lap_profile = df.loc[cell_idx, 'per_lap_profile']

        if per_lap_profile is None:
            continue

        # Convert to numpy array if needed
        if isinstance(per_lap_profile, list):
            per_lap_profile = np.array(per_lap_profile)

        if per_lap_profile.ndim != 2:
            continue

        # Normalize for display
        per_lap_norm = normalize(per_lap_profile)

        n_laps, n_bins = per_lap_profile.shape
        # Use actual time (0-4s) for x-axis
        im = ax.imshow(per_lap_norm, aspect='auto', cmap='viridis',
                       extent=[0, TIME_RANGE_S, n_laps, 0],
                       interpolation='none')

        time_field_pos_s = df.loc[cell_idx, 'time_field_position_s']
        TI = df.loc[cell_idx, 'temporal_information_bits']

        ax.axvline(time_field_pos_s, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Time in Lap (s)')
        ax.set_ylabel('Lap #')
        ax.set_title(f'Cell {cell_idx}, TI={TI:.3f}, peak={time_field_pos_s:.2f}s')
        plt.colorbar(im, ax=ax, label='Normalized activity')

    plt.suptitle(f'{label} - Per-Lap Temporal Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#%% Per-date heatmaps and stability comparison

unique_dates = sorted(
    set(df_time_field_all_ss1['rec_date'].unique())
    | set(df_time_field_all_ss2['rec_date'].unique())
)

for rec_date in unique_dates:
    print(f"\n{'='*60}")
    print(f"  {rec_date}")
    print(f"{'='*60}")

    # Subset data for this date
    df_date_ss1 = df_time_field_all_ss1.loc[
        df_time_field_all_ss1['rec_date'] == rec_date
    ].reset_index(drop=True)
    df_date_ss2 = df_time_field_all_ss2.loc[
        df_time_field_all_ss2['rec_date'] == rec_date
    ].reset_index(drop=True)

    if len(df_date_ss1) == 0 and len(df_date_ss2) == 0:
        print("  No data for either session, skipping.")
        continue

    # Apply time cell selection per date
    for df_date in [df_date_ss1, df_date_ss2]:
        if len(df_date) == 0:
            continue
        df_date['shuffle_TI_thresh'] = df_date['shuffled_TI'].apply(
            lambda x: np.nanpercentile(x, shuff_TI_thresh) if x is not None else np.nan
        )
        df_date['is_time_cell'] = (
            (df_date['temporal_information_bits'] > TI_threshold)
            & (df_date['temporal_information_bits'] > df_date['shuffle_TI_thresh'])
        )

    tc_ss1 = df_date_ss1.loc[df_date_ss1['is_time_cell']] if len(df_date_ss1) > 0 else pd.DataFrame()
    tc_ss2 = df_date_ss2.loc[df_date_ss2['is_time_cell']] if len(df_date_ss2) > 0 else pd.DataFrame()

    print(f"  SS1 time cells: {len(tc_ss1)},  SS2 time cells: {len(tc_ss2)}")

    # --- Heatmaps sorted by time field position ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    for ax, tc_df, label in [(axes[0], tc_ss1, 'SS1 (baseline)'),
                              (axes[1], tc_ss2, 'SS2 (drug)')]:
        if len(tc_df) > 0:
            tc_sorted = tc_df.sort_values('time_field_position_s')
            imshow_array = np.stack(tc_sorted['time_field_map_norm'].values)
            n_cells_shown = imshow_array.shape[0]
            im = ax.imshow(imshow_array, aspect='auto', cmap='Greys',
                           extent=[0, TIME_RANGE_S, n_cells_shown, 0],
                           interpolation='nearest')
            ax.set_xlabel('Time in Lap (s)')
            ax.set_ylabel('Cell # (sorted by time field)')
            ax.set_title(f'{label}\n(n={n_cells_shown} time cells)')
            plt.colorbar(im, ax=ax, label='Normalized activity')
        else:
            ax.set_title(f'{label}\n(no time cells)')
            ax.set_visible(False)

    fig.suptitle(f'{rec_date} — {drug} — Time Cell Sequences',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # --- Stability comparison ---
    stab_ss1 = tc_ss1['stability'].dropna().values if len(tc_ss1) > 0 else np.array([])
    stab_ss2 = tc_ss2['stability'].dropna().values if len(tc_ss2) > 0 else np.array([])

    if len(stab_ss1) == 0 and len(stab_ss2) == 0:
        print("  No stability data for either session, skipping comparison.")
        continue

    # Stats
    if len(stab_ss1) > 0 and len(stab_ss2) > 0:
        stat_mw_date, p_mw_date = stats.mannwhitneyu(
            stab_ss1, stab_ss2, alternative='two-sided'
        )
        p_str = f'p={p_mw_date:.4f}'
    else:
        p_mw_date = np.nan
        p_str = 'N/A (single session)'

    print(f"  Stability — SS1: n={len(stab_ss1)}, median={np.median(stab_ss1):.3f}" if len(stab_ss1) > 0 else "  Stability — SS1: no data")
    print(f"  Stability — SS2: n={len(stab_ss2)}, median={np.median(stab_ss2):.3f}" if len(stab_ss2) > 0 else "  Stability — SS2: no data")
    print(f"  Mann-Whitney {p_str}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Box plot
    ax1 = axes[0]
    # box_data = []
    # box_labels = []
    # box_colors = []
    # if len(stab_ss1) > 0:
    #     box_data.append(stab_ss1)
    #     box_labels.append('SS1 (baseline)')
    #     box_colors.append('lightblue')
    # if len(stab_ss2) > 0:
    #     box_data.append(stab_ss2)
    #     box_labels.append('SS2 (drug)')
    #     box_colors.append('lightcoral')

    # bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    # for patch, color in zip(bp['boxes'], box_colors):
    #     patch.set_facecolor(color)
    # ax1.set_ylabel('Stability (odd-even correlation)')
    # ax1.set_title(f'Time Cell Stability\nMann-Whitney {p_str}')

    # for i, data in enumerate(box_data, 1):
    #     x = np.random.normal(i, 0.04, size=len(data))
    #     ax1.scatter(x, data, alpha=0.3, s=10, color='black')


    # Histogram
    ax2 = axes[1]
    bins = np.linspace(-0.5, 1, 20)
    if len(stab_ss1) > 0:
        ax2.hist(stab_ss1, bins=bins, alpha=0.5,
                 label=f'SS1 (n={len(stab_ss1)})', color='blue')
        ax2.axvline(np.median(stab_ss1), color='blue', linestyle='--',
                    label=f'SS1 median: {np.median(stab_ss1):.2f}')
    if len(stab_ss2) > 0:
        ax2.hist(stab_ss2, bins=bins, alpha=0.5,
                 label=f'SS2 (n={len(stab_ss2)})', color='red')
        ax2.axvline(np.median(stab_ss2), color='red', linestyle='--',
                    label=f'SS2 median: {np.median(stab_ss2):.2f}')
    ax2.set_xlabel('Stability (odd-even correlation)')
    ax2.set_ylabel('Count')
    ax2.set_title('Stability Distribution')
    ax2.legend()

    fig.suptitle(f'{rec_date} — {drug} — Stability: Baseline vs Drug',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

#%% Percentage of time cells per recording day

# Calculate % time cells per recording (rec_date) for SS1 and SS2
time_cell_pct_ss1 = df_ss1_shared.groupby('rec_date').apply(
    lambda g: 100 * g['is_time_cell'].sum() / len(g), include_groups=False)
time_cell_pct_ss2 = df_ss2_shared.groupby('rec_date').apply(
    lambda g: 100 * g['is_time_cell'].sum() / len(g), include_groups=False)

# Get common rec_dates
common_rec_dates = time_cell_pct_ss1.index.intersection(time_cell_pct_ss2.index)
time_cell_pct_ss1 = time_cell_pct_ss1.loc[common_rec_dates]
time_cell_pct_ss2 = time_cell_pct_ss2.loc[common_rec_dates]

print(f"\n=== Time Cell Percentage Analysis ===")
print(f"SS1 (baseline): mean={time_cell_pct_ss1.mean():.2f}%, median={time_cell_pct_ss1.median():.2f}%")
print(f"SS2 (drug):     mean={time_cell_pct_ss2.mean():.2f}%, median={time_cell_pct_ss2.median():.2f}%")

# Paired stats
# stat, p_val = stats.wilcoxon(time_cell_pct_ss1.values, time_cell_pct_ss2.values)
# print(f"Wilcoxon signed-rank test: p={p_val:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(1, 2), dpi=300)
pf.plot_bar_with_paired_scatter(ax, time_cell_pct_ss1.values, time_cell_pct_ss2.values)
# ax.set_ylabel('% Time Cells')
# ax.set_xticklabels(['SS1 (baseline)', f'SS2 ({drug})'])
# ax.set_title(f'{drug} - Time Cell Percentage\nWilcoxon p={p_val:.4f}')
# plt.tight_layout()
# plt.show()
