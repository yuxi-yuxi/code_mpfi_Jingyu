# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 14:28:07 2026

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from place_cell_analysis import place_cell_functions as pcf
from common.utils_basic import nearest_mapping, normalize
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
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\GCaMP_drug_infusion")
# OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
OUTPUT_RES = OUT_DIR_RAW_DATA /'place_cell_dataframe'
#%% Main
df_place_field_all_ss1 = pd.DataFrame()
df_place_field_all_ss2 = pd.DataFrame()

for _, rec in rec_drug.iterrows():
    anm = rec['anm']
    date = rec['date']
    print(f'\n{anm}-{date}')
    
    data_path = OUT_DIR_RAW_DATA/'raw_signals'/f'{anm}-{date}'
    
    ss = '02'
    rec_id = f'{anm}-{date}-{ss}'
    print(f'loading {rec_id}----------------------')
    # p_beh = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec_id}.pkl'
    # beh = pd.read_pickle(p_beh)
    try:
        df_place_field_ss1 = pd.read_parquet(OUTPUT_RES / f'{rec_id}_place_cell_dataframe.parquet')
        df_place_field_ss1['rec_date'] = f'{anm}-{date}'
        df_place_field_all_ss1 = pd.concat((df_place_field_all_ss1, df_place_field_ss1))
    except:
        print (rec_id)
    
    ss = '04'
    rec_id = f'{anm}-{date}-{ss}'
    print(f'loading {rec_id}----------------------')
    # p_beh = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec_id}.pkl'
    # beh = pd.read_pickle(p_beh)
    try:
        df_place_field_ss2 = pd.read_parquet(OUTPUT_RES / f'{rec_id}_place_cell_dataframe.parquet')
        df_place_field_ss2['rec_date'] = f'{anm}-{date}'
        df_place_field_all_ss2 = pd.concat((df_place_field_all_ss2, df_place_field_ss2))
    except:
        print (rec_id)
        
#%% Calculate place cell stability
from scipy import stats

# Use GPU-accelerated batch calculation for specified correlation types
# Options: 'odd_even', 'mean_pairwise', 'consecutive' (or None for all)
corr_methods = ['odd_even', 'consecutive']

print("Calculating trial correlations for SS1 (GPU)...")
per_lap_profiles_ss1 = df_place_field_all_ss1['per_lap_profile'].tolist()
corr_results_ss1 = pcf.calculate_all_trial_correlations_gpu(per_lap_profiles_ss1, methods=corr_methods, gpu=True)
df_place_field_all_ss1['stability'] = corr_results_ss1['odd_even']
df_place_field_all_ss1['consecutive_corr'] = corr_results_ss1['consecutive']

print("Calculating trial correlations for SS2 (GPU)...")
per_lap_profiles_ss2 = df_place_field_all_ss2['per_lap_profile'].tolist()
corr_results_ss2 = pcf.calculate_all_trial_correlations_gpu(per_lap_profiles_ss2, methods=corr_methods, gpu=True)
df_place_field_all_ss2['stability'] = corr_results_ss2['odd_even']
df_place_field_all_ss2['consecutive_corr'] = corr_results_ss2['consecutive']

#%% visualization
SI_threshold = 0.15
shuff_SI_thresh = 99

key = ['rec_date', 'cell_id']
shared_keys = (
    df_place_field_all_ss1[key]
    .merge(df_place_field_all_ss2[key], on=key, how='inner')
    .drop_duplicates()
)

df_ss1_shared = df_place_field_all_ss1.merge(shared_keys, on=key, how='inner')
df_ss2_shared = df_place_field_all_ss2.merge(shared_keys, on=key, how='inner')

def select_place_cell(df, SI_threshold, shuff_SI_thresh):
    df['shuffle_SI_thresh'] = df['shuffled_SI'].apply(
        lambda x: np.nanpercentile(x, shuff_SI_thresh) if x is not None else np.nan)
    df['is_place_cell'] = ((df['spatial_information_bits'] > SI_threshold) &
                           (df['spatial_information_bits'] > df['shuffle_SI_thresh']))
    return df

df_ss1_shared = select_place_cell(df_ss1_shared, SI_threshold, shuff_SI_thresh)
df_ss2_shared = select_place_cell(df_ss2_shared, SI_threshold, shuff_SI_thresh)

df_place_cell_ss1 = df_ss1_shared.loc[df_ss1_shared['is_place_cell']]
df_place_cell_ss2 = df_ss2_shared.loc[df_ss2_shared['is_place_cell']]


df = df_place_cell_ss1.iloc[0]
a = df['place_field_map']

cell_order_ss1 = (
    df_place_cell_ss1.sort_values('place_field_position_cm')[key]
    .drop_duplicates()
    .apply(tuple, axis=1)
    .to_numpy()
)

cell_order_ss2= (
    df_place_cell_ss2.sort_values('place_field_position_cm')[key]
    .drop_duplicates()
    .apply(tuple, axis=1)
    .to_numpy()
)

df_ss1_ordered = (
    df_place_cell_ss1.set_index(key)
    .reindex(cell_order_ss1)
    .reset_index()
)

df_ss2_ordered = (
    df_place_cell_ss2.set_index(key)
    .reindex(cell_order_ss2)
    .reset_index()
)
for imshow_array in [np.stack(df_ss1_ordered['place_field_map_norm']),
                 np.stack(df_ss2_ordered['place_field_map_norm'])
                 ]:

    fig, ax = plt.subplots(figsize=(3, 3))
    # Place cell sequence heatmap
    track_length = 180
    n_cells_shown =  imshow_array.shape[0]
    im1 = ax.imshow(imshow_array, aspect='auto', cmap='Greys',
                     extent=[0, track_length, n_cells_shown, 0],
                     interpolation='nearest')
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Cell # (sorted by place field)')
    ax.set_title('Place Cell Sequence')
    plt.colorbar(im1, ax=ax, label='Normalized activity')

#%% Place cell stability comparison: ss1 (baseline) vs ss2 (drug)
# Get stability values for place cells in each session
stability_ss1 = df_place_cell_ss1['stability'].dropna().values
stability_ss2 = df_place_cell_ss2['stability'].dropna().values

print(f"\n=== Place Cell Stability Analysis ===")
print(f"SS1 (baseline): n={len(stability_ss1)}, mean={np.mean(stability_ss1):.3f}, median={np.median(stability_ss1):.3f}")
print(f"SS2 (drug):     n={len(stability_ss2)}, mean={np.mean(stability_ss2):.3f}, median={np.median(stability_ss2):.3f}")

# Statistical test: Mann-Whitney U (unpaired comparison)
stat_mw, p_mw = stats.mannwhitneyu(stability_ss1, stability_ss2, alternative='two-sided')
print(f"\nMann-Whitney U test: U={stat_mw:.1f}, p={p_mw:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 1. Box plot comparison
ax1 = axes[0]
box_data = [stability_ss1, stability_ss2]
bp = ax1.boxplot(box_data, labels=['SS1 (baseline)', 'SS2 (drug)'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax1.set_ylabel('Stability (mean lap correlation)')
ax1.set_title(f'Place Cell Stability\nMann-Whitney p={p_mw:.4f}')

# Add individual points
for i, data in enumerate(box_data, 1):
    x = np.random.normal(i, 0.04, size=len(data))
    ax1.scatter(x, data, alpha=0.3, s=10, color='black')

# 2. Histogram comparison
ax2 = axes[1]
bins = np.linspace(-0.5, 1, 20)
ax2.hist(stability_ss1, bins=bins, alpha=0.5, label=f'SS1 (n={len(stability_ss1)})', color='blue')
ax2.hist(stability_ss2, bins=bins, alpha=0.5, label=f'SS2 (n={len(stability_ss2)})', color='red')
ax2.axvline(np.median(stability_ss1), color='blue', linestyle='--', label=f'SS1 median: {np.median(stability_ss1):.2f}')
ax2.axvline(np.median(stability_ss2), color='red', linestyle='--', label=f'SS2 median: {np.median(stability_ss2):.2f}')
ax2.set_xlabel('Stability (mean lap correlation)')
ax2.set_ylabel('Count')
ax2.set_title('Stability Distribution')
ax2.legend()

plt.suptitle(f'{drug} - Place Cell Stability: Baseline vs Drug', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#%%


















# --- Heatmap ordering option ---
# None  : each session sorted by its own place field positions
# 'ss1' : both heatmaps use ss1 place cell order
# 'ss2' : both heatmaps use ss2 place cell order
shared_sort_by = 'ss1'


# --- Pre-process both DataFrames ---
processed = {}
for label, df_raw in [('ss1', df_place_field_all_ss1), ('ss2', df_place_field_all_ss2)]:
    df = df_raw.reset_index(drop=True)
    df['shuffle_SI_thresh'] = df['shuffled_SI'].apply(
        lambda x: np.nanpercentile(x, shuff_SI_thresh) if x is not None else np.nan
    )
    df['is_place_cell'] = ((df['spatial_information_bits'] > SI_threshold) &
                           (df['spatial_information_bits'] > df['shuffle_SI_thresh']))
    place_cell_indices = np.where(df['is_place_cell'])[0]
    place_field_pos = df['place_field_position_cm'].values
    sort_order = np.argsort(place_field_pos[place_cell_indices])
    sorted_indices = place_cell_indices[sort_order]
    processed[label] = dict(
        df=df,
        place_cell_indices=place_cell_indices,
        sorted_indices=sorted_indices,
        place_field_pos=place_field_pos,
        spatial_information=df['spatial_information_bits'].values,
    )
    print(f"  {label} place cells (SI > {SI_threshold}): {len(place_cell_indices)}")

# Build cell_id-keyed lookup for cross-session matching
# (DataFrames differ in size so row indices don't correspond to the same cell)
for label in ('ss1', 'ss2'):
    processed[label]['df_by_id'] = processed[label]['df'].set_index('cell_id')

# Identify shared sort reference: ordered cell_ids from reference session's place cells
if shared_sort_by in ('ss1', 'ss2'):
    ref_p = processed[shared_sort_by]
    ref_sorted_cell_ids = ref_p['df'].loc[ref_p['sorted_indices'], 'cell_id'].values
    print(f"\nHeatmap order: using {shared_sort_by} sort for both sessions "
          f"({len(ref_sorted_cell_ids)} reference place cells)")

# --- Plot each session ---
for label, p in processed.items():
    df = p['df']
    place_cell_indices = p['place_cell_indices']
    place_field_pos = p['place_field_pos']
    spatial_information_all = p['spatial_information']

    # Build heatmap matrix using session-specific place_field_map (raw firing rate)
    if shared_sort_by in ('ss1', 'ss2'):
        # Match reference place cells by cell_id; skip cells absent in this session
        df_by_id = p['df_by_id']
        valid_ids = [cid for cid in ref_sorted_cell_ids if cid in df_by_id.index]
        place_fields_sorted = np.vstack(df_by_id.loc[valid_ids, 'place_field_map'].values)
        n_cells_shown = len(valid_ids)
        sort_label = f'sorted by {shared_sort_by} order ({n_cells_shown} cells)'
    else:
        place_fields_sorted = np.vstack(df.loc[p['sorted_indices'], 'place_field_map'].values)
        n_cells_shown = len(p['sorted_indices'])
        sort_label = 'sorted by own place field'

    # Normalize each row to [0, 1] for display (place_field_map is raw firing rate)
    row_max = np.nanmax(place_fields_sorted, axis=1, keepdims=True)
    imshow_trace = place_fields_sorted / np.where(row_max > 0, row_max, 1)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Place cell sequence heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(imshow_trace, aspect='auto', cmap='Greys',
                     extent=[0, track_length, n_cells_shown, 0],
                     interpolation='nearest')
    ax1.set_xlabel('Position (cm)')
    ax1.set_ylabel('Cell # (sorted by place field)')
    ax1.set_title(f'Place Cell Sequence\n({sort_label})')
    plt.colorbar(im1, ax=ax1, label='Normalized activity')

    # 2. Distribution of place field positions (this session's own place cells)
    ax2 = axes[0, 1]
    ax2.hist(place_field_pos[place_cell_indices], bins=n_bins // 2,
             range=(0, track_length), edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Place field position (cm)')
    ax2.set_ylabel('Number of place cells')
    ax2.set_title(f'Place Field Distribution (SI > {SI_threshold})')
    ax2.set_xlim([0, track_length])

    # 3. Spatial information distribution
    ax3 = axes[1, 0]
    si_valid = spatial_information_all[~np.isnan(spatial_information_all)]
    ax3.hist(si_valid, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(SI_threshold, color='g', linestyle='-', linewidth=2, label=f'SI = {SI_threshold} threshold')
    ax3.axvline(np.median(si_valid), color='r', linestyle='--',
                label=f'Median: {np.median(si_valid):.2f}')
    ax3.set_xlabel('Spatial information (bits/event)')
    ax3.set_ylabel('Number of cells')
    ax3.set_title('Spatial Information Distribution')
    ax3.legend()

    # 4. SI vs shuffle threshold scatter
    ax4 = axes[1, 1]
    shuffle_thresh_all = df['shuffle_SI_thresh'].values
    valid_mask = ~np.isnan(shuffle_thresh_all)
    ax4.scatter(spatial_information_all[valid_mask], shuffle_thresh_all[valid_mask],
                alpha=0.3, s=10)
    ax4.plot([0, np.nanmax(spatial_information_all)], [0, np.nanmax(spatial_information_all)],
             'r--', label='SI = shuffle threshold')
    ax4.set_xlabel('Spatial information (bits)')
    ax4.set_ylabel('99th percentile shuffle SI')
    ax4.set_title('SI vs Shuffle Threshold')
    ax4.legend()

    plt.suptitle(f'{label} — All recordings (n={len(df)} cells)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
