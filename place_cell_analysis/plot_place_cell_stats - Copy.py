# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:48:11 2026

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from place_cell_functions import select_significant_cells
import common.plotting_functions_Jingyu as pf 
from config import CONFIG_TIME, CONFIG_PLACE
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
    
# Use GPU-accelerated batch calculation for specified correlation types
# Options: 'odd_even', 'mean_pairwise', 'consecutive' (or None for all)
corr_methods = ['odd_even', 'consecutive']

# PATHS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\GCaMP_drug_infusion")
# OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
OUTPUT_RES = OUT_DIR_RAW_DATA /'place_cell_dataframe_3rsd'
#%% Main
df_place_field_all_ss1 = pd.DataFrame()
df_place_field_all_ss2 = pd.DataFrame()

# drug = 'ctrl'
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
        parquet_path = OUTPUT_RES / f'{rec_id}_place_cell_dataframe.parquet'
        df_place_field_ss1 = pd.read_parquet(parquet_path)
        df_place_field_ss1['rec_date'] = f'{anm}-{date}'
        df_place_field_ss1['SCH_days'] = rec['SCH_days']
        df_place_field_ss1['propranolol_days'] = rec['propranolol_days']
        df_place_field_ss1['prazosin_days'] = rec['prazosin_days']
        df_place_field_ss1['ctrl_days'] = rec['ctrl_days']
        df_place_field_all_ss1 = pd.concat((df_place_field_all_ss1, df_place_field_ss1))
    except Exception as e:
        print(f"  Error loading {rec_id}: {e}")

    ss = '04'
    rec_id = f'{anm}-{date}-{ss}'
    print(f'loading {rec_id}----------------------')
    # p_beh = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec_id}.pkl'
    # beh = pd.read_pickle(p_beh)
    try:
        parquet_path = OUTPUT_RES / f'{rec_id}_place_cell_dataframe.parquet'
        df_place_field_ss2 = pd.read_parquet(parquet_path)
        df_place_field_ss2['rec_date'] = f'{anm}-{date}'
        df_place_field_ss2['SCH_days'] = rec['SCH_days']
        df_place_field_ss2['propranolol_days'] = rec['propranolol_days']
        df_place_field_ss2['prazosin_days'] = rec['prazosin_days']
        df_place_field_ss2['ctrl_days'] = rec['ctrl_days']
        df_place_field_all_ss2 = pd.concat((df_place_field_all_ss2, df_place_field_ss2))
    except Exception as e:
        print(f"  Error loading {rec_id}: {e}")

#%% select place cells
info_threshold = 0.2
shuff_thresh = 99
mean_dff_threshold = 0
perc_active_laps_threshold = 0.2
perc_active_frames_threshold = 0

df_place_cells_ss1  = select_significant_cells(df_place_field_all_ss1, 
                                               CONFIG_PLACE,
                                               min_in_out_ratio=3,
                                               min_width_cm=15.0,
                                               min_transient_fraction=0.1
                                               # info_threshold,
                                               # shuff_thresh,
                                               # mean_dff_threshold,
                                               # perc_active_laps_threshold,
                                               # perc_active_frames_threshold
                                               )
df_place_cells_ss2  = select_significant_cells(df_place_field_all_ss2, 
                                               CONFIG_PLACE,
                                               min_in_out_ratio=3,
                                               min_width_cm=15.0,
                                               min_transient_fraction=0.1
                                               # info_threshold,
                                               # shuff_thresh,
                                               # mean_dff_threshold,
                                               # perc_active_laps_threshold,
                                               # perc_active_frames_threshold
                                               )

# Select only the first 3 recording dates per animal for ctrl
df_place_cells_ss1_first3 = df_place_cells_ss1.loc[df_place_cells_ss1[f'{drug}_days']<4]
df_place_cells_ss1 = df_place_cells_ss1_first3
# df_place_cells_ss1.to_parquet(OUTPUT_RES/ rf"df_place_cells_ss1_first3_{drug}.parquet")

df_place_cells_ss2_first3 = df_place_cells_ss2.loc[df_place_cells_ss2[f'{drug}_days']<4]
df_place_cells_ss2 = df_place_cells_ss2_first3
# df_place_cells_ss1.to_parquet(OUTPUT_RES/ rf"df_place_cells_ss1_first3_{drug}.parquet")
#%% plotting
perc_place_cell_ss1 = df_place_cells_ss1_first3.groupby('rec_date')['is_significant'].mean()
perc_place_cell_ss2 = df_place_cells_ss2_first3.groupby('rec_date')['is_significant'].mean()
common = perc_place_cell_ss1.index.intersection(perc_place_cell_ss2.index)
perc_place_cell_ss1 = perc_place_cell_ss1.loc[common]
perc_place_cell_ss2 = perc_place_cell_ss2.loc[common]
fig, ax = plt.subplots(figsize=(1,2), dpi=300)
pf.plot_bar_with_paired_scatter(ax, perc_place_cell_ss1, perc_place_cell_ss2)

stab_place_cell_ss1 = (
    df_place_cells_ss1_first3[df_place_cells_ss1_first3['is_significant']]
    .groupby('rec_date')['odd_even_corr']
    .mean()
)
stab_place_cell_ss2 = (
    df_place_cells_ss2_first3[df_place_cells_ss2_first3['is_significant']]
    .groupby('rec_date')['odd_even_corr']
    .mean()
)
common = stab_place_cell_ss1.index.intersection(stab_place_cell_ss2.index)
stab_place_cell_ss1 = stab_place_cell_ss1.loc[common]
stab_place_cell_ss2 = stab_place_cell_ss2.loc[common]
fig, ax = plt.subplots(figsize=(1,2), dpi=300)
pf.plot_bar_with_paired_scatter(ax, stab_place_cell_ss1, stab_place_cell_ss2)

#%% Place cell sequence: ss1 vs ss2
# Match cells present in both sessions within each recording, sort by ss1 peak,
# then show the same cells (same order) in ss2 to compare sequence preservation.
peak_col = CONFIG_PLACE['peak_position_col']
field_col = CONFIG_PLACE['field_map_col']

ss1_sig = df_place_cells_ss1[df_place_cells_ss1['is_significant']]
shared = ss1_sig.merge(
    df_place_field_all_ss2[['rec_date', 'cell_id', peak_col, field_col]],
    on=['rec_date', 'cell_id'], how='inner', suffixes=('_ss1', '_ss2'),
)
shared = shared.sort_values(f'{peak_col}_ss1').reset_index(drop=True)

maps_ss1 = np.vstack(shared[f'{field_col}_ss1'].values)
maps_ss2 = np.vstack(shared[f'{field_col}_ss2'].values)

fig, axes = plt.subplots(1, 2, figsize=(4, 4), dpi=300, sharey=True)
for ax, maps, title in zip(axes, [maps_ss1, maps_ss2], ['ss1', 'ss2 (ss1 order)']):
    ax.imshow(maps, aspect='auto', cmap='viridis',
              extent=[0, track_length, len(shared), 0], interpolation='nearest')
    ax.set_xlabel('Position (cm)')
    ax.set_title(title)
axes[0].set_ylabel(f'Place cell # sorted by ss1 peak (n={len(shared)})')
plt.suptitle(f'{drug}: place cell sequence ss1 vs ss2')
plt.tight_layout()
plt.show()

# Reverse: sort by ss2 peak, show both
ss2_sig = df_place_cells_ss2[df_place_cells_ss2['is_significant']]
shared_rev = ss2_sig.merge(
    df_place_field_all_ss1[['rec_date', 'cell_id', peak_col, field_col]],
    on=['rec_date', 'cell_id'], how='inner', suffixes=('_ss2', '_ss1'),
)
shared_rev = shared_rev.sort_values(f'{peak_col}_ss2').reset_index(drop=True)

maps_ss1_rev = np.vstack(shared_rev[f'{field_col}_ss1'].values)
maps_ss2_rev = np.vstack(shared_rev[f'{field_col}_ss2'].values)

fig, axes = plt.subplots(1, 2, figsize=(4, 4), dpi=300, sharey=True)
for ax, maps, title in zip(axes, [maps_ss1_rev, maps_ss2_rev], ['ss1 (ss2 order)', 'ss2']):
    ax.imshow(maps, aspect='auto', cmap='viridis',
              extent=[0, track_length, len(shared_rev), 0], interpolation='nearest')
    ax.set_xlabel('Position (cm)')
    ax.set_title(title)
axes[0].set_ylabel(f'Place cell # sorted by ss2 peak (n={len(shared_rev)})')
plt.suptitle(f'{drug}: place cell sequence ss2 vs ss1')
plt.tight_layout()
plt.show()

#%% Independent sequences: each session's place cells sorted by its own peak
ss1_own = ss1_sig.sort_values(peak_col).reset_index(drop=True)
ss2_own = ss2_sig.sort_values(peak_col).reset_index(drop=True)

maps_ss1_own = np.vstack(ss1_own[field_col].values)
maps_ss2_own = np.vstack(ss2_own[field_col].values)

fig, axes = plt.subplots(1, 2, figsize=(4, 3), dpi=300)
for ax, maps, title, n in zip(axes,
                              [maps_ss1_own, maps_ss2_own],
                              ['ss1', 'ss2'],
                              [len(ss1_own), len(ss2_own)]):
    ax.imshow(maps, aspect='auto', cmap='viridis',
              extent=[0, track_length, n, 0], interpolation='nearest')
    ax.set_xlabel('Position (cm)')
    ax.set_title(f'{title} (n={n})')
axes[0].set_ylabel('Place cell # sorted by own peak')
plt.suptitle(f'{drug}: place cell sequence (independent sort)')
plt.tight_layout()
plt.show()