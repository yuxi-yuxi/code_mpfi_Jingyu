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

#%% ================================================================
#   Quantitative ss1 vs ss2 comparison of place-cell properties
#   Significant place cells only; aggregated per recording (paired
#   across rec_date) and compared with a paired Wilcoxon test.
# =================================================================
from scipy.stats import wilcoxon

# ---- adjustable salient-zone definitions (cm) ----
RUN_ONSET_ZONE = (0.0, 30.0)            # track-start / run-onset zone
REWARD_ZONE    = (150.0, 220.0)         # reward zone (clamped by track_length)

field_col_raw = 'place_field_map'       # raw (un-normalised) rate map -> mean dF/F
peak_bin_col  = 'place_field_peak_bin'


def _primary_field_index(row):
    """Index of the tentative field containing the global peak bin;
    falls back to the field with the largest in/out ratio."""
    masks = row['tentative_field']
    if masks is None or len(masks) == 0:
        return None
    pb = row[peak_bin_col]
    if pb is not None and not (isinstance(pb, float) and np.isnan(pb)):
        for j, m in enumerate(masks):
            m = np.asarray(m, dtype=bool)
            if 0 <= int(pb) < m.size and m[int(pb)]:
                return j
    ratios = np.asarray(row['tentative_field_in_out_ratio'], dtype=float)
    if ratios.size == 0 or np.all(np.isnan(ratios)):
        return 0
    return int(np.nanargmax(ratios))


def _field_scalar(row, col):
    """Value of a per-field list column for the cell's primary field."""
    j = _primary_field_index(row)
    if j is None:
        return np.nan
    vals = np.asarray(row[col], dtype=float)
    return vals[j] if j < vals.size else np.nan


def build_cell_metrics(df):
    """Per-cell scalar metrics for significant place cells only."""
    d = df[df['is_significant']].copy()
    d['peak_rate']    = d['place_field_peak_amplitude'].astype(float)
    d['mean_rate']    = d[field_col_raw].apply(
        lambda m: np.nanmean(np.asarray(m, dtype=float)))
    d['tt_corr']      = d['odd_even_corr'].astype(float)
    d['cc_corr']      = d['consecutive_corr'].astype(float)
    d['in_out_ratio'] = d.apply(
        lambda r: _field_scalar(r, 'tentative_field_in_out_ratio'), axis=1)
    d['si_bits']      = d['spatial_information_bits'].astype(float)
    d['field_width']  = d.apply(
        lambda r: _field_scalar(r, 'tentative_field_width_cm'), axis=1)
    peak_cm = d['place_field_position_cm'].astype(float)
    d['in_run_onset'] = peak_cm.between(*RUN_ONSET_ZONE).astype(float)
    d['in_reward']    = peak_cm.between(*REWARD_ZONE).astype(float)
    return d


cells_ss1 = build_cell_metrics(df_place_cells_ss1)
cells_ss2 = build_cell_metrics(df_place_cells_ss2)


def paired_by_rec(d1, d2, col, agg='mean'):
    """Per-recording aggregate of `col`, aligned on rec_dates present in both."""
    s1 = d1.groupby('rec_date')[col].agg(agg)
    s2 = d2.groupby('rec_date')[col].agg(agg)
    common = s1.index.intersection(s2.index)
    return s1.loc[common], s2.loc[common]


# (column, panel label, per-recording aggregator)
metrics = [
    ('peak_rate',    'Peak dF/F',                   'mean'),
    ('mean_rate',    'Mean dF/F',                   'mean'),
    ('tt_corr',      'Trial-trial corr\n(odd/even)', 'mean'),
    ('cc_corr',      'Trial-trial corr\n(consecutive)', 'mean'),
    ('in_out_ratio', 'In/out-field ratio',          'mean'),
    ('si_bits',      'Spatial info (bits)',          'mean'),
    ('field_width',  'Field width (cm)',            'mean'),
    ('in_run_onset', 'Frac. @ run-onset\n(%s-%s cm)' % RUN_ONSET_ZONE, 'mean'),
    ('in_reward',    'Frac. @ reward\n(%s-%s cm)' % REWARD_ZONE,        'mean'),
]

# ---- paired, per-recording comparison (Wilcoxon signed-rank) ----
ncol = 4
nrow = int(np.ceil(len(metrics) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.25 * ncol, 2.75 * nrow), dpi=300)
axes = np.atleast_1d(axes).ravel()
summary = []
for ax, (col, label, agg) in zip(axes, metrics):
    v1, v2 = paired_by_rec(cells_ss1, cells_ss2, col, agg)
    pf.plot_bar_with_paired_scatter(
        ax, v1, v2, title=label, ylabel='', xticklabels=('ss1', 'ss2'))
    try:
        stat, p = wilcoxon(v1.values, v2.values)
    except ValueError:
        stat, p = np.nan, np.nan
    summary.append({
        'metric': label.replace('\n', ' '),
        'n_rec': len(v1),
        'ss1_mean': v1.mean(),
        'ss2_mean': v2.mean(),
        'wilcoxon_p': p,
    })
for ax in axes[len(metrics):]:      # blank any unused panels
    ax.axis('off')
plt.suptitle(f'{drug}: ss1 vs ss2 place-cell properties (paired by recording)')
plt.tight_layout()
plt.show()

summary_df = pd.DataFrame(summary)
print(f'\n=== {drug}: ss1 vs ss2 (paired by recording, Wilcoxon) '
      f'(n cells ss1={len(cells_ss1)}, ss2={len(cells_ss2)}) ===')
print(summary_df.to_string(index=False,
      float_format=lambda x: f'{x:.4g}'))

#%% ----- pooled per-cell comparison (all cells, unpaired Mann-Whitney) -----
# Same metrics, but every significant cell is one observation (ignores the
# within-recording pairing). More power; complements the paired view above.
from scipy.stats import mannwhitneyu

fig, axes = plt.subplots(nrow, ncol, figsize=(2.25 * ncol, 2.75 * nrow), dpi=300)
axes = np.atleast_1d(axes).ravel()
summary_cell = []
for ax, (col, label, agg) in zip(axes, metrics):
    x1 = cells_ss1[col].to_numpy(dtype=float)
    x2 = cells_ss2[col].to_numpy(dtype=float)
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]

    means = [np.mean(x1), np.mean(x2)]
    sems  = [np.std(x1, ddof=1) / np.sqrt(len(x1)),
             np.std(x2, ddof=1) / np.sqrt(len(x2))]
    bars = ax.bar([0, 1], means, yerr=sems, width=0.6, capsize=2,
                  color=['grey', 'firebrick'], alpha=.6,
                  error_kw={'elinewidth': 0.6, 'capthick': 0.6, 'ecolor': 'k'})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['ss1', 'ss2'], fontsize=6)
    ax.set_title(label, fontsize=6)

    try:
        stat, p = mannwhitneyu(x1, x2, alternative='two-sided')
    except ValueError:
        stat, p = np.nan, np.nan
    summary_cell.append({
        'metric': label.replace('\n', ' '),
        'n_ss1': len(x1), 'n_ss2': len(x2),
        'ss1_mean': means[0], 'ss2_mean': means[1],
        'mannwhitney_p': p,
    })
for ax in axes[len(metrics):]:
    ax.axis('off')
plt.suptitle(f'{drug}: ss1 vs ss2 place-cell properties (pooled per cell)')
plt.tight_layout()
plt.show()

summary_cell_df = pd.DataFrame(summary_cell)
print(f'\n=== {drug}: ss1 vs ss2 (pooled per cell, Mann-Whitney U) ===')
print(summary_cell_df.to_string(index=False,
      float_format=lambda x: f'{x:.4g}'))