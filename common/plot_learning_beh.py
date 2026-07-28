# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:55:59 2026

@author: Jingyu Cao
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter1d
from common.utils_basic import zero_padding
from common.trial_selection import seperate_valid_trial

def extract_behaviour_info(rec, p_beh):
    """
    Build a one-row behavioural summary DataFrame for a single session.

    Parameters
    ----------
    rec : str
        Recording ID of the form 'ANM-YYYYMMDD-SS' (e.g. 'AC330-20260602-02').
    p_beh : str or Path
        Path to the behaviour pickle file for this session.

    Returns
    -------
    df_beh_pool : pd.DataFrame
        Single-row DataFrame with summary speed / lick / first-lick stats.
        Empty if the recording id is malformed, the file is missing, or
        no valid trials are found.
    """
    df_beh_pool = pd.DataFrame({
         'anm_id': str,
         'rec_id': str,
         # 'date': str,
         # 'session': int,
         'block_num': int,
         'block_stat': [],
         'passive_trials': [],
         'random_trials': [],
         

         'rew_rate': float,

         # running
         'speed_time_profile': [],
         'speed_time_profile_array': [],
         'speed_time_profile_mean': [],

         'speed_distance_profile': [],
         'speed_distance_profile_mean': [],
         'speed_distance_mean': [],
         'speed_distance_max': [],
         'speed_distance_median': [],

         # licks
         'lick_distance_profile': [],          # licks/cm
         'lick_distance_profile_mean': [],
         'lick_dist_norm.': [],                # normalised by session peak (over distance)
         'lick_dist_fraction': [],             # per-trial fraction (n_licks in bin / total licks in trial)
         'lick_distance_selectivity': [],      # per-trial post/(pre+post) selectivity

         'lick_time_profile': [],
         'lick_frequency_profile': [],         # licks/s
         'lick_frequency_mean': [],
         'lick_time_norm.': [],                # normalised by session peak (over time)
         'lick_time_fraction': [],             # per-trial fraction over time
         'lick_time_selectivity': [],          # per-trial post/(pre+post) selectivity

         'first_licks_dist': [],
         'first_licks_dist_median': float,
         'first_licks_dist_iqr': float,
         'first_licks_dist_var': float,

         'first_licks_time': [],
         'first_licks_time_median': float,
         'first_licks_time_iqr': float,
         'first_licks_time_var': float,
        })

    print('processing {}-------------------------'.format(rec))

    # parse recording id
    try:
        anm, date, ss = rec.split('-')
    except ValueError:
        print(f'  bad rec id: {rec}')
        return df_beh_pool

    # load behaviour
    p_beh = Path(p_beh)
    if not p_beh.exists():
        print(f'  no behaviour file: {p_beh}')
        return df_beh_pool
    beh = pd.read_pickle(p_beh)

    valid_trials = seperate_valid_trial(beh)[1:]
    n_trials = len(beh['new_trial_statements'])
    # if not np.any(valid_trials):
    #     print(f'  no valid trials for {rec}, skipping')
    #     return df_beh_pool
    block_stat = beh['block_statements']
    block_num = beh['block_numbers'][1:]
    # for trials with block_nums==n, block_info for that trial is block_stat[n-1] (block_stat[n-1][2] == n),
    # whether trial passive = block_info[4], whether trial_length random = block_info[5]
    passive_trials = []
    random_trials = []
    for n in block_num:
        block_info = block_stat[int(n) - 1]
        if int(block_info[2]) != int(n):
            print(f'  block index mismatch: block_num={n} vs block_stat[{n-1}][2]={block_info[2]}')
        passive_trials.append(int(block_info[4]))
        random_trials.append(int(block_info[5]))
    passive_trials = np.asarray(passive_trials, dtype=bool)
    random_trials = np.asarray(random_trials, dtype=bool)
    
    # ---- first lick distances (skip first trial as before) ----
    licks_dist = beh['lick_distances_aligned'][1:]
    first_licks_dist = [t[t > 30][0]
                        if type(t) != float and len(t) > 0 and len(t[t > 30]) > 0
                        else np.nan
                        for t in licks_dist]
    first_licks_dist_median = np.nanmedian(first_licks_dist)
    first_licks_dist_iqr = iqr(first_licks_dist, nan_policy='omit')
    first_licks_dist_var = np.nanstd(first_licks_dist)

    # ---- lick times + per-trial lick frequency profile ----
    licks_time = beh['lick_times_aligned']
    licks_time_filtered = []
    lick_freqs_per_trial = np.zeros((n_trials, 4000))
    for t, licks in enumerate(licks_time):
        if type(licks) != float and len(licks) > 0:
            licks_arr = np.array(licks)
            lick_freq = np.histogram(licks_arr, bins=4000, range=(0, 4000))[0] * 1000  # Hz
            # lick_freq = gaussian_filter1d(lick_freq, sigma=10)
            licks_time_filtered.append(licks_arr)
            lick_freqs_per_trial[t] = lick_freq
        else:
            licks_time_filtered.append(np.nan)

    lick_freqs = lick_freqs_per_trial[valid_trials]
    lick_freqs_mean = np.nanmean(lick_freqs, axis=0)

    # ---- first lick times ----
    first_licks_time = [t[t > 500][0]
                        if type(t) != float and len(t[t > 500]) > 0
                        else np.nan
                        for t in licks_time_filtered]
    first_licks_time_median = np.nanmedian(first_licks_time)
    first_licks_time_iqr = iqr(first_licks_time, nan_policy='omit')
    first_licks_time_var = np.nanstd(first_licks_time)

    # ---- lick distance profile ----
    licks_dist_map = beh['lick_maps']
    if len(licks_dist_map)!=n_trials:
        print('Warning, check trial numbers')
        return df_beh_pool
        
    # licks_dist_map = np.vstack([licks_dist_map[t]
    #                             for t in range(n_trials) 
    #                             if valid_trials[t]
    #                             ])
    licks_dist_map = np.vstack([licks for licks in licks_dist_map
                                if len(licks)>0
                                ])
    licks_dist_profile = licks_dist_map.reshape(licks_dist_map.shape[0], 220, 10).mean(axis=2)
    licks_dist_profile_mean = np.nanmean(licks_dist_profile, axis=0)

    # ---- normalised / fractional lick profiles (distance) ----
    # session peak = max across all trials and bins
    dist_peak = np.nanmax(licks_dist_profile)
    if np.isfinite(dist_peak) and dist_peak > 0:
        lick_dist_norm = licks_dist_profile / dist_peak
    else:
        lick_dist_norm = np.full_like(licks_dist_profile, np.nan)

    with np.errstate(invalid='ignore', divide='ignore'):
        row_sum_dist = np.nansum(licks_dist_profile, axis=1, keepdims=True)
        lick_dist_fraction = np.where(row_sum_dist > 0,
                                      licks_dist_profile / row_sum_dist,
                                      np.nan)

    # post-vs-pre selectivity (uses pre-computed per-trial values)
    lick_distance_selectivity = np.asarray(beh['lick_selectivities'])

    # ---- normalised / fractional lick profiles (time) ----
    # session peak = max across all trials and bins
    time_peak = np.nanmax(lick_freqs)
    if np.isfinite(time_peak) and time_peak > 0:
        lick_time_norm = lick_freqs / time_peak
    else:
        lick_time_norm = np.full_like(lick_freqs, np.nan)

    with np.errstate(invalid='ignore', divide='ignore'):
        row_sum_time = np.nansum(lick_freqs, axis=1, keepdims=True)
        lick_time_fraction = np.where(row_sum_time > 0,
                                      lick_freqs / row_sum_time,
                                      np.nan)

    # per-trial time-aligned selectivity: post/(pre+post), split at midpoint
    mid_t = lick_freqs.shape[1] // 2
    sums_pre = np.nansum(lick_freqs[:, :mid_t], axis=1)
    sums_post = np.nansum(lick_freqs[:, mid_t:], axis=1)
    denom_t = sums_pre + sums_post
    with np.errstate(invalid='ignore', divide='ignore'):
        lick_time_selectivity = np.where(denom_t > 0, sums_post / denom_t, np.nan)

    # ---- speed aligned to time ----
    speed_time_aligned = [np.vstack(speed)[:, 1] if len(speed) > 0 else []
                          for speed in beh['speed_times_aligned']]
    speed_time_aligned_array = np.vstack([zero_padding(speed_time_aligned[t], 4000)
                                          for t in range(1, n_trials) 
                                          # if valid_trials[t]
                                          ])
    speed_time_aligned_mean = np.nanmean(speed_time_aligned_array, axis=0)

    # ---- speed aligned to distance ----
    speed_dist = beh['speed_distances_aligned']
    speed_dist = np.vstack([
                            speed_dist[t] if len(speed_dist[t]) == 2200 else np.full(2200, np.nan)
                            for t in range(1, n_trials)
                            ])
    speed_dist_profile_mean = np.nanmean(speed_dist, axis=0)
    speed_dist_mean = np.nanmean(speed_dist)
    speed_dist_max = np.nanmean(np.nanmax(speed_dist, axis=1))
    speed_dist_median = np.nanmean(np.nanmedian(speed_dist, axis=1))

    # ---- reward rate ----
    rews = beh['reward_times']
    non_rew = np.isnan(rews)
    rew_rate = 1 - (np.sum(non_rew) / len(rews))

    # ---- assemble row (order MUST match df_beh_pool columns) ----
    df_beh_pool.loc[len(df_beh_pool)] = [
        anm,                            # anm_id
        rec, 
        # date,                           # date
        # int(ss),                        # session
        block_num,
        block_stat,
        passive_trials,
        random_trials,
        
        rew_rate,                       # rew_rate

        # running (time)
        speed_time_aligned,             # speed_time_profile
        speed_time_aligned_array,       # speed_time_profile_array
        speed_time_aligned_mean,        # speed_time_profile_mean

        # running (distance)
        speed_dist,                     # speed_distance_profile
        speed_dist_profile_mean,        # speed_distance_profile_mean
        speed_dist_mean,                # speed_distance_mean
        speed_dist_max,                 # speed_distance_max
        speed_dist_median,              # speed_distance_median

        # licks (distance)
        licks_dist_profile,             # lick_distance_profile
        licks_dist_profile_mean,        # lick_distance_profile_mean
        lick_dist_norm,                 # lick_dist_norm.
        lick_dist_fraction,             # lick_dist_fraction
        lick_distance_selectivity,      # lick_distance_selectivity

        # licks (time)
        licks_time_filtered,            # lick_time_profile
        lick_freqs,                     # lick_frequency_profile
        lick_freqs_mean,                # lick_frequency_mean
        lick_time_norm,                 # lick_time_norm.
        lick_time_fraction,             # lick_time_fraction
        lick_time_selectivity,          # lick_time_selectivity

        # first licks (distance)
        first_licks_dist,
        first_licks_dist_median,
        first_licks_dist_iqr,
        first_licks_dist_var,

        # first licks (time)
        first_licks_time,
        first_licks_time_median,
        first_licks_time_iqr,
        first_licks_time_var,
    ]

    return df_beh_pool

#%%
rec_lst = [
# 'AC324-20260527-02',
# 'AC324-20260528-02',
# 'AC324-20260529-02',
# 'AC324-20260530-02',
# 'AC324-20260531-02',    

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

'AC327-20260602-02',     
'AC327-20260603-02',     
'AC327-20260604-02',     
'AC327-20260605-02',     
'AC327-20260606-02',     
'AC327-20260607-02',     
'AC327-20260608-02',     
'AC327-20260609-02',     
'AC327-20260610-02',     
'AC327-20260611-02',
'AC327-20260612-02',     
    ]

df_all = pd.DataFrame()
for rec in rec_lst:
    p_beh = rf"Z:\Jingyu\dlight_learning\geco_dlight\behaviour_profile\{rec}.pkl"
    df = extract_behaviour_info(rec, p_beh)
    df_all = pd.concat([df_all, df], ignore_index=True)


#%% per-trial lick-distance selectivity across learning days

def _contiguous_spans(mask):
    """Yield (start, end) inclusive indices for contiguous True runs in `mask`."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
    starts = [idx[0]]
    ends = []
    for j in range(1, len(idx)):
        if idx[j] != idx[j - 1] + 1:
            ends.append(idx[j - 1])
            starts.append(idx[j])
    ends.append(idx[-1])
    for s, e in zip(starts, ends):
        yield s, e


def plot_per_trial_metric(df_all, key,
                          ax_w=3, ax_h=3,
                          ylabel=None, ylim=(0, 1.1),
                          marker='o',
                          color='grey', marker_size=1,
                          title_prefix='', wspace=0.15):
    """
    Plot a per-trial behaviour metric across sessions, one panel per session.

    Background bands mark trial conditions:
      - random trials: grey, alpha=0.3
      - passive-but-not-random trials: lightsteelblue, alpha=0.3

    Parameters
    ----------
    df_all : pd.DataFrame
        Pooled per-session behaviour summary (one row per session). Must contain
        columns 'passive_trials', 'random_trials', 'rec_id', 'anm_id' and `key`.
    key : str
        Column of df_all holding a per-trial 1D array to plot.
    ylabel : str, optional
        Y-axis label (defaults to `key`).
    ylim : tuple or None
        Y-axis limits, applied to the shared axis. Pass None to autoscale.
    color : str
        Line/marker colour.
    marker_size : float
        Marker size for per-trial points.
    title_prefix : str
        Prepended to the figure suptitle.

    Returns
    -------
    fig, axs
    """
    n_sess = len(df_all)
    if n_sess == 0:
        return None, None

    fig, axs = plt.subplots(1, n_sess, figsize=(ax_w * n_sess, ax_h),
                            sharey=True, dpi=150,
                            gridspec_kw={'wspace': wspace})
    if n_sess == 1:
        axs = [axs]

    for i, (ax, (_, row)) in enumerate(zip(axs, df_all.iterrows())):
        y = np.asarray(row[key], dtype=float)
        passive = np.asarray(row['passive_trials'], dtype=bool)
        random = np.asarray(row['random_trials'], dtype=bool)

        # passive/random come from block_numbers[1:]; metric is per-trial.
        # align by trimming the shared tail
        n = min(len(y), len(passive), len(random))
        y = y[-n:]
        passive = passive[-n:]
        random = random[-n:]
        x = np.arange(n)

        # background bands: random (grey), then passive-but-not-random (lightsteelblue)
        passive_only = passive & ~random
        for s, e in _contiguous_spans(random):
            ax.axvspan(s - 0.5, e + 0.5, color='grey',
                       alpha=0.3, lw=0, zorder=0)
        for s, e in _contiguous_spans(passive_only):
            ax.axvspan(s - 0.5, e + 0.5, color='lightblue',
                       alpha=0.3, lw=0, zorder=0)

        ax.plot(x, y, color=color, lw=0.8,
                marker=marker, 
                ms=marker_size, zorder=2)
        ax.set_xlabel('trial #')
        ax.set_xlim(-0.5, n - 0.5)

        rec_id = row['rec_id']
        title = (rec_id.split('-')[1]
                 if isinstance(rec_id, str) and '-' in rec_id
                 else str(rec_id))
        ax.set_title(title, fontsize=9)

        ax.spines[['top', 'right']].set_visible(False)
        if i == 0:
            ax.set_ylabel(ylabel if ylabel is not None else key)
        else:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0, labelleft=False)

    if ylim is not None:
        axs[0].set_ylim(*ylim)

    anm = df_all['anm_id'].iloc[0]
    metric_label = ylabel if ylabel is not None else key
    fig.suptitle(f'{title_prefix}{anm} - per-trial {metric_label} over learning',
                 fontsize=11)
    # leave room for suptitle; do NOT use tight_layout (it would override wspace)
    fig.subplots_adjust(wspace=wspace, top=0.85)
    return fig, axs

#%%
def plot_per_trial_heatmap(df_all, key,
                           ax_w=3, ax_h=3,
                           ylabel='lick distance (cm)',
                           y_extent=(0, 220),
                           cmap='Greys', vmin=None, vmax=None,
                           title_prefix='', wspace=0.15):
    """
    Plot a per-trial 2D behaviour profile as a heatmap, one panel per session.

    Each session's array is transposed so the bin axis (e.g. distance) sits on
    the y-axis with origin at the bottom; trials are on the x-axis.

    Parameters
    ----------
    df_all : pd.DataFrame
        Pooled per-session behaviour summary; the column `key` must hold a
        2D array of shape (n_trials, n_bins) per row.
    key : str
        Column name to plot (e.g. 'lick_dist_fraction', 'lick_dist_norm.').
    y_extent : tuple
        (y_min, y_max) for the heatmap's bin axis. Default (0, 220) → cm.
    cmap, vmin, vmax :
        Forwarded to imshow.

    Returns
    -------
    fig, axs
    """
    n_sess = len(df_all)
    if n_sess == 0:
        return None, None

    fig, axs = plt.subplots(1, n_sess, figsize=(ax_w * n_sess, ax_h),
                            sharey=True, dpi=150,
                            gridspec_kw={'wspace': wspace})
    if n_sess == 1:
        axs = [axs]

    for i, (ax, (_, row)) in enumerate(zip(axs, df_all.iterrows())):
        a = np.asarray(row[key], dtype=float)
        n_trials, _ = a.shape
        ax.imshow(a.T, cmap=cmap, vmin=vmin, vmax=vmax,
                  origin='lower', aspect='auto',
                  extent=(-0.5, n_trials - 0.5, y_extent[0], y_extent[1]))
        ax.set_xlim(-0.5, n_trials - 0.5)
        ax.set_ylim(y_extent[0], y_extent[1])
        ax.set_xlabel('trial #')

        rec_id = row['rec_id']
        title = (rec_id.split('-')[1]
                 if isinstance(rec_id, str) and '-' in rec_id
                 else str(rec_id))
        ax.set_title(title, fontsize=9)

        ax.spines[['top', 'right']].set_visible(False)
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0, labelleft=False)

    anm = df_all['anm_id'].iloc[0]
    fig.suptitle(f'{title_prefix}{anm} - per-trial {key} over learning',
                 fontsize=11)
    fig.subplots_adjust(wspace=wspace, top=0.85)
    return fig, axs


plot_key = 'first_licks_dist'
# plot_key = 'first_licks_time'
fig, axs = plot_per_trial_metric(df_all, plot_key,
                                 ylabel=plot_key,
                                 ylim=(0, 220),
                                 ax_w=1,
                                 # marker=None,
                                 wspace=0.05)
plt.show()

#%% per-trial lick distance fraction across days
fig, axs = plot_per_trial_heatmap(df_all, 'lick_dist_norm.',
                                  ylabel='lick distance (cm)',
                                  ax_h = 2,
                                  y_extent=(0, 220),
                                  cmap='tab20b',
                                  ax_w=1, wspace=0.05)
plt.show()
