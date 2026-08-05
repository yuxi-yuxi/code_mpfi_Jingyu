# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:53:26 2026

@author: Jingyu Cao
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Spyder's %runfile --wdir runs this file from the lc_stim_gcamp directory.
# Add the repository root so sibling packages such as common remain importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lc_stim_gcamp.calculate_dff import process_F_trace
from common.utils_imaging import percentile_dff, align_trials
from common.trial_selection import seperate_valid_trial
from common.utils_behaviour import speed_match, extract_first_licks
from place_cell_analysis import place_cell_functions as pcf
from common.robust_sd_filter import robust_filter_along_axis
import common.plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()
from common.event_response_quantification import quantify_event_response

def align_pulses(df_beh, max_pulse_delay=1000):
    """Assign stim pulses to trials (indexed by run_onset) and flag misaligned ones.

    Parameters
    ----------
    df_beh : dict-like
        Must expose 'pulse_start_times', 'pulse_end_times', 'run_onsets',
        'frame_times', and one 'pulse_descriptions' entry per trial.
    max_pulse_delay : float
        Threshold on |first_pul_start - run_onset|. Trials exceeding this AND
        the neighbor they likely stole a pulse from are marked invalid. Same
        units as run_onsets / pulse times.

    Returns
    -------
    dict with:
        paired_starts, paired_ends : arrays of matched pulse start/end times
        pulse_trains               : list length n_trials of [(s,e), ...]
        pulse_descriptions         : list length n_trials, copied from df_beh
        frames_per_trial           : list length n_trials of frame-index arrays
        trials_with_stim           : bool mask, True where pulse_trains[i] non-empty
        onset_to_first_pulse       : float array, first pul_start - run_onset (NaN if no stim)
        valid_trials               : bool mask, False for misassigned trials + contaminated neighbors
    """
    frame_times = np.asarray(df_beh['frame_times'])
    pul_start_arr = np.asarray(df_beh['pulse_start_times'])
    pul_end_arr = np.asarray(df_beh['pulse_end_times'])
    run_onset_times = df_beh['run_onsets']

    # Pair each start with the first end at/after it, dropping unpaired extras.
    paired_starts, paired_ends = [], []
    j = 0
    for s in pul_start_arr:
        while j < len(pul_end_arr) and pul_end_arr[j] < s:
            j += 1
        if j >= len(pul_end_arr):
            break
        paired_starts.append(s)
        paired_ends.append(pul_end_arr[j])
        j += 1
    paired_starts = np.asarray(paired_starts)
    paired_ends = np.asarray(paired_ends)

    # Assign each paired pulse to the trial whose run_onset is closest in time.
    # Online vs. offline run-onset detection can differ by a few frames, so
    # strict interval membership can misassign pulses at the trial boundary.
    onsets = np.array(
        [np.nan if t is None else t for t in run_onset_times], dtype=float
    )
    n_trials = len(onsets)

    # Pulse descriptions are already recorded in behavioral-trial order. Keep
    # them in that same trial space rather than trying to pair them with each
    # pulse edge (a stimulation train contains many pulse edges).
    pulse_descriptions = list(df_beh['pulse_descriptions'])
    # if len(pulse_descriptions) != n_trials:
    #     raise ValueError(
    #         "'pulse_descriptions' must contain one entry per run_onset "
    #         f"({len(pulse_descriptions)} descriptions for {n_trials} trials)"
    #     )

    valid_mask = ~np.isnan(onsets)
    valid_trial_idx = np.where(valid_mask)[0]
    valid_onsets = onsets[valid_mask]

    pulse_trains = [[] for _ in range(n_trials)]

    if len(paired_starts) and len(valid_onsets):
        sort_order = np.argsort(valid_onsets)
        sorted_onsets = valid_onsets[sort_order]
        sorted_trial_idx = valid_trial_idx[sort_order]

        pos = np.searchsorted(sorted_onsets, paired_starts)
        left = np.clip(pos - 1, 0, len(sorted_onsets) - 1)
        right = np.clip(pos, 0, len(sorted_onsets) - 1)
        left_d = np.abs(paired_starts - sorted_onsets[left])
        right_d = np.abs(paired_starts - sorted_onsets[right])
        nearest = np.where(left_d <= right_d, left, right)
        assigned_trial = sorted_trial_idx[nearest]

        for s, e, t_i in zip(paired_starts, paired_ends, assigned_trial):
            pulse_trains[t_i].append((s, e))

    # The last pulse's pul_end isn't written until the NEXT trial starts, so
    # its recorded end time is bogus (way after the actual pulse). Replace
    # it with (last_pul_start + mean duration of the earlier pulses in the
    # same train); fall back to the global mean if the train has only one
    # pulse.
    inner_durs = [
        e - s for train in pulse_trains if len(train) > 1 for s, e in train[:-1]
    ]
    global_mean_dur = float(np.mean(inner_durs)) if inner_durs else np.nan

    for train in pulse_trains:
        if not train:
            continue
        if len(train) > 1:
            mean_dur = float(np.mean([e - s for s, e in train[:-1]]))
        else:
            mean_dur = global_mean_dur
        if not np.isnan(mean_dur):
            last_s = train[-1][0]
            train[-1] = (last_s, last_s + mean_dur)

    # Rebuild paired_ends from the corrected trains so the flat output is
    # consistent with pulse_trains. paired_starts are unchanged.
    all_pulses = sorted(
        (p for train in pulse_trains for p in train), key=lambda p: p[0]
    )
    if all_pulses:
        paired_ends = np.asarray([e for _, e in all_pulses])

    # Per-trial: frame indices whose acquisition window overlaps any pulse.
    # Individual pulses can be much shorter than a frame interval (e.g. ~40us
    # pulses vs. ~33ms frames), so requiring frame_times to fall INSIDE [s, e]
    # misses almost every pulse. Instead, treat frame i as covering
    # [frame_times[i], frame_times[i+1]) and pick indices [i_s .. i_e] where
    # i_s is the frame containing s and i_e is the frame containing e.
    n_frames = len(frame_times)
    stim_frames_per_trial = []
    for train in pulse_trains:
        per_pulse = []
        for s, e in train:
            i_s = max(0, np.searchsorted(frame_times, s, side='right') - 1)
            i_e = max(0, np.searchsorted(frame_times, e, side='right') - 1)
            per_pulse.append(np.arange(i_s, min(n_frames, i_e + 1)))
        stim_frames_per_trial.append(
            np.unique(np.concatenate(per_pulse)) if per_pulse
            else np.array([], dtype=int)
        )

    # Flat, deduped: all frames physically covered by any pulse.
    non_empty = [f for f in stim_frames_per_trial if len(f)]
    stim_covered_frames = (
        np.unique(np.concatenate(non_empty)) if non_empty
        else np.array([], dtype=int)
    )

    # Per-trial: continuous span from first to last stim-covered frame
    # (fills in inter-pulse gaps within the same trial).
    frames_per_trial = [
        np.arange(f.min(), f.max() + 1) if len(f) else np.array([], dtype=int)
        for f in stim_frames_per_trial
    ]

    # Flat, deduped: continuous train coverage across all trials.
    non_empty = [f for f in frames_per_trial if len(f)]
    train_covered_frames = (
        np.unique(np.concatenate(non_empty)) if non_empty
        else np.array([], dtype=int)
    )

    trials_with_stim = np.array([bool(t) for t in pulse_trains], dtype=bool)

    # Delay from each trial's run_onset to its first pul_start.
    onset_to_first_pulse = np.full(n_trials, np.nan)
    for i, train in enumerate(pulse_trains):
        if train and not np.isnan(onsets[i]):
            onset_to_first_pulse[i] = train[0][0] - onsets[i]

    # A pulse landing late (positive delay) likely belongs to trial i+1; one
    # landing early (negative delay) likely belongs to trial i-1. Mark the
    # bad trial AND the neighbor it stole from.
    valid_trials = np.ones(n_trials, dtype=bool)
    # bad = np.where(np.abs(onset_to_first_pulse) > max_pulse_delay)[0]
    bad = np.where((onset_to_first_pulse < 0) | 
                   (onset_to_first_pulse > max_pulse_delay)
                    )[0]
    for i in bad:
        valid_trials[i] = False
        if onset_to_first_pulse[i] > 0 and i + 1 < n_trials:
            valid_trials[i + 1] = False
        elif onset_to_first_pulse[i] < 0 and i - 1 >= 0:
            valid_trials[i - 1] = False

    return {
        'paired_starts': paired_starts,
        'paired_ends': paired_ends,
        'pulse_trains': pulse_trains,
        'pulse_descriptions': pulse_descriptions,
        'frames_per_trial': frames_per_trial,
        'stim_covered_frames': stim_covered_frames,
        'train_covered_frames': train_covered_frames,
        'trials_with_stim': trials_with_stim,
        'onset_to_first_pulse': onset_to_first_pulse,
        'valid_trials': valid_trials,
    }

def classify_pyrs(df_stats, 
                  pyrUp_thresh=1.12,
                  pyrDown_thresh=1/1.12,
                  profile_mean_thresh=None,
                  profile_max_thresh=None,
                  ratio_key = 'response_ratio',
                  ):
    df_stats_sorted = df_stats.copy() # withou modifying the original pooled data
    
    df_stats_sorted['profile_mean'] = df_stats_sorted['mean_profile'].apply(lambda x: np.nanmean(x))
    df_stats_sorted['profile_exm'] = (
    df_stats_sorted['mean_profile'].apply(
        lambda x: np.nanmax(np.abs(x))
        if np.any(np.isfinite(x)) else np.nan
    )
)

    if  profile_mean_thresh is not None:  
        df_stats_sorted['valid'] = df_stats_sorted['profile_mean'].apply(lambda x: 0<x<profile_mean_thresh)     
    elif profile_max_thresh is not None:
        df_stats_sorted['valid'] = df_stats_sorted['profile_exm'].apply(lambda x: x<profile_max_thresh)
    else:
        df_stats_sorted['valid'] = True
         
                                
    df_stats_sorted['pyrUp'] = np.where(
                                (df_stats_sorted[ratio_key]> pyrUp_thresh)
                                &(df_stats_sorted['valid']),
                                True, False)
    df_stats_sorted['pyrDown'] = np.where(
                                (df_stats_sorted[ratio_key]<pyrDown_thresh)
                                &(df_stats_sorted['valid']),
                                True, False)
    df_stats_sorted['pyrStable'] = (~df_stats_sorted['pyrUp'])&(~df_stats_sorted['pyrDown'])&(df_stats_sorted['valid'])
    
    
    # try:  
    #     df_stats_sorted.loc[df_stats_sorted['pyrUp'],     'geco_type'] = 'Up'
    #     df_stats_sorted.loc[df_stats_sorted['pyrDown'],   'geco_type'] = 'Down'
    #     df_stats_sorted.loc[df_stats_sorted['pyrStable'], 'geco_type'] = 'Stable'

    # except:
    #     print('error')
    
    
    return df_stats_sorted


#%%
# 'AC334-20260724-02',
# 'AC334-20260725-02'
# 'AC333-20260724-02',
# 'AC333-20260724-04',
# 'AC333-20260725-02'
# 'AC333-20260725-04'
# 'AC322-20260505-02' 
# 'AC322-20260505-04'
# 'AC322-20260506-02'
# 'AC322-20260506-04' # issue session
# 'AC322-20260507-02'
# 'AC334-20260728-02',
# 'AC334-20260728-04',
rec = 'AC336-20260803-04'

anm, date, ss = rec.split('-')
df_beh = pd.read_pickle(rf"Z:\Jingyu\raw_data\lc_stim_gcamp\processed_data\{rec}\{rec}.pkl")
dff, is_active_soma, shutter_masks = process_F_trace(rec,
                                                     active_soma_only=True,
                                                     overwrite={"shutter_mask": True},
                                                     )  
# filter for extreme values
rsd_factor=3
# thresh =pcf.dff_thresh(dff, hard_thresh=100, factor=5)
kept_frames = ~shutter_masks
dff_sd_kept = robust_filter_along_axis(
    dff[:, kept_frames],
    factor=rsd_factor,
)
dff_sd = np.full_like(dff, np.nan)
dff_sd[:, kept_frames] = dff_sd_kept
# dff_sd[abs(dff_sd)>thresh]=np.nan
dff = dff_sd

# pulse alignment
pulse_method = [trial_stat[15] for trial_stat in df_beh['trial_statements']]
pulse_method = max(pulse_method)
if pulse_method == '2': # run-onset pulse
    df_pulse = align_pulses(df_beh, max_pulse_delay=500)
elif pulse_method == '7': # pulse after 1500 ms post run onset
    df_pulse = align_pulses(df_beh, max_pulse_delay=2000)
    
stim_covered_frames = df_pulse['stim_covered_frames']
train_covered_frames = df_pulse['train_covered_frames']

# black frames
dff_stim_masked = dff.copy()
# dff_stim_masked[:, train_covered_frames] = np.nan
dff_stim_masked[:, shutter_masks] = np.nan

# trial alignment
bef, aft = 2, 4
dff_aligned = align_trials(dff_stim_masked, 'run', df_beh, bef, aft)

#%% select stim and ctrl trials
stim_trials = df_pulse['trials_with_stim']
stim_valid_trials = df_pulse['valid_trials']&stim_trials
stim_plus_one_trials = np.zeros_like(stim_valid_trials)
stim_plus_one_trials[1:] = stim_valid_trials[:-1]
stim_plus_two_trials = np.zeros_like(stim_valid_trials)
stim_plus_two_trials[2:] = stim_valid_trials[:-2]
beh_valid_trials = seperate_valid_trial(df_beh, time_thresh=10000)
run_onset_frames = np.array(df_beh['run_onset_frames'])
block_num = np.array(df_beh['block_numbers'])

ctrl_valid_trials = np.array(beh_valid_trials)&np.array(stim_plus_two_trials)
# ctrl_valid_trials = np.array(beh_valid_trials)&(block_num == 1)
stim_valid_trials = np.array(beh_valid_trials)&(stim_valid_trials)

#%% quantify ru-onset response
baseline_window=(-1, 0)
response_window=(1, 1.5) # seconds
profile_max_thresh=3
profile_min_thresh=0.1
pyrUp_thresh = 1.5
pyrDown_thresh = 1/pyrUp_thresh

# ctrl
df_response_ctrl = quantify_event_response(corrected_traces = dff_stim_masked, 
                                    event_frames=run_onset_frames[ctrl_valid_trials],
                                    baseline_window=baseline_window, 
                                    response_window=response_window, # seconds
                                    dilation_k = 0,
                                    imaging_rate=30.0, shuffle_test=0,
                                    shuffle_params={'times': 1000,
                                                    'pre_event_window':  2, # seconds
                                                    'post_event_window': 4 }
                                    )
df_response_ctrl['profile_mean'] = df_response_ctrl['mean_profile'].apply(lambda x: np.nanmean(x))
df_response_ctrl['profile_exm']  = df_response_ctrl['mean_profile'].apply(
    lambda x: np.nanmax(np.abs(x))
    if np.any(np.isfinite(x)) else np.nan
)
df_response_ctrl['is_soma'] = df_response_ctrl['profile_exm'].apply(lambda x: x<profile_max_thresh)
df_response_ctrl['is_soma'] = (df_response_ctrl['is_soma']
                               &(df_response_ctrl['profile_mean']>profile_min_thresh))
# df_response_ctrl['is_soma'] = True

# stim
df_response_stim = quantify_event_response(corrected_traces = dff_stim_masked, 
                                    event_frames=run_onset_frames[stim_valid_trials],
                                    baseline_window=baseline_window, 
                                    response_window=response_window, 
                                    dilation_k = 0,
                                    imaging_rate=30.0, shuffle_test=0,
                                    shuffle_params={'times': 1000,
                                                    'pre_event_window':  2, # seconds
                                                    'post_event_window': 4 },
                                    stim_window=shutter_masks
                                    )
# df_response_stim['is_soma'] = (df_response_stim['valid'])&(df_response_stim['response_ratio']>0) 
df_response_stim['profile_mean'] = df_response_stim['mean_profile'].apply(lambda x: np.nanmean(x))
df_response_stim['profile_exm'] = df_response_stim['mean_profile'].apply(
    lambda x: np.nanmax(np.abs(x))
    if np.any(np.isfinite(x)) else np.nan
)
df_response_stim['is_soma'] = df_response_stim['profile_exm'].apply(lambda x: x<profile_max_thresh)
df_response_stim['is_soma'] = (df_response_stim['is_soma']
                               &(df_response_stim['profile_mean']>profile_min_thresh))


# assign pyrUp and pyrDown
assert df_response_ctrl['roi_id'].equals(df_response_stim['roi_id'])
both_soma = (
    df_response_ctrl['is_soma'].astype(bool)
    & df_response_stim['is_soma'].astype(bool)
)


df_response_ctrl_sorted = classify_pyrs(df_response_ctrl.loc[
    # df_response_ctrl['is_soma']
    both_soma
    ], 
                                   pyrUp_thresh=pyrUp_thresh,
                                   pyrDown_thresh=pyrDown_thresh,
                                   profile_mean_thresh=None,
                                   profile_max_thresh=None,
                                   )
pyrup_ctrl = df_response_ctrl_sorted.loc[(df_response_ctrl_sorted['pyrUp']&
                                          # df_response_ctrl_sorted['is_soma']
                                          both_soma
                                          )]
                                         #, 'roi_id'].to_list()
print(f'Number of pyrUp cells for ctrl condition: {len(pyrup_ctrl)}')

df_response_stim_sorted = classify_pyrs(df_response_stim.loc[
    # df_response_stim['is_soma']
    both_soma
    ], 
                                   pyrUp_thresh=pyrUp_thresh,
                                   pyrDown_thresh=pyrDown_thresh,
                                   profile_mean_thresh=None,
                                   profile_max_thresh=None,
                                   )
pyrup_stim = df_response_stim_sorted.loc[(df_response_stim_sorted['pyrUp']&
                                          # df_response_stim_sorted['is_soma']
                                          both_soma
                                          )]
                                         # , 'roi_id'].to_list()

print(f'Number of pyrUp cells for stim condition: {len(pyrup_stim)}')


#%% plot heatmaps
# df_response_ctrl_pyr = df_response_ctrl_sorted.loc[df_response_ctrl_sorted['is_soma']]
# df_response_stim_pyr = df_response_stim_sorted.loc[df_response_stim_sorted['is_soma']]
df_response_ctrl_pyr = df_response_ctrl_sorted.loc[both_soma]
df_response_stim_pyr = df_response_stim_sorted.loc[both_soma]

prefix = 'ss1_baseline'
prof_col_heatmap = 'mean_profile'
ratio_col = 'response_ratio'
save_plot = 0

fig, ax=pf.plot_pyr_sorted_heatmap(df_response_ctrl_pyr, rec, bef, aft, 'ss1', prefix=prefix,
                               activity_profile=prof_col_heatmap, ratio=ratio_col,
                               plot_mean=0)
ax.set(xlim=(-1, 4), title=rec)
plt.show()
fig, ax=pf.plot_pyr_sorted_heatmap(df_response_stim_pyr, rec, bef, aft, 'ss1', prefix=prefix,
                               activity_profile=prof_col_heatmap, ratio=ratio_col,
                               plot_mean=0)
ax.set(xlim=(-1, 4), title=rec)
plt.show()
# save_fig(fig, OUT_DIR_FIG, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)

#%% plot dff traces

# ctrl_trials_trace = np.nanmean(ctrl_trials_trace, axis=1)
stim_trials_trace = np.stack(pyrup_stim['mean_profile'])
# stim_trials_trace[:, 60:95] = np.nan
ctrl_trials_trace = np.stack(pyrup_ctrl['mean_profile'])
fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
xaxis= np.arange(30*(bef+aft))/30-bef
pf.plot_mean_trace(stim_trials_trace, ax, xaxis, color='blue')
pf.plot_mean_trace(ctrl_trials_trace, ax, xaxis, color='green')
ax.set(title=f'{rec}\nn_stim: {np.sum(stim_valid_trials)}, n_ctrl: {np.sum(ctrl_valid_trials)}',
       xlim=(-1, 4)
       # ylim=(0, 1),
       )

#%%
first_lick_distance = extract_first_licks(df_beh, align_by='distance')
first_lick_time = extract_first_licks(df_beh, align_by='time')

ctrl_valid_trials = np.array(beh_valid_trials)&np.array(stim_plus_two_trials)
stim_matched, ctrl_matched, pvalue = speed_match(df_beh, stim_valid_trials, ctrl_valid_trials,
                                                 align_by='distance', 
                                                 tolerance=1.5, 
                                                 plot_validation=1)
stim_lick_distance = first_lick_distance[stim_valid_trials]
ctrl_lick_distance = first_lick_distance[ctrl_valid_trials]
fig, ax = plt.subplots(figsize=(1, 2))
pf.plot_bar_with_unpaired_scatter(ax, ctrl_lick_distance, stim_lick_distance,
                                  ylabel='1st lick dist')
ax.set_title(rec, size=8)

stim_lick_time = first_lick_time[stim_valid_trials]/1000
ctrl_lick_time = first_lick_time[ctrl_valid_trials]/1000
fig, ax = plt.subplots(figsize=(1, 2))
pf.plot_bar_with_unpaired_scatter(ax, ctrl_lick_time, stim_lick_time,
                                  ylabel='1st lick time')
ax.set_title(rec, size=8)


#%%
fig, ax = plt.subplots(figsize=(3, 3))
ax.hist(df_pulse['onset_to_first_pulse'])
ax.set(xlabel='run onset to first pulse delay (ms)',
       ylabel='trial count',
       title=f'{rec}')
