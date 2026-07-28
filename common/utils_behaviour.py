# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:32:24 2026

@author: Jingyu Cao
"""
import numpy as np
from scipy.stats import ttest_ind

# def extract_licks(beh):
#     lick_times = beh['lick_times_aligned']
#     # tot_trials = len(lick_times)
#     for licks in lick_times:
#         if licks is np.nan:
#             first_lick_times.append(np.nan)
#         else:
#             licks = np.array(licks)
#             licks_filtered = licks[licks>500] # only include licks after 0.5s
#             if len(licks_filtered)>0:
#                 first_lick_times.append(licks_filtered[0])
#             else: # no licks after 0.5s
#                 first_lick_times.append(np.nan)
#     first_lick_times = np.array(first_lick_times)         
    
#     return  first_lick_times

def extract_first_licks(beh, align_by='time'):
    if align_by == 'time':
        lick_times = beh['lick_times_aligned']
        th = 500 # only include licks after 0.5s
    elif align_by == 'distance':
        lick_times = beh['lick_distances_aligned']
        th = 30 # only include licks after 30 cm
    first_lick_times = []
    # tot_trials = len(lick_times)
    for licks in lick_times:
        if licks is np.nan:
            first_lick_times.append(np.nan)
        else:
            licks = np.array(licks)
            licks_filtered = licks[licks>th] # only include licks after 0.5s
            if len(licks_filtered)>0:
                first_lick_times.append(licks_filtered[0])
            else: # no licks after 0.5s
                first_lick_times.append(np.nan)
    first_lick_times = np.array(first_lick_times)         
    
    return  first_lick_times
    
def extract_speed_trace(beh, align_by='time'):
    if align_by == 'time':
        speeds_aligned = [np.stack(speed)[:, 1] if len(speed)>0 else [] 
                          for speed in beh['speed_times_aligned']]
    elif align_by == 'distance':
        speeds_aligned = [speed if len(speed)>0 else [] 
                          for speed in beh['speed_distances_aligned']]
    return speeds_aligned

def speed_match(beh, idx_a, idx_b, tolerance=1.5, align_by='time',
                plot_validation=False, significance_level=0.05):
    '''
    

    Parameters
    ----------
    beh : TYPE
        DESCRIPTION.
    idx_a : TYPE
        DESCRIPTION.
    idx_b : TYPE
        DESCRIPTION.
    tolerance : TYPE, optional
        How tight the speed match should be. The default is 1.5.
    significance_level : float, optional
        Per-bin t-test threshold used to decide whether matching is needed.
        Matching is skipped when no bin has a finite p-value below this
        threshold. The default is 0.05.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if type(idx_a) is np.ndarray:
        idx_a = np.where(idx_a)[0]
    if type(idx_b) is np.ndarray:
        idx_b = np.where(idx_b)[0]

    if align_by=='time':
        trial_length = 4000 # ms
        speeds_aligned = [np.stack(speed)[:, 1] if len(speed)>0 else [] 
                              for speed in beh['speed_times_aligned']]
    elif align_by=='distance':
        trial_length = 1800 # mm
        speeds_aligned = [speed if len(speed)>0 else [] 
                              for speed in beh['speed_distances_aligned']]

    
    # binning
    speeds_binned = []
    speed_trial_idx = []
    if align_by=='time':
        for idx, speed in enumerate(speeds_aligned):
            if len(speed)<trial_length:
                speeds_binned.append(np.nan)
            else: # 0.5s bin, 7 bins
                speed_binned = speed[:trial_length].reshape(8, 500).mean(axis=1)
                speeds_binned.append(speed_binned)
                speed_trial_idx.append(idx)
    elif align_by=='distance':
        for idx, speed in enumerate(speeds_aligned):
            if len(speed)<trial_length:
                speeds_binned.append(np.nan)
            else: # 30 cm bin, 5 bins
                speed_binned = speed[:trial_length].reshape(6, 300).mean(axis=1)
                speeds_binned.append(speed_binned)
                speed_trial_idx.append(idx)
    idx_a_speed = [i for i in idx_a if i in speed_trial_idx]
    idx_b_speed = [i for i in idx_b if i in speed_trial_idx]
    if len(idx_a_speed)==0 or len(idx_b_speed)==0:
        return  #skip sessions without out any early or late trials                        
    speed_a_binned = np.asarray([speeds_binned[i] for i in idx_a_speed])
    speed_b_binned = np.asarray([speeds_binned[i] for i in idx_b_speed])

    def _binwise_ttest(group_a, group_b):
        return np.asarray([
            ttest_ind(group_a[:, i], group_b[:, i]).pvalue
            for i in range(group_a.shape[1])
        ])

    # Test before matching. NaN p-values provide no evidence of a difference;
    # only a finite, significant bin triggers the matching step.
    pre_match_p_values = _binwise_ttest(speed_a_binned, speed_b_binned)
    matching_needed = np.any(
        np.isfinite(pre_match_p_values)
        & (pre_match_p_values < significance_level)
    )
    speed_a_miu = np.mean(speed_a_binned, axis=0)
    speed_a_std = np.std(speed_a_binned, axis=0)
    speed_b_miu = np.mean(speed_b_binned, axis=0)
    speed_b_std = np.std(speed_b_binned, axis=0)
    
    # Cross-group speed matching: keep trials from each group that fall
    # within the OTHER group's mean ± tolerance*std.
    # This ensures both groups have overlapping speed distributions.
    if matching_needed:
        a_bound_low = speed_a_miu-tolerance*speed_a_std
        a_bound_high = speed_a_miu+tolerance*speed_a_std
        b_bound_low = speed_b_miu-tolerance*speed_b_std
        b_bound_high = speed_b_miu+tolerance*speed_b_std

        # Keep group A trials within group B's speed range.
        a_in_bound = np.all(
            (speed_a_binned >= b_bound_low)
            & (speed_a_binned <= b_bound_high),
            axis=1,
        )
        # Keep group B trials within group A's speed range.
        b_in_bound = np.all(
            (speed_b_binned >= a_bound_low)
            & (speed_b_binned <= a_bound_high),
            axis=1,
        )
        a_in_bound_idx = [
            idx_a_speed[i] for i in np.where(a_in_bound)[0]
        ]
        b_in_bound_idx = [
            idx_b_speed[i] for i in np.where(b_in_bound)[0]
        ]
    else:
        # Every bin is already non-significant; retain all eligible trials.
        a_in_bound_idx = idx_a_speed
        b_in_bound_idx = idx_b_speed
    
    # if len(early_in_bound_idx)<10 or len(late_in_bound_idx)<10:
    #     continue # skip sessions without enough trials
    
    # check speed match using t-test
    a_speed_binned_inbound = np.vstack([speeds_binned[i] for i in a_in_bound_idx])
    b_speed_binned_inbound = np.vstack([speeds_binned[i] for i in b_in_bound_idx])
    p_values = _binwise_ttest(
        a_speed_binned_inbound,
        b_speed_binned_inbound,
    )
    
    if plot_validation:
        import matplotlib.pyplot as plt
        import common.plotting_functions_Jingyu as pf
        from common.utils_basic import zero_padding
    
        # # plot speed match validation
        fig, axs = plt.subplots(1, 2, figsize=(4,2), dpi=300); fig.tight_layout()
        # plt.suptitle=(f'{rec}')
        early_speeds = [zero_padding(speeds_aligned[i], trial_length) for i in idx_a]
        late_speeds = [zero_padding(speeds_aligned[i], trial_length) for i in idx_b]
        early_speeds_matched = [zero_padding(speeds_aligned[i], trial_length) for i in a_in_bound_idx]
        late_speeds_matched = [zero_padding(speeds_aligned[i], trial_length) for i in b_in_bound_idx]
        ax=axs[0]
        pf.plot_mean_trace(early_speeds, ax, color='grey')
        pf.plot_mean_trace(late_speeds, ax, color='steelblue')
        ax.set(title='before_match', ylabel='speed', xlabel=align_by)
        ax=axs[1]
        pf.plot_mean_trace(early_speeds_matched, ax, color='grey')
        pf.plot_mean_trace(late_speeds_matched, ax, color='steelblue')
        ax.set(title='after_match', ylabel='speed', xlabel=align_by)
        # plt.savefig(figure_out+r'\speed_match_{}.png'.format(rec))
        plt.show()
        # plt.close()
    
    return a_in_bound_idx, b_in_bound_idx, p_values

# def early_late_lick_trials(df_beh,
#                            SPLIT_MODE = 'threshold',
#                            EARLY_THRESH = None,
#                            LATE_THRESH = None,
#                            xaxis = 'distance', # distance or time,
#                            SPEED_MATCH = True,
#                            plot_speed_match = True
#                            ):
                           
#     if xaxis == 'distance':
#         x_unit = 'cm'
#     if xaxis == 'time':
#         x_unit = 's'
        
#     if (EARLY_THRESH is None):
#         if xaxis == 'distance':
#             EARLY_THRESH = 100 
#         if xaxis == 'time':
#             EARLY_THRESH = 1.5 
#     if (LATE_THRESH is None):
#         if xaxis == 'distance':
#             LATE_THRESH = 120 
#         if xaxis == 'time':
#             LATE_THRESH = 2 
    
#     first_licks = extract_first_licks(df_beh, align_by='distance')
#     # n_trials = len(first_licks)
        
#     # first_lick_per_lap = np.full(n_laps, np.nan)
#     # for i_lap in range(n_laps):
#     #     t_idx = lap_trial_idx[i_lap]
#     #     if 0 <= t_idx < n_trials:
#     #         first_lick_per_lap[i_lap] = first_licks[t_idx]

#     # Lap indices for early and late lick
#     if SPLIT_MODE == 'threshold':
#         early_idx = np.where(first_licks < EARLY_THRESH)[0]
#         late_idx  = np.where(first_licks > LATE_THRESH)[0]
#         print(f"  early-lick laps (<{EARLY_THRESH} {x_unit}): {len(early_idx)}")
#         print(f"  late-lick  laps (>{LATE_THRESH} {x_unit}):  {len(late_idx)}")
#     elif SPLIT_MODE == 'median':
#         median_lick = np.nanmedian(first_licks)
#         early_idx = np.where(first_licks < median_lick)[0]
#         late_idx  = np.where(first_licks >= median_lick)[0]
#         # else:
#         #     early_idx = np.array([], dtype=int)
#         #     late_idx  = np.array([], dtype=int)
#         #     median_lick = np.nan
#         # print(f"  median first-lick: {median_lick:.1f} cm")
#         # print(f"  early-lick laps (< median): {len(early_idx)}")
#         # print(f"  late-lick  laps (>= median): {len(late_idx)}")

#     # Optional speed matching: filter early/late to speed-matched subsets
#     if SPEED_MATCH and len(early_idx) > 0 and len(late_idx) > 0:
#         result = speed_match(df_beh, early_idx, late_idx,
#                              align_by= xaxis, 
#                              tolerance=2, 
#                              plot_validation=plot_speed_match)
#         if result is not None:
#             early_matched_trials, late_matched_trials, p_vals = result
#             print(f"  after speed match: early={len(early_matched_trials)}, late={len(late_matched_trials)}"
#                   f"  (p={p_vals})")

#             # # Collect distance-aligned speed traces for validation plot
#             # speeds_dist_aligned = [s if len(s) > 0 else []
#             #                        for s in beh['speed_distances_aligned']]
#             # n_speed_samples = 1800  # 180 cm at 0.1 cm resolution

#             # def _get_speed_traces(trial_ids):
#             #     traces = []
#             #     for t in trial_ids:
#             #         s = speeds_dist_aligned[t]
#             #         if len(s) >= n_speed_samples:
#             #             traces.append(s[:n_speed_samples])
#             #     return np.array(traces) if traces else None

#             # speed_val_per_day.append({
#             #     'rec': rec, 'day': day_i,
#             #     'early_before': _get_speed_traces(early_trial_ids_before),
#             #     'late_before':  _get_speed_traces(late_trial_ids_before),
#             #     'early_after':  _get_speed_traces(list(early_matched_trials)),
#             #     'late_after':   _get_speed_traces(list(late_matched_trials)),
#             # })

#             # Use speed-matched lap indices going forward
#             early_idx = early_idx
#             late_idx  = late_idx
#         else:
#             print("  speed match failed — using unmatched sets")
#             # speed_val_per_day.append(None)


def align_pulses(df_beh, max_pulse_delay=500):
    """Assign stim pulses to trials (indexed by run_onset) and flag misaligned ones.

    Parameters
    ----------
    df_beh : dict-like
        Must expose 'pulse_start_times', 'pulse_end_times', 'run_onsets',
        and 'frame_times'.
    max_pulse_delay : float
        Threshold on |first_pul_start - run_onset|. Trials exceeding this AND
        the neighbor they likely stole a pulse from are marked invalid. Same
        units as run_onsets / pulse times.

    Returns
    -------
    dict with:
        paired_starts, paired_ends : arrays of matched pulse start/end times
        pulse_trains               : list length n_trials of [(s,e), ...]
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
    bad = np.where(np.abs(onset_to_first_pulse) > max_pulse_delay)[0]
    # bad = np.where((onset_to_first_pulse < 0) | 
    #                (onset_to_first_pulse > max_pulse_delay)
    #                 )[0]
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
        'frames_per_trial': frames_per_trial,
        'stim_covered_frames': stim_covered_frames,
        'train_covered_frames': train_covered_frames,
        'trials_with_stim': trials_with_stim,
        'onset_to_first_pulse': onset_to_first_pulse,
        'valid_trials': valid_trials,
    }
