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
    required = {'frame_times', 'pulse_start_times', 'pulse_end_times',
                'run_onsets'}
    missing = required.difference(df_beh)
    if missing:
        raise KeyError(
            f'align_pulses requires trial-based data; missing {sorted(missing)}. '
            'Use align_pulses_to_stim_cycles for sensor-only sessions.'
        )

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


def align_pulses_to_stim_cycles(df_beh):
    '''Align sensor-session pulses to $PT cycles and $BT/$BE blocks.

    Unlike ``align_pulses``, this function does not require run onsets or a
    trial structure. A stimulation cycle is the interval from a ``$PT,1``
    timestamp to the corresponding ``$PT,0`` timestamp. Pulses are assigned
    to the cycle containing their start time, and cycles are assigned to the
    block containing their start time.

    Parameters
    ----------
    df_beh : dict-like
        Output from ``process_imaging_block_pulse_session``. It must contain
        frame times, pulse edges, pulse-train edges, and block boundaries.

    Returns
    -------
    dict
        Cycle-level pulse lists and frame ranges, block membership and frame
        ranges, flat stimulation masks, and validation flags. All frame
        indices refer to ``df_beh['frame_times']``.
    '''
    required = {
        'frame_times', 'pulse_start_times', 'pulse_end_times',
        'pulse_train_start_times', 'pulse_train_end_times',
        'block_start_times', 'block_end_times', 'block_ids',
    }
    missing = required.difference(df_beh)
    if missing:
        raise KeyError(f'Missing sensor-session fields: {sorted(missing)}')

    frame_times = np.asarray(df_beh['frame_times'], dtype=float)
    pulse_starts = np.asarray(df_beh['pulse_start_times'], dtype=float)
    pulse_ends = np.asarray(df_beh['pulse_end_times'], dtype=float)
    cycle_starts = np.asarray(
        df_beh['pulse_train_start_times'], dtype=float
    )
    cycle_ends = np.asarray(df_beh['pulse_train_end_times'], dtype=float)
    block_starts = np.asarray(df_beh['block_start_times'], dtype=float)
    block_ends = np.asarray(df_beh['block_end_times'], dtype=float)
    block_ids = np.asarray(df_beh['block_ids'], dtype=int)

    if frame_times.ndim != 1 or not len(frame_times):
        raise ValueError('frame_times must be a non-empty one-dimensional array')
    if np.any(np.diff(frame_times) < 0):
        raise ValueError('frame_times must be sorted')
    if not (len(block_starts) == len(block_ends) == len(block_ids)):
        raise ValueError('Block starts, ends, and IDs must have equal lengths')

    def _pair_edges(starts, ends, label):
        paired_starts, paired_ends = [], []
        end_index = 0
        for start in starts:
            while end_index < len(ends) and ends[end_index] < start:
                end_index += 1
            if end_index >= len(ends):
                break
            paired_starts.append(start)
            paired_ends.append(ends[end_index])
            end_index += 1
        if len(paired_starts) != len(starts) or len(paired_ends) != len(ends):
            raise ValueError(
                f'Could not pair every {label} start/end edge: '
                f'{len(starts)} starts, {len(ends)} ends'
            )
        return np.asarray(paired_starts), np.asarray(paired_ends)

    paired_starts, paired_ends = _pair_edges(
        pulse_starts, pulse_ends, 'pulse'
    )
    cycle_starts, cycle_ends = _pair_edges(
        cycle_starts, cycle_ends, 'cycle'
    )

    def _frames_in_interval(start, end):
        first = max(0, np.searchsorted(frame_times, start, side='right') - 1)
        last = max(0, np.searchsorted(frame_times, end, side='right') - 1)
        if end < frame_times[0] or start > frame_times[-1]:
            return np.array([], dtype=int)
        return np.arange(first, min(len(frame_times), last + 1), dtype=int)

    pulse_trains = []
    pulse_frames_per_cycle = []
    frames_per_cycle = []
    assigned_pulse = np.zeros(len(paired_starts), dtype=bool)
    for start, end in zip(cycle_starts, cycle_ends):
        in_cycle = (paired_starts >= start) & (paired_starts <= end)
        assigned_pulse |= in_cycle
        train = list(zip(paired_starts[in_cycle], paired_ends[in_cycle]))
        pulse_trains.append(train)
        per_pulse = [_frames_in_interval(s, e) for s, e in train]
        non_empty = [frames for frames in per_pulse if len(frames)]
        pulse_frames_per_cycle.append(
            np.unique(np.concatenate(non_empty)) if non_empty
            else np.array([], dtype=int)
        )
        frames_per_cycle.append(_frames_in_interval(start, end))

    cycle_block_indices = np.full(len(cycle_starts), -1, dtype=int)
    for block_index, (start, end) in enumerate(zip(block_starts, block_ends)):
        in_block = (cycle_starts >= start) & (cycle_starts <= end)
        cycle_block_indices[in_block] = block_index
    cycle_block_ids = np.full(len(cycle_starts), -1, dtype=int)
    in_known_block = cycle_block_indices >= 0
    cycle_block_ids[in_known_block] = block_ids[
        cycle_block_indices[in_known_block]
    ]

    block_frames = [
        _frames_in_interval(start, end)
        for start, end in zip(block_starts, block_ends)
    ]
    cycles_per_block = [
        np.where(cycle_block_indices == block_index)[0]
        for block_index in range(len(block_ids))
    ]

    block_statements = df_beh.get('block_start_statements', [])
    if len(block_statements) == len(block_ids):
        block_is_stim = np.asarray([
            bool(int(statement[3])) if len(statement) > 3 else False
            for statement in block_statements
        ])
    else:
        block_is_stim = np.asarray(
            [bool(len(cycles)) for cycles in cycles_per_block], dtype=bool
        )

    def _group_statements_by_block(statements):
        grouped = [[] for _ in block_ids]
        for statement in statements:
            if len(statement) < 2:
                continue
            try:
                timestamp = float(statement[1])
            except (TypeError, ValueError):
                continue
            matches = np.where(
                (block_starts <= timestamp) & (timestamp <= block_ends)
            )[0]
            if len(matches):
                grouped[matches[-1]].append(statement)
        return grouped

    pulse_commands = df_beh.get(
        'pulse_command_statements', df_beh.get('pulse_descriptions', [])
    )
    pulse_statements = df_beh.get('pulse_statements', [])
    pulse_command_statements_per_block = _group_statements_by_block(
        pulse_commands
    )
    pulse_statements_per_block = _group_statements_by_block(pulse_statements)
    pulse_statement_per_block = [
        statements[0] if statements else None
        for statements in pulse_command_statements_per_block
    ]
    pulse_parameters_per_block = [
        statement[2:] if statement is not None else None
        for statement in pulse_statement_per_block
    ]

    non_empty = [frames for frames in pulse_frames_per_cycle if len(frames)]
    stim_covered_frames = (
        np.unique(np.concatenate(non_empty)) if non_empty
        else np.array([], dtype=int)
    )
    non_empty = [frames for frames in frames_per_cycle if len(frames)]
    cycle_covered_frames = (
        np.unique(np.concatenate(non_empty)) if non_empty
        else np.array([], dtype=int)
    )
    stim_blocks = [
        frames for frames, is_stim in zip(block_frames, block_is_stim)
        if is_stim and len(frames)
    ]
    control_blocks = [
        frames for frames, is_stim in zip(block_frames, block_is_stim)
        if not is_stim and len(frames)
    ]

    cycles_with_stim = np.asarray([bool(train) for train in pulse_trains])
    valid_cycles = cycles_with_stim & in_known_block & (cycle_ends > cycle_starts)

    return {
        'paired_starts': paired_starts,
        'paired_ends': paired_ends,
        'pulse_trains': pulse_trains,
        'pulse_frames_per_cycle': pulse_frames_per_cycle,
        'frames_per_cycle': frames_per_cycle,
        'stim_covered_frames': stim_covered_frames,
        'cycle_covered_frames': cycle_covered_frames,
        'cycle_start_times': cycle_starts,
        'cycle_end_times': cycle_ends,
        'cycle_start_frames': np.asarray([
            frames[0] if len(frames) else -1 for frames in frames_per_cycle
        ]),
        'cycle_end_frames': np.asarray([
            frames[-1] if len(frames) else -1 for frames in frames_per_cycle
        ]),
        'cycles_with_stim': cycles_with_stim,
        'valid_cycles': valid_cycles,
        'cycle_block_indices': cycle_block_indices,
        'cycle_block_ids': cycle_block_ids,
        'unassigned_pulse_indices': np.where(~assigned_pulse)[0],
        'block_ids': block_ids,
        'block_is_stim': block_is_stim,
        'block_frames': block_frames,
        'cycles_per_block': cycles_per_block,
        'pulse_command_statements_per_block': (
            pulse_command_statements_per_block
        ),
        'pulse_statements_per_block': pulse_statements_per_block,
        'pulse_statement_per_block': pulse_statement_per_block,
        'pulse_parameters_per_block': pulse_parameters_per_block,
        'stim_block_pulse_statements': {
            int(block_id): statements
            for block_id, is_stim, statements in zip(
                block_ids, block_is_stim,
                pulse_command_statements_per_block,
            )
            if is_stim
        },
        'stim_block_frames': (
            np.unique(np.concatenate(stim_blocks)) if stim_blocks
            else np.array([], dtype=int)
        ),
        'control_block_frames': (
            np.unique(np.concatenate(control_blocks)) if control_blocks
            else np.array([], dtype=int)
        ),
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
rec = 'AD192-20260714-02'

anm, date, ss = rec.split('-')
data_base = (Path(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}")/
             (r"nonrigid_reg\suite2p\plane0"))
ch1_bin = np.memmap(data_base/'data.bin', mode='r', dtype='int16', shape=(10000, 512, 512))
# black frames
frame_mean = np.nanmean(ch1_bin, axis=(1,2))
black_frames = frame_mean<1
# dff_stim_masked[:, train_covered_frames] = np.nan
dff_stim_masked[:, shutter_masks] = np.nan

df_beh = pd.read_pickle(rf"Z:\Jingyu\raw_data\lc_stim_sensor\processed_data\{rec}\{rec}.pkl")

# pulse alignment
df_pulse = align_pulses_to_stim_cycles(df_beh)
stim_covered_frames = df_pulse['stim_covered_frames']
train_covered_frames = df_pulse['train_covered_frames']

# dff, is_active_soma, shutter_masks = process_F_trace(rec,
#                                                      active_soma_only=True,
#                                                      overwrite={"shutter_mask": True},
#                                                      )  

# filter for extreme values
# rsd_factor=3
# # thresh =pcf.dff_thresh(dff, hard_thresh=100, factor=5)
# kept_frames = ~shutter_masks
# dff_sd_kept = robust_filter_along_axis(
#     dff[:, kept_frames],
#     factor=rsd_factor,
# )
# dff_sd = np.full_like(dff, np.nan)
# dff_sd[:, kept_frames] = dff_sd_kept
# # dff_sd[abs(dff_sd)>thresh]=np.nan
# dff = dff_sd





# trial alignment
# bef, aft = 2, 4
# dff_aligned = align_trials(dff_stim_masked, 'run', df_beh, bef, aft)

#%% select stim and ctrl trials
stim_trials = df_pulse['trials_with_stim']
stim_valid_trials = df_pulse['valid_trials']&stim_trials
stim_plus_one_trials = np.zeros_like(stim_valid_trials)
stim_plus_one_trials[1:] = stim_valid_trials[:-1]
stim_plus_two_trials = np.zeros_like(stim_valid_trials)
stim_plus_two_trials[2:] = stim_valid_trials[:-2]
beh_valid_trials = seperate_valid_trial(df_beh, time_thresh=8000)
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
pyrUp_thresh = 2.5
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
