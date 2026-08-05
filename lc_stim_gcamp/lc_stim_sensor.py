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

from lc_stim_gcamp.calculate_dff import find_frame_cutoff, process_F_trace
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
    if len(block_starts) != len(block_ids):
        raise ValueError(
            'Every block start must have one block ID: '
            f'{len(block_starts)} starts, {len(block_ids)} IDs'
        )

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

    paired_block_ends = []
    block_end_index = 0
    inferred_block_end_indices = []
    for block_index, block_start in enumerate(block_starts):
        next_block_start = (
            block_starts[block_index + 1]
            if block_index + 1 < len(block_starts) else np.inf
        )
        while (block_end_index < len(block_ends)
               and block_ends[block_end_index] < block_start):
            block_end_index += 1
        if (block_end_index < len(block_ends)
                and block_ends[block_end_index] <= next_block_start):
            paired_block_ends.append(block_ends[block_end_index])
            block_end_index += 1
        else:
            inferred_end = (
                np.nextafter(next_block_start, -np.inf)
                if np.isfinite(next_block_start)
                else frame_times[-1]
            )
            if inferred_end < block_start:
                raise ValueError(
                    f'Block {block_ids[block_index]} starts after the last '
                    'imaging frame and has no matching end'
                )
            paired_block_ends.append(inferred_end)
            inferred_block_end_indices.append(block_index)
    block_ends = np.asarray(paired_block_ends, dtype=float)
    if inferred_block_end_indices:
        inferred_ids = block_ids[inferred_block_end_indices].tolist()
        print(
            f'Inferred missing block end(s) for block IDs {inferred_ids} '
            'from the next block start or final imaging frame.'
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

    train_onset_times = np.asarray([
        train[0][0] if train else np.nan for train in pulse_trains
    ], dtype=float)
    train_end_times = np.asarray([
        train[-1][1] if train else np.nan for train in pulse_trains
    ], dtype=float)
    train_onset_frames = np.asarray([
        max(0, np.searchsorted(frame_times, onset, side='right') - 1)
        if np.isfinite(onset) and frame_times[0] <= onset <= frame_times[-1]
        else -1
        for onset in train_onset_times
    ], dtype=int)

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
    pulse_statement_per_cycle = []
    for cycle_start, block_index in zip(cycle_starts, cycle_block_indices):
        if block_index < 0:
            pulse_statement_per_cycle.append(None)
            continue
        commands = pulse_command_statements_per_block[block_index]
        preceding = [
            statement for statement in commands
            if float(statement[1]) <= cycle_start
        ]
        pulse_statement_per_cycle.append(
            max(preceding, key=lambda statement: float(statement[1]))
            if preceding else None
        )

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
        'train_onset_times': train_onset_times,
        'train_end_times': train_end_times,
        'train_onset_frames': train_onset_frames,
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
        'pulse_statement_per_cycle': pulse_statement_per_cycle,
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


def _frame_batch_progress(n_frames, chunk_size, description, enabled=True):
    '''Yield frame batches while displaying a dependency-free progress bar.'''
    bar_width = 30
    try:
        for start in range(0, n_frames, chunk_size):
            stop = min(start + chunk_size, n_frames)
            yield start, stop
            if enabled:
                fraction = stop / n_frames
                filled = int(round(bar_width * fraction))
                bar = '#' * filled + '-' * (bar_width - filled)
                print(
                    f'\r{description}: [{bar}] {stop}/{n_frames} '
                    f'({fraction:6.1%})',
                    end='', flush=True,
                )
    finally:
        if enabled:
            print()


def _buffer_shutter_mask(shutter_masks, pre_frames=1, post_frames=1):
    '''Expand shutter masks by fixed pre-closing and post-opening buffers.'''
    raw_mask = np.asarray(shutter_masks, dtype=bool)
    buffered_mask = raw_mask.copy()
    for offset in range(1, int(pre_frames) + 1):
        buffered_mask[:-offset] |= raw_mask[offset:]
    for offset in range(1, int(post_frames) + 1):
        buffered_mask[offset:] |= raw_mask[:-offset]
    return buffered_mask


def load_frame_mean(
        data_base, black_threshold='auto', chunk_size=256, n_frames=None,
        backend='auto', show_progress=True, save_frame_mean=True,
        load_saved_frame_mean=True, shutter_pre_buffer_frames=1,
        shutter_post_buffer_frames=1):
    '''Load the Suite2p binary and calculate batched frame means.

    ``backend='auto'`` uses CuPy when it and a CUDA GPU are available, then
    falls back to NumPy. Use ``'cpu'`` or ``'gpu'`` to select explicitly.
    '''
    if backend not in {'auto', 'cpu', 'gpu'}:
        raise ValueError('backend must be auto, cpu, or gpu')
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    data_base = Path(data_base)
    ops = np.load(data_base / 'ops.npy', allow_pickle=True).item()
    ly, lx = int(ops['Ly']), int(ops['Lx'])
    data_path = data_base / 'data.bin'
    bytes_per_frame = np.dtype(np.int16).itemsize * ly * lx
    file_size = data_path.stat().st_size
    total_frames = file_size // bytes_per_frame

    if n_frames is None:
        n_frames = total_frames
    n_frames = int(n_frames)
    if n_frames <= 0 or n_frames > total_frames:
        raise ValueError(
            f'n_frames must be between 1 and {total_frames}, got {n_frames}'
        )

    if file_size % bytes_per_frame:
        raise ValueError(
            f'{data_path} size is not divisible by one {ly}x{lx} int16 frame'
        )

    ch1_bin = np.memmap(
        data_path, mode='r', dtype=np.int16,
        shape=(n_frames, ly, lx),
    )
    frame_mean_path = data_base / 'frame_mean.npy'
    shutter_masks_path = data_base / 'shutter_masks.npy'
    shutter_buffer_path = data_base / 'shutter_mask_buffer.npy'
    requested_buffer = np.asarray(
        [shutter_pre_buffer_frames, shutter_post_buffer_frames], dtype=int
    )
    if load_saved_frame_mean and frame_mean_path.exists():
        saved_frame_mean = np.load(frame_mean_path)
        if saved_frame_mean.shape == (n_frames,):
            frame_mean = np.asarray(saved_frame_mean, dtype=float)
            if shutter_masks_path.exists():
                shutter_masks = np.asarray(
                    np.load(shutter_masks_path), dtype=bool
                )
                if shutter_masks.shape != frame_mean.shape:
                    raise ValueError(
                        f'{shutter_masks_path} does not match frame_mean shape'
                    )
                saved_buffer = (
                    np.asarray(np.load(shutter_buffer_path), dtype=int)
                    if shutter_buffer_path.exists() else np.asarray([0, 1])
                )
                if not np.array_equal(saved_buffer, requested_buffer):
                    if np.any(requested_buffer < saved_buffer):
                        raise ValueError(
                            'Cannot shrink a saved shutter mask without raw '
                            'frame means; recalculate with '
                            'load_saved_frame_mean=False'
                        )
                    shutter_masks = _buffer_shutter_mask(
                        shutter_masks,
                        pre_frames=requested_buffer[0] - saved_buffer[0],
                        post_frames=requested_buffer[1] - saved_buffer[1],
                    )
                    np.save(shutter_masks_path, shutter_masks)
            else:
                shutter_masks = ~np.isfinite(frame_mean)
                shutter_masks = _buffer_shutter_mask(
                    shutter_masks, *requested_buffer
                )
                np.save(shutter_masks_path, shutter_masks)
            np.save(shutter_buffer_path, requested_buffer)
            frame_mean[shutter_masks] = np.nan
            if save_frame_mean:
                np.save(frame_mean_path, frame_mean)
            kept_frames = ~shutter_masks
            ops['data_bin_nframes'] = total_frames
            ops['loaded_nframes'] = n_frames
            ops['frame_mean_backend'] = 'saved'
            ops['frame_mean_path'] = str(frame_mean_path)
            ops['shutter_masks'] = shutter_masks
            ops['shutter_masks_path'] = str(shutter_masks_path)
            ops['shutter_off_frame_count'] = int(np.sum(shutter_masks))
            ops['kept_frame_count'] = int(np.sum(kept_frames))
            print(f'Loaded saved frame means from {frame_mean_path}')
            return ch1_bin, frame_mean, ops
        print(
            f'Ignoring {frame_mean_path}: expected {(n_frames,)}, '
            f'got {saved_frame_mean.shape}.'
        )

    frame_mean = np.empty(n_frames, dtype=float)
    cp = None
    if backend in {'auto', 'gpu'}:
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                cp = None
        except (ImportError, ModuleNotFoundError, RuntimeError):
            cp = None
        if backend == 'gpu' and cp is None:
            raise RuntimeError('CuPy and an available CUDA GPU are required')

    frame_mean_backend = 'cupy' if cp is not None else 'numpy'
    if cp is not None:
        try:
            for start, stop in _frame_batch_progress(
                    n_frames, chunk_size, 'Frame means (GPU)', show_progress):
                gpu_frames = cp.asarray(ch1_bin[start:stop])
                frame_mean[start:stop] = cp.asnumpy(
                    cp.mean(gpu_frames, axis=(1, 2), dtype=cp.float64)
                )
                del gpu_frames
        except (
                cp.cuda.memory.OutOfMemoryError,
                cp.cuda.runtime.CUDARuntimeError,
        ) as gpu_error:
            if backend == 'gpu':
                raise
            print(
                f'GPU frame-mean calculation failed ({gpu_error}); '
                'retrying on CPU.'
            )
            cp.get_default_memory_pool().free_all_blocks()
            cp = None
            frame_mean_backend = 'numpy'

    if cp is None:
        for start, stop in _frame_batch_progress(
                n_frames, chunk_size, 'Frame means (CPU)', show_progress):
            frame_mean[start:stop] = np.mean(
                ch1_bin[start:stop], axis=(1, 2), dtype=np.float64
            )
    print(
        f'Calculated {n_frames} frame means in batches of {chunk_size} '
        f'with {frame_mean_backend}.'
    )
    if isinstance(black_threshold, str):
        if black_threshold != 'auto':
            raise ValueError('black_threshold must be numeric or auto')
        applied_black_threshold = find_frame_cutoff(frame_mean, min_gap=0.5)
    else:
        applied_black_threshold = float(black_threshold)

    shutter_masks = frame_mean < applied_black_threshold
    shutter_masks = _buffer_shutter_mask(
        shutter_masks, *requested_buffer
    )
    kept_frames = ~shutter_masks
    frame_mean[shutter_masks] = np.nan
    ops['data_bin_nframes'] = total_frames
    ops['loaded_nframes'] = n_frames
    ops['frame_mean_backend'] = frame_mean_backend
    ops['frame_mean_chunk_size'] = chunk_size
    ops['black_frame_threshold'] = applied_black_threshold
    ops['black_frame_count'] = int(np.sum(shutter_masks))
    ops['shutter_frame_cutoff'] = applied_black_threshold
    ops['shutter_off_frame_count'] = int(np.sum(shutter_masks))
    ops['kept_frame_count'] = int(np.sum(kept_frames))
    ops['shutter_masks'] = shutter_masks
    ops['shutter_masks_path'] = str(shutter_masks_path)
    np.save(shutter_masks_path, shutter_masks)
    np.save(shutter_buffer_path, requested_buffer)
    if save_frame_mean:
        np.save(frame_mean_path, frame_mean)
        ops['frame_mean_path'] = str(frame_mean_path)
        print(f'Saved frame means to {frame_mean_path}')
    print(
        f'Shutter-frame cutoff={applied_black_threshold:.3f}; '
        f'masked {np.sum(shutter_masks)} of {n_frames} loaded frames.'
    )
    return ch1_bin, frame_mean, ops


def calculate_pulse_train_dff(
        frame_mean, df_beh, df_pulse, imaging_rate=None, baseline_s=1.0,
        frame_alignment='start', total_movie_frames=None,
        align_to='pulse', shutter_masks=None):
    '''Align frame means to pulse or shutter-cutoff onset and calculate dF/F.

    F0 is the mean of finite frames in the ``baseline_s`` interval before
    the selected onset. Set ``align_to='cutoff'`` to use the first shutter-off
    frame in each stimulation cycle instead of the first ``$PC`` pulse. The
    Post-stimulation duration is measured from each ``$PT,0`` end to the next
    ``$PT,1`` start in the same block. The within-block median fills the final
    cycle, which has no following cycle from which to measure the gap.
    '''
    frame_mean = np.asarray(frame_mean, dtype=float)
    frame_times = np.asarray(df_beh['frame_times'], dtype=float)
    if not len(frame_mean) or not len(frame_times):
        raise ValueError('No movie/behaviour frames are available for alignment')
    total_movie_frames_was_missing = total_movie_frames is None
    if total_movie_frames is None:
        total_movie_frames = len(frame_mean)
    total_movie_frames = int(total_movie_frames)
    if total_movie_frames < len(frame_mean):
        raise ValueError('total_movie_frames cannot be smaller than loaded frames')

    if frame_alignment == 'auto':
        frame_alignment = 'start'
    if frame_alignment not in {'start', 'end'}:
        raise ValueError('frame_alignment must be auto, start, or end')
    if (total_movie_frames_was_missing and frame_alignment == 'end'
            and len(frame_mean) < len(frame_times)):
        raise ValueError(
            'total_movie_frames is required when aligning a loaded prefix '
            'to the end of a longer FM series'
        )
    frame_index_offset = (
        total_movie_frames - len(frame_times) if frame_alignment == 'end' else 0
    )
    first_behaviour_frame = max(0, -frame_index_offset)
    last_behaviour_frame = min(
        len(frame_times), len(frame_mean) - frame_index_offset
    )
    n_aligned_frames = max(0, last_behaviour_frame - first_behaviour_frame)
    if n_aligned_frames == 0:
        raise ValueError('Loaded movie prefix does not overlap the FM timestamps')

    if imaging_rate is None:
        positive_steps = np.diff(frame_times[:n_aligned_frames])
        positive_steps = positive_steps[positive_steps > 0]
        if not len(positive_steps):
            raise ValueError('Cannot infer imaging rate from frame_times')
        imaging_rate = 1000 / np.median(positive_steps)
    imaging_rate = float(imaging_rate)
    pre_frames = int(round(baseline_s * imaging_rate))

    if align_to not in {'pulse', 'cutoff'}:
        raise ValueError('align_to must be pulse or cutoff')
    if align_to == 'pulse':
        alignment_onset_frames = np.asarray(
            df_pulse['train_onset_frames'], dtype=int
        )
    else:
        if shutter_masks is None:
            raise ValueError(
                'shutter_masks is required for cutoff alignment'
            )
        shutter_masks = np.asarray(shutter_masks, dtype=bool)
        if len(shutter_masks) != len(frame_mean):
            raise ValueError(
                'shutter_masks and frame_mean must contain the same number '
                'of loaded frames'
            )
        cutoff_onsets = np.flatnonzero(
            shutter_masks & np.r_[True, ~shutter_masks[:-1]]
        )
        alignment_onset_frames = np.full(
            len(df_pulse['cycle_start_frames']), -1, dtype=int
        )
        for cycle_index, (cycle_start, cycle_end) in enumerate(zip(
                df_pulse['cycle_start_frames'],
                df_pulse['cycle_end_frames'])):
            cycle_cutoffs = cutoff_onsets[
                (cutoff_onsets >= int(cycle_start) + frame_index_offset)
                & (cutoff_onsets <= int(cycle_end) + frame_index_offset)
            ]
            if len(cycle_cutoffs):
                alignment_onset_frames[cycle_index] = cycle_cutoffs[0]

    statements = df_pulse['pulse_statement_per_cycle']
    if len(statements) != len(df_pulse['cycle_start_frames']):
        raise ValueError('Every pulse cycle must have one mapped $PP statement')

    programmed_duration_ms = np.full(len(statements), np.nan)
    condition_keys = []
    for cycle_index, statement in enumerate(statements):
        if statement is None or len(statement) <= 3:
            condition_keys.append(None)
            continue
        try:
            post_duration_us = float(statement[3])
            programmed_duration_ms[cycle_index] = post_duration_us / 1000
        except (TypeError, ValueError):
            condition_keys.append(None)
            continue
        condition_keys.append(tuple(statement[2:]))

    finite_programmed = programmed_duration_ms[
        np.isfinite(programmed_duration_ms)
    ]
    if not len(finite_programmed) or np.any(finite_programmed <= 0):
        raise ValueError(
            'All valid $PP statement[3] durations must be positive '
            'microsecond values'
        )
    train_duration_ms = (
        np.asarray(df_pulse['train_end_times'], dtype=float)
        - np.asarray(df_pulse['train_onset_times'], dtype=float)
    )
    cycle_starts_ms = np.asarray(df_pulse['cycle_start_times'], dtype=float)
    cycle_ends_ms = np.asarray(df_pulse['cycle_end_times'], dtype=float)
    cycle_blocks = np.asarray(df_pulse['cycle_block_indices'], dtype=int)
    post_duration_ms = np.full(len(statements), np.nan)
    for cycle_index in range(len(statements) - 1):
        if (cycle_blocks[cycle_index] >= 0
                and cycle_blocks[cycle_index + 1] == cycle_blocks[cycle_index]):
            measured_post = (
                cycle_starts_ms[cycle_index + 1]
                - cycle_ends_ms[cycle_index]
            )
            if measured_post > 0:
                post_duration_ms[cycle_index] = measured_post

    finite_measured_post = post_duration_ms[np.isfinite(post_duration_ms)]
    if not len(finite_measured_post):
        raise ValueError(
            'Could not measure a post-stimulation interval between cycles'
        )
    global_post_median = float(np.median(finite_measured_post))
    for block_index in np.unique(cycle_blocks[cycle_blocks >= 0]):
        in_block = cycle_blocks == block_index
        block_post = post_duration_ms[in_block]
        finite_block_post = block_post[np.isfinite(block_post)]
        block_post_median = (
            float(np.median(finite_block_post))
            if len(finite_block_post) else global_post_median
        )
        missing_in_block = in_block & ~np.isfinite(post_duration_ms)
        post_duration_ms[missing_in_block] = block_post_median
    trace_duration_ms = train_duration_ms + post_duration_ms
    finite_trace_duration = trace_duration_ms[np.isfinite(trace_duration_ms)]
    max_post_frames = int(np.ceil(
        np.max(finite_trace_duration) / 1000 * imaging_rate
    ))
    n_timepoints = pre_frames + max_post_frames + 1
    dff_traces = np.full((len(statements), n_timepoints), np.nan)
    f0 = np.full(len(statements), np.nan)
    valid_cycles = np.zeros(len(statements), dtype=bool)

    for cycle_index, (onset_frame, trace_ms) in enumerate(zip(
            alignment_onset_frames, trace_duration_ms)):
        onset_frame = int(onset_frame)
        if align_to == 'pulse':
            onset_frame += frame_index_offset
        if not np.isfinite(trace_ms) or onset_frame < pre_frames:
            continue
        if onset_frame >= len(frame_mean):
            continue

        baseline = frame_mean[onset_frame - pre_frames:onset_frame]
        finite_baseline = baseline[np.isfinite(baseline)]
        if not len(finite_baseline):
            continue
        cycle_f0 = float(np.mean(finite_baseline))
        if not np.isfinite(cycle_f0) or cycle_f0 == 0:
            continue

        post_frames = int(np.ceil(trace_ms / 1000 * imaging_rate))
        source_start = onset_frame - pre_frames
        source_stop = onset_frame + post_frames + 1
        if source_stop > len(frame_mean):
            continue
        trace = frame_mean[source_start:source_stop]
        dff_traces[cycle_index, :len(trace)] = (trace - cycle_f0) / cycle_f0
        f0[cycle_index] = cycle_f0
        valid_cycles[cycle_index] = True

    excluded_timepoints_by_condition = {}
    ordered_conditions = []
    for cycle_index, key in enumerate(condition_keys):
        if valid_cycles[cycle_index] and key is not None and key not in ordered_conditions:
            ordered_conditions.append(key)
    for key in ordered_conditions:
        cycle_indices = np.asarray([
            cycle_index for cycle_index, cycle_key in enumerate(condition_keys)
            if valid_cycles[cycle_index] and cycle_key == key
        ], dtype=int)
        excluded = np.any(~np.isfinite(dff_traces[cycle_indices]), axis=0)
        excluded_timepoints_by_condition[key] = excluded
        excluded_columns = np.where(excluded)[0]
        if len(excluded_columns):
            dff_traces[np.ix_(cycle_indices, excluded_columns)] = np.nan

    time_s = (np.arange(n_timepoints) - pre_frames) / imaging_rate
    return {
        'dff_traces': dff_traces,
        'time_s': time_s,
        'f0': f0,
        'valid_cycles': valid_cycles,
        'post_duration_ms': post_duration_ms,
        'programmed_duration_ms': programmed_duration_ms,
        'train_duration_ms': train_duration_ms,
        'trace_duration_ms': trace_duration_ms,
        'condition_keys': condition_keys,
        'excluded_timepoints_by_condition': excluded_timepoints_by_condition,
        'imaging_rate': imaging_rate,
        'n_aligned_frames': n_aligned_frames,
        'loaded_movie_frames': len(frame_mean),
        'total_movie_frames': total_movie_frames,
        'first_behaviour_frame': first_behaviour_frame,
        'last_behaviour_frame': last_behaviour_frame,
        'frame_alignment': frame_alignment,
        'frame_index_offset': frame_index_offset,
        'align_to': align_to,
        'alignment_onset_frames': alignment_onset_frames,
    }

def plot_pulse_train_dff_by_condition(dff_result, df_pulse, ax=None):
    '''Plot pulse-train mean dF/F +/- SEM for each stimulation condition.'''
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    else:
        fig = ax.figure

    traces = np.asarray(dff_result['dff_traces'], dtype=float)
    time_s = np.asarray(dff_result['time_s'], dtype=float)
    valid = (
        np.asarray(dff_result['valid_cycles'], dtype=bool)
        & np.asarray(df_pulse['valid_cycles'], dtype=bool)
    )
    condition_keys = dff_result['condition_keys']

    ordered_conditions = []
    for cycle_index, key in enumerate(condition_keys):
        if valid[cycle_index] and key is not None and key not in ordered_conditions:
            ordered_conditions.append(key)

    condition_summary = {}
    for key in ordered_conditions:
        cycle_indices = np.asarray([
            cycle_index for cycle_index, cycle_key in enumerate(condition_keys)
            if valid[cycle_index] and cycle_key == key
        ], dtype=int)
        condition_traces = traces[cycle_indices]
        excluded_timepoints = dff_result[
            'excluded_timepoints_by_condition'
        ].get(key, np.zeros(condition_traces.shape[1], dtype=bool))
        condition_traces = condition_traces.copy()
        condition_traces[:, excluded_timepoints] = np.nan
        finite_count = np.sum(np.isfinite(condition_traces), axis=0)
        mean_trace = np.divide(
            np.nansum(condition_traces, axis=0), finite_count,
            out=np.full(condition_traces.shape[1], np.nan),
            where=finite_count > 0,
        )
        squared_error = np.nansum(
            (condition_traces - mean_trace[None, :]) ** 2, axis=0
        )
        sem_trace = np.full_like(mean_trace, np.nan)
        enough_data = finite_count > 1
        sem_trace[enough_data] = (
            np.sqrt(squared_error[enough_data] / (finite_count[enough_data] - 1))
            / np.sqrt(finite_count[enough_data])
        )

        statement = df_pulse['pulse_statement_per_cycle'][cycle_indices[0]]
        block_ids = np.unique(
            np.asarray(df_pulse['cycle_block_ids'])[cycle_indices]
        )
        block_id_text = ','.join(map(str, block_ids))
        try:
            frequency_hz = 1_000_000 / float(statement[4])
            pulse_count = int(float(statement[5]))
            condition_label = (
                f'{frequency_hz:g} Hz x {pulse_count} pulses '
                f'(block {block_id_text}, n={len(cycle_indices)})'
            )
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            condition_label = (
                f'block {block_id_text} '
                f'(n={len(cycle_indices)})'
            )

        line, = ax.plot(time_s, mean_trace, label=condition_label)
        ax.fill_between(
            time_s, mean_trace - sem_trace, mean_trace + sem_trace,
            color=line.get_color(), alpha=0.2, linewidth=0,
        )
        condition_summary[key] = {
            'cycle_indices': cycle_indices,
            'mean': mean_trace,
            'sem': sem_trace,
            'n_per_timepoint': finite_count,
            'excluded_timepoints': excluded_timepoints,
            'label': condition_label,
        }

    onset_label = (
        'shutter cutoff onset'
        if dff_result.get('align_to') == 'cutoff'
        else 'pulse-train onset'
    )
    ax.axvline(0, color='black', linestyle='--', linewidth=1,
               label=onset_label)
    ax.axhline(0, color='0.7', linewidth=0.8)
    ax.set(
        xlabel=f'Time from {onset_label} (s)',
        ylabel='dF/F',
        title='Sensor response by stimulation condition',
    )
    ax.legend(frameon=False, fontsize=8)
    return fig, ax, condition_summary


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
# Load all movie frames and collapse each frame to the channel-1 sensor mean.
ch1_bin, frame_mean, ops = load_frame_mean(data_base, black_threshold='auto',
                                           # n_frames=10000
                                           )

df_beh = pd.read_pickle(rf"Z:\Jingyu\raw_data\lc_stim_sensor\processed_data\{rec}\{rec}.pkl")
#%%
# Align pulse cycles, calculate train-wise dF/F, and group by $PP condition.
df_pulse = align_pulses_to_stim_cycles(df_beh)
align_to = 'pulse'  # Change to 'cutoff' for shutter-off onset alignment.
# align_to = 'cutoff'
loaded_movie_frames = len(frame_mean)
total_movie_frames = int(ops['data_bin_nframes'])
fm_frame_count = len(df_beh['frame_times'])
max_allowed_frame_index_offset = 150
frame_index_offset = total_movie_frames - fm_frame_count
if not 0 <= frame_index_offset <= max_allowed_frame_index_offset:
    raise ValueError(
        'Movie/FM frame index offset must be between 0 and '
        f'{max_allowed_frame_index_offset}, got {frame_index_offset} '
        f'(movie={total_movie_frames}, FM timestamps={fm_frame_count})'
    )
frame_alignment = 'start'
if total_movie_frames != fm_frame_count:
    print(
        f'Frame-count mismatch: movie={total_movie_frames}, '
        f'FM timestamps={fm_frame_count}; using their shared recording start.'
    )
if loaded_movie_frames < total_movie_frames:
    print(
        f'Using first {loaded_movie_frames} of {total_movie_frames} movie frames.'
    )

baseline_s = 2
df_cycle_dff = calculate_pulse_train_dff(
    frame_mean,
    df_beh,
    df_pulse,
    imaging_rate=ops.get('fs'),
    baseline_s=baseline_s,
    frame_alignment=frame_alignment,
    total_movie_frames=total_movie_frames,
    align_to=align_to,
    shutter_masks=ops['shutter_masks'],
)
alignment_mode = df_cycle_dff['frame_alignment']
alignment_offset = df_cycle_dff['frame_index_offset']
print(f'Frame alignment: {alignment_mode}, index offset={alignment_offset}')

#%%
fig, ax, condition_summary = plot_pulse_train_dff_by_condition(
    df_cycle_dff, df_pulse
)
# ax.set(ylim=(-0.05, 0.05))
ax.set_xlim(-baseline_s)
ax.set_title(f'{rec}: sensor response by stimulation condition')
plt.show()
