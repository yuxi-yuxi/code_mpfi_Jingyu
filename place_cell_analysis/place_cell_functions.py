# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 17:42:21 2026

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from tqdm import tqdm
from common.utils_basic import nearest_mapping, normalize
from common.robust_sd_filter import robust_filter_along_axis

# Import shared trial correlation utilities
from place_cell_analysis.utils_trial_correlation import (
    normalize_per_lap_profile,
    pearson_corr_rows_gpu,
    calculate_trial_correlations_gpu,
    calculate_all_trial_correlations_gpu,
)
    
def spatial_binning(running_calcium_map_img,
                    running_distance_map_img,
                    track_length = 180,
                    bin_size = 4,
                    save_lap_profile=True,
                    ):

    n_bins = int(track_length / bin_size)  # 45 bins

    # Assign each running frame to a spatial bin
    bin_edges = np.linspace(0, track_length, n_bins + 1)
    bin_indices = np.digitize(running_distance_map_img, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # handle edge cases

    # Compute occupancy (frame count per bin)
    occupancy_raw = np.bincount(bin_indices, minlength=n_bins).astype(float)  # frames per bin

    # Compute mean Ca2+ dF/F per bin for each neuron
    n_cells = running_calcium_map_img.shape[0]
    event_count_raw = np.zeros((n_cells, n_bins))
    event_rate_raw = np.zeros((n_cells, n_bins))  # for spatial information calculation

    for i_cell in range(n_cells):
        cell_trace = running_calcium_map_img[i_cell]
        # Sum of dF/F values in each bin
        dff_sum = np.bincount(bin_indices, weights=cell_trace, minlength=n_bins)
        # Mean dF/F per bin (sum / occupancy)
        # event_count_raw[i_cell] = np.divide(dff_sum, occupancy_raw,
        #                                     out=np.zeros_like(dff_sum),
        #                                     where=occupancy_raw > 0)
        event_count_raw[i_cell] = dff_sum

    # Compute unsmoothed event rate (for spatial information)
    # Avoid division by zero
    occupancy_safe = occupancy_raw.copy()
    occupancy_safe[occupancy_safe == 0] = np.nan
    event_rate_raw = event_count_raw / occupancy_safe  # events per second per bin

    # Compute per-lap profiles
    # Detect lap boundaries: where distance decreases significantly (lap reset)
    distance_diff = np.diff(running_distance_map_img)
    lap_start_indices = np.where(distance_diff < -track_length / 2)[0] + 1  # frames where new lap starts
    lap_start_indices = np.concatenate([[0], lap_start_indices, [len(running_distance_map_img)]])

    n_laps = len(lap_start_indices) - 1

    if save_lap_profile:
        # Per-lap event rate: (n_cells, n_laps, n_bins)
        per_lap_profile = np.full((n_cells, n_laps, n_bins), np.nan)
        per_lap_occupancy = np.zeros((n_laps, n_bins))

        for i_lap in range(n_laps):
            lap_start = lap_start_indices[i_lap]
            lap_end = lap_start_indices[i_lap + 1]

            lap_bin_indices = bin_indices[lap_start:lap_end]
            lap_calcium = running_calcium_map_img[:, lap_start:lap_end]  # (n_cells, n_lap_frames)

            # Occupancy for this lap
            lap_occupancy = np.bincount(lap_bin_indices, minlength=n_bins).astype(float)
            per_lap_occupancy[i_lap] = lap_occupancy
            lap_occupancy_safe = lap_occupancy.copy()
            lap_occupancy_safe[lap_occupancy_safe == 0] = np.nan

            # Vectorized: sum dF/F per bin for all cells at once
            # Create a sparse-like accumulation using advanced indexing
            dff_sum_lap = np.zeros((n_cells, n_bins))
            np.add.at(dff_sum_lap, (slice(None), lap_bin_indices), lap_calcium)
            per_lap_profile[:, i_lap, :] = dff_sum_lap / lap_occupancy_safe
    else:
        per_lap_profile = []
        per_lap_occupancy = []

    return event_count_raw, event_rate_raw, occupancy_raw, per_lap_profile, per_lap_occupancy


def map_laps_to_trials(running_distance_map_img, running_time_map_img_abs,
                       beh, track_length=180):
    """Map each running lap (from spatial_binning) to a behavioural trial index.

    Uses absolute timestamps of running frames to match each running lap
    to the behavioural trial whose run_onset is closest.

    Parameters
    ----------
    running_distance_map_img : 1-D array
        Distance trace for running frames only (output of align_run_frame_calcium).
    running_time_map_img_abs : 1-D array
        Absolute timestamp (ms) trace for running frames only (4th return
        value of align_run_frame_calcium).
    beh : dict
        Behavioural data; must contain 'run_onsets' (one per trial, NaN
        if no onset).
    track_length : float
        Track length in cm (must match spatial_binning).

    Returns
    -------
    lap_trial_idx : 1-D int array, length n_laps
        For each running lap, the index of the behavioural trial it belongs to.
        -1 if no matching trial was found.
    """
    # Detect lap boundaries (same logic as spatial_binning)
    distance_diff = np.diff(running_distance_map_img)
    lap_start_frames = np.where(distance_diff < -track_length / 2)[0] + 1
    lap_start_frames = np.concatenate([[0], lap_start_frames])
    n_laps = len(lap_start_frames)

    # Absolute timestamp at the start of each running lap
    lap_start_times = running_time_map_img_abs[lap_start_frames]

    # Behavioural trial onsets (absolute ms)
    run_onsets = np.array(beh['run_onsets'], dtype=float)
    valid_onsets = ~np.isnan(run_onsets)

    lap_trial_idx = np.full(n_laps, -1, dtype=int)

    for i_lap, t_lap in enumerate(lap_start_times):
        # Find trial whose run_onset is closest to this lap start
        diffs = np.abs(t_lap - run_onsets)
        diffs[~valid_onsets] = np.inf
        best = np.argmin(diffs)
        if diffs[best] < np.inf:
            lap_trial_idx[i_lap] = best

    return lap_trial_idx


def compute_valid_event_rate(per_lap_profile, per_lap_occupancy, valid_trials_mask=None):
    """
    Recompute per-cell event rate and occupancy from only valid laps.

    Invalid laps (with any NaN bin — typically from dFF thresholding or
    unvisited bins) are excluded per cell.

    Parameters:
    -----------
    per_lap_profile : ndarray (n_cells, n_laps, n_bins)
        Per-lap mean dF/F per bin
    per_lap_occupancy : ndarray (n_laps, n_bins)
        Frame count per bin per lap
    valid_trials_mask : ndarray (n_cells, n_laps) or None
        Boolean mask; True = use that lap for that cell. If None, defaults
        to excluding any lap with a NaN bin.

    Returns:
    --------
    event_count : ndarray (n_cells, n_bins)
    event_rate : ndarray (n_cells, n_bins)
    occupancy : ndarray (n_cells, n_bins)
    """
    per_lap_profile = np.asarray(per_lap_profile)
    per_lap_occupancy = np.asarray(per_lap_occupancy)

    if valid_trials_mask is None:
        valid_trials_mask = ~np.any(np.isnan(per_lap_profile), axis=-1)

    # per_lap_profile is mean dF/F per bin; multiply back by occupancy
    # to get sum of dF/F per bin per lap
    per_lap_count = per_lap_profile * per_lap_occupancy[None, :, :]  # (n_cells, n_laps, n_bins)

    mask_3d = valid_trials_mask[:, :, None]  # (n_cells, n_laps, 1)

    # Zero out invalid laps before summing so NaN bins don't contaminate
    event_count = np.nansum(np.where(mask_3d, per_lap_count, 0.0), axis=1)
    occupancy = np.sum(np.where(mask_3d, per_lap_occupancy[None, :, :], 0.0), axis=1)

    occupancy_safe = np.where(occupancy > 0, occupancy, np.nan)
    event_rate = event_count / occupancy_safe

    return event_count, event_rate, occupancy

## Compute spatial information (bits per event)
# Spatial Information = sum_i [ p_i * (r_i / r_bar) * log2(r_i / r_bar) ]
# where r_i = event rate in bin i, p_i = probability of being in bin i, r_bar = mean event rate

def calculate_spatial_info(event_rate, occupancy, gpu=False):
    """
    Vectorized calculation of spatial information for all ROIs.

    Parameters:
    -----------
    event_rate : ndarray (n_cells, n_bins) or (n_bins,)
        Unsmoothed event rate (mean dF/F) per spatial bin for each cell
    occupancy : ndarray (n_bins,) or (n_cells, n_bins)
        Frame count per spatial bin. If 2D, per-cell occupancies are used
        (e.g. after excluding invalid trials per cell).
    gpu : bool
        Whether to use GPU acceleration with CuPy

    Returns:
    --------
    spatial_info : ndarray (n_cells,) or scalar
        Spatial information in bits per event for each cell
    """
    if gpu:
        import cupy as cp
        xp = cp
        event_rate = cp.asarray(event_rate)
        occupancy = cp.asarray(occupancy)
    else:
        xp = np

    # Handle 1D input (single ROI)
    squeeze_output = False
    if event_rate.ndim == 1:
        event_rate = event_rate[xp.newaxis, :]  # (1, n_bins)
        squeeze_output = True

    # Support shared (n_bins,) or per-cell (n_cells, n_bins) occupancy
    if occupancy.ndim == 1:
        total_frames = xp.sum(occupancy)  # scalar
        p_i = occupancy / total_frames  # (n_bins,)
    else:
        total_frames = xp.sum(occupancy, axis=1, keepdims=True)  # (n_cells, 1)
        total_frames_safe = xp.where(total_frames == 0, xp.nan, total_frames)
        p_i = occupancy / total_frames_safe  # (n_cells, n_bins)

    # Mean event rate for each cell: r_bar = sum(r_i * p_i)
    r_bar = xp.nansum(event_rate * p_i, axis=1, keepdims=True)  # (n_cells, 1)

    # Avoid division by zero
    r_bar_safe = xp.where(r_bar == 0, xp.nan, r_bar)

    # r_ratio = r_i / r_bar for each cell and bin
    r_ratio = event_rate / r_bar_safe  # (n_cells, n_bins)

    # Mask for valid bins: occupancy > 0 and r_i > 0
    valid_mask = (occupancy > 0) & (event_rate > 0)  # broadcasts for 1D occupancy

    # Compute: p_i * (r_i / r_bar) * log2(r_i / r_bar)
    safe_ratio = xp.where(r_ratio > 0, r_ratio, 1)
    log_term = xp.log2(safe_ratio)
    si_per_bin = p_i * r_ratio * log_term  # (n_cells, n_bins)

    # Zero out invalid bins and sum
    si_per_bin = xp.where(valid_mask, si_per_bin, 0)
    spatial_info = xp.nansum(si_per_bin, axis=1)  # (n_cells,)

    # Set to NaN for cells with r_bar == 0
    spatial_info = xp.where(r_bar.flatten() == 0, xp.nan, spatial_info)

    # Squeeze output if input was 1D
    if squeeze_output:
        spatial_info = spatial_info[0]

    # Convert back to numpy if using GPU
    if gpu:
        spatial_info = cp.asnumpy(spatial_info)

    return spatial_info

def calculate_spatial_info_shuffle(running_calcium_map_img,
                                   running_distance_map_img,
                                   occupancy, shuff_times=1000,
                                   min_shift=450,
                                   gpu=True,
                                   track_length=180,
                                   bin_size=4,
                                   roi_batch_size=16,
                                   random_seed=None,
                                   show_progress=True):
    """Compute circular-shuffle SI with FFT-equivalent spatial-bin sums.

    This produces the same bin sums as explicitly applying ``np.roll`` for
    every ROI/shuffle, but calculates all possible circular shifts by FFT and
    samples the requested shifts. Inputs must share one valid-frame mask.
    """
    calcium = np.asarray(running_calcium_map_img)
    distance = np.asarray(running_distance_map_img)
    occupancy = np.asarray(occupancy, dtype=float)
    if calcium.ndim != 2:
        raise ValueError("running_calcium_map_img must have shape (ROIs, frames)")
    n_rois, n_frames = calcium.shape
    if distance.ndim != 1 or len(distance) != n_frames:
        raise ValueError("running_distance_map_img length must match calcium frames")
    n_bins = int(track_length / bin_size)
    if occupancy.shape != (n_bins,):
        raise ValueError(f"occupancy must have shape ({n_bins},)")
    if np.any(~np.isfinite(calcium)):
        raise ValueError("Shuffle calcium input must be finite after valid-lap filtering")
    if n_frames <= 2 * min_shift:
        raise ValueError(
            f"Circular shuffle needs more than {2 * min_shift} frames for "
            f"min_shift={min_shift}, but received {n_frames}. Filter each ROI "
            "by its own valid laps rather than intersecting all ROIs."
        )
    if n_rois == 0:
        return np.empty((0, shuff_times), dtype=float)

    rng = np.random.default_rng(random_seed)
    shift_amounts = rng.integers(
        min_shift, n_frames - min_shift, size=(shuff_times, n_rois)
    )
    bin_edges = np.linspace(0, track_length, n_bins + 1)
    frame_bins = np.clip(np.digitize(distance, bin_edges) - 1, 0, n_bins - 1)
    bin_masks = (frame_bins[None, :] == np.arange(n_bins)[:, None]).astype(float)
    shuffled_si = np.full((n_rois, shuff_times), np.nan, dtype=float)

    if gpu:
        import cupy as cp
        xp = cp
    else:
        xp = np
    bin_masks_xp = xp.asarray(bin_masks)
    mask_fft = xp.fft.rfft(bin_masks_xp, axis=1)
    batches = range(0, n_rois, roi_batch_size)
    batches = tqdm(
        batches, total=int(np.ceil(n_rois / roi_batch_size)),
        desc="FFT shuffle ROI batches", disable=not show_progress,
    )
    for roi_start in batches:
        roi_stop = min(roi_start + roi_batch_size, n_rois)
        calcium_batch = xp.asarray(calcium[roi_start:roi_stop])
        calcium_fft = xp.fft.rfft(calcium_batch, axis=1)
        correlations = xp.fft.irfft(
            xp.conj(calcium_fft[:, None, :]) * mask_fft[None, :, :],
            n=n_frames, axis=2,
        )
        shifts = xp.asarray(shift_amounts[:, roi_start:roi_stop])
        roi_selector = xp.arange(roi_stop - roi_start)[None, :, None]
        bin_selector = xp.arange(n_bins)[None, None, :]
        event_count = correlations[
            roi_selector, bin_selector, shifts[:, :, None]
        ]
        occupancy_xp = xp.asarray(occupancy)
        occupancy_safe = xp.where(occupancy_xp > 0, occupancy_xp, xp.nan)
        event_rate = event_count / occupancy_safe[None, None, :]
        flat_rate = event_rate.reshape(-1, n_bins)
        si_batch = calculate_spatial_info(flat_rate, occupancy, gpu=gpu)
        shuffled_si[roi_start:roi_stop] = np.asarray(si_batch).reshape(
            shuff_times, roi_stop - roi_start
        ).T
        del correlations, event_count, event_rate, flat_rate
        if gpu:
            cp.get_default_memory_pool().free_all_blocks()
    return shuffled_si


def calculate_spatial_info_shuffle_by_valid_laps(
        running_calcium_map_img, running_distance_map_img,
        valid_trials_mask, candidate_indices, min_valid_laps=80,
        shuff_times=1000, min_shift=450, gpu=True,
        track_length=180, bin_size=4, roi_batch_size=16,
        random_seed=None):
    """Shuffle candidates in exact groups sharing the same valid-lap mask."""
    calcium = np.asarray(running_calcium_map_img)
    distance = np.asarray(running_distance_map_img)
    valid_trials_mask = np.asarray(valid_trials_mask, dtype=bool)
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    enough = valid_trials_mask[candidate_indices].sum(axis=1) >= min_valid_laps
    eligible_indices = candidate_indices[enough]
    eligible_valid = valid_trials_mask[eligible_indices]
    if not len(eligible_indices):
        return eligible_indices, np.empty((0, shuff_times), dtype=float)

    distance_diff = np.diff(distance)
    lap_starts = np.where(distance_diff < -track_length / 2)[0] + 1
    lap_starts = np.concatenate([[0], lap_starts, [len(distance)]])
    n_detected_laps = len(lap_starts) - 1
    if n_detected_laps != valid_trials_mask.shape[1]:
        raise ValueError(
            f"Detected {n_detected_laps} running laps but valid_trials_mask "
            f"contains {valid_trials_mask.shape[1]} laps"
        )

    groups = {}
    for position, lap_mask in enumerate(eligible_valid):
        groups.setdefault(np.packbits(lap_mask).tobytes(), []).append(position)
    group_positions = sorted(groups.values(), key=len, reverse=True)
    print(
        f"  {len(eligible_indices)} candidates with >= {min_valid_laps} valid laps; "
        f"processing {len(group_positions)} exact valid-lap mask groups"
    )
    shuffled = np.full((len(eligible_indices), shuff_times), np.nan, dtype=float)
    group_iterator = tqdm(group_positions, desc="valid-lap mask groups")
    for group_number, positions in enumerate(group_iterator):
        lap_mask = eligible_valid[positions[0]]
        valid_frames = np.zeros(len(distance), dtype=bool)
        for lap_index, is_valid in enumerate(lap_mask):
            if is_valid:
                valid_frames[lap_starts[lap_index]:lap_starts[lap_index + 1]] = True
        roi_indices = eligible_indices[positions]
        frame_count = int(valid_frames.sum())
        group_iterator.set_postfix(cells=len(positions), frames=frame_count)
        calcium_group = calcium[roi_indices][:, valid_frames]
        distance_group = distance[valid_frames]
        n_bins = int(track_length / bin_size)
        bin_edges = np.linspace(0, track_length, n_bins + 1)
        frame_bins = np.clip(
            np.digitize(distance_group, bin_edges) - 1, 0, n_bins - 1
        )
        occupancy_group = np.bincount(
            frame_bins, minlength=n_bins
        ).astype(float)
        group_seed = (None if random_seed is None else random_seed + group_number)
        shuffled[positions] = calculate_spatial_info_shuffle(
            calcium_group, distance_group, occupancy_group,
            shuff_times=shuff_times, min_shift=min_shift, gpu=gpu,
            track_length=track_length, bin_size=bin_size,
            roi_batch_size=roi_batch_size, random_seed=group_seed,
            show_progress=False,
        )
    return eligible_indices, shuffled

def calculate_lap_stability(per_lap_profile, method='odd_even'):
    """
    Calculate place cell stability from per-lap spatial profiles.

    Stability is quantified as the correlation between split-half averages,
    measuring how consistent the spatial tuning is across laps.

    Parameters:
    -----------
    per_lap_profile : ndarray (n_laps, n_bins)
        Event rate per spatial bin for each lap
    method : str
        Method for calculating stability:
        - 'odd_even': Correlation between odd and even lap averages (default, fast)
        - 'first_second_half': Correlation between first and second half averages

    Returns:
    --------
    stability : float
        Stability metric (correlation coefficient, range [-1, 1])
    """
    if per_lap_profile is None or len(per_lap_profile) == 0:
        return np.nan

    # Convert to numpy array if needed (from list of lists after parquet loading)
    if isinstance(per_lap_profile, list):
        per_lap_profile = np.array(per_lap_profile)

    # Handle edge cases: 1D array or wrong shape
    if per_lap_profile.ndim != 2:
        return np.nan

    n_laps, n_bins = per_lap_profile.shape

    if n_laps < 2 or n_bins < 3:
        return np.nan

    # Exclude laps with any NaN bin (invalid trials — e.g. dFF thresholded out)
    valid_laps = ~np.any(np.isnan(per_lap_profile), axis=1)
    per_lap_profile = per_lap_profile[valid_laps]
    n_laps = per_lap_profile.shape[0]

    if n_laps < 2:
        return np.nan

    if method == 'odd_even':
        # Correlation between odd and even lap averages
        odd_laps = per_lap_profile[::2]
        even_laps = per_lap_profile[1::2]

        if len(odd_laps) == 0 or len(even_laps) == 0:
            return np.nan

        odd_mean = np.nanmean(odd_laps, axis=0)
        even_mean = np.nanmean(even_laps, axis=0)

        valid_mask = ~np.isnan(odd_mean) & ~np.isnan(even_mean)
        if np.sum(valid_mask) < 3:
            return np.nan

        return np.corrcoef(odd_mean[valid_mask], even_mean[valid_mask])[0, 1]

    elif method == 'first_second_half':
        # Correlation between first and second half averages
        half = n_laps // 2
        if half < 1:
            return np.nan

        first_half = np.nanmean(per_lap_profile[:half], axis=0)
        second_half = np.nanmean(per_lap_profile[half:], axis=0)

        valid_mask = ~np.isnan(first_half) & ~np.isnan(second_half)
        if np.sum(valid_mask) < 3:
            return np.nan

        return np.corrcoef(first_half[valid_mask], second_half[valid_mask])[0, 1]

    else:
        raise ValueError(f"Unknown method: {method}")


def _bool_mask_to_field_list(mask):
    """Split a 1D boolean array into a list of per-run boolean masks (one per contiguous True run)."""
    n = mask.size
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    fields = []
    for s, e in zip(starts, ends):
        f = np.zeros(n, dtype=bool)
        f[s:e] = True
        fields.append(f)
    return fields


def detect_tentative_fields(place_field_map,
                            n_baseline_bins=25,
                            threshold_factor=0.25):
    """
    Step (a): identify tentative place fields per cell.

    For each cell, threshold = baseline + threshold_factor * (peak - baseline),
    where baseline is the mean of the lowest n_baseline_bins values and peak
    is the max of place_field_map. Returns contiguous runs above threshold.

    Parameters
    ----------
    place_field_map : ndarray (n_cells, n_bins)
        Smoothed event rate per spatial bin.
    n_baseline_bins : int
        Number of lowest bins used to compute baseline. Default 25.
    threshold_factor : float
        Fraction of (peak - baseline) above baseline that defines the
        tentative-field threshold. Default 0.25.

    Returns
    -------
    tentative_fields : list[list[ndarray(bool, n_bins)]]
        Per-cell list of boolean masks; one mask per contiguous tentative
        field. Empty list = no field detected.
    """
    n_cells, n_bins = place_field_map.shape
    tentative_fields = []
    for i_cell in range(n_cells):
        row = place_field_map[i_cell]
        valid_vals = row[~np.isnan(row)]
        if valid_vals.size == 0:
            tentative_fields.append([])
            continue
        peak = valid_vals.max()
        if valid_vals.size < n_baseline_bins:
            baseline = valid_vals.mean()
        else:
            baseline = np.sort(valid_vals)[:n_baseline_bins].mean()
        threshold = baseline + threshold_factor * (peak - baseline)
        # NaN > threshold -> False, so NaN bins automatically break contiguity
        candidate_mask = row > threshold
        tentative_fields.append(_bool_mask_to_field_list(candidate_mask))
    return tentative_fields


def compute_transient_count_per_bin(running_calcium_map_img,
                                    running_distance_map_img,
                                    valid_trials_mask=None,
                                    track_length=180,
                                    bin_size=4):
    """
    Per cell, count running frames where dF/F > 0, binned by mouse position.

    The pipeline already SD-filters dF/F (`robust_filter_along_axis`),
    zeroing non-significant frames, so `> 0` indicates a significant
    calcium transient frame. NaN frames (e.g. dFF thresholded out) do not
    contribute to the count.

    Parameters
    ----------
    running_calcium_map_img : ndarray (n_cells, n_frames)
    running_distance_map_img : ndarray (n_frames,)
    valid_trials_mask : ndarray (n_cells, n_laps) bool, optional
        If provided, only frames in valid laps for each cell are counted —
        making the result consistent with `occupancy_valid` / `place_field_map`.
        If None, all running frames are counted (occupancy is then shared
        across cells).
    track_length, bin_size : float
        Same as elsewhere in this module.

    Returns
    -------
    transient_count_per_bin : ndarray (n_cells, n_bins)
    """
    n_cells, n_frames = running_calcium_map_img.shape
    n_bins = int(track_length / bin_size)
    bin_edges = np.linspace(0, track_length, n_bins + 1)
    bin_idx = np.clip(np.digitize(running_distance_map_img, bin_edges) - 1, 0, n_bins - 1)

    transient_mask = (running_calcium_map_img > 0)  # NaN > 0 -> False

    if valid_trials_mask is not None:
        # Build per-cell frame validity mask from valid_trials_mask + lap boundaries
        distance_diff = np.diff(running_distance_map_img)
        lap_starts = np.where(distance_diff < -track_length / 2)[0] + 1
        lap_starts = np.concatenate([[0], lap_starts, [n_frames]])
        valid_per_cell = np.zeros((n_cells, n_frames), dtype=bool)
        for i_lap in range(len(lap_starts) - 1):
            s, e = lap_starts[i_lap], lap_starts[i_lap + 1]
            valid_per_cell[:, s:e] = valid_trials_mask[:, i_lap:i_lap + 1]
        transient_mask = transient_mask & valid_per_cell

    transient_count_per_bin = np.zeros((n_cells, n_bins))
    np.add.at(transient_count_per_bin, (slice(None), bin_idx), transient_mask.astype(float))
    return transient_count_per_bin


def compute_field_metrics(tentative_fields,
                          place_field_map,
                          transient_count_per_bin,
                          occupancy_per_bin,
                          bin_size=4):
    """
    For each tentative field, compute the four scalars used by
    filter_tentative_fields:

        width_cm           = n_field_bins * bin_size
        peak_dff           = nanmax(place_field_map[field])
        in_out_ratio       = nanmean(in-field) / nanmean(out-of-field)
                             (NaN if out-of-field mean is not > 0)
        transient_fraction = sum(transient_count[field]) / sum(occupancy[field])
                             (NaN if total occupancy is 0)

    Storing these alongside `tentative_field` lets the filter be re-run
    offline without recomputing any per-bin math.

    Parameters
    ----------
    tentative_fields : list[list[ndarray(bool, n_bins)]]
        Per-cell list of tentative-field masks.
    place_field_map : ndarray (n_cells, n_bins)
    transient_count_per_bin : ndarray (n_cells, n_bins)
    occupancy_per_bin : ndarray (n_bins,) or (n_cells, n_bins)

    Returns
    -------
    metrics : dict[str, list[list[float]]]
        Keys: 'width_cm', 'peak_dff', 'in_out_ratio', 'transient_fraction'.
        Each value is a length-n_cells list of per-field float lists,
        parallel-indexed with `tentative_fields`. Empty inner list if the
        cell has no tentative fields.
    """
    n_cells, n_bins = place_field_map.shape
    
    if occupancy_per_bin is None:
        occ_2d = np.nan
    elif occupancy_per_bin.ndim == 1:
        occupancy_per_bin = np.asarray(occupancy_per_bin)
        occ_2d = np.broadcast_to(occupancy_per_bin, (n_cells, n_bins))
    else:
        occ_2d = np.asarray(occupancy_per_bin)

    width_all, peak_all, ratio_all, frac_all = [], [], [], []
    for i_cell in range(n_cells):
        cell_pf = place_field_map[i_cell]
        cell_transient = transient_count_per_bin[i_cell]
        if occupancy_per_bin is not None:
            cell_occ = occ_2d[i_cell]
        widths, peaks, ratios, fracs = [], [], [], []
        for field_mask in tentative_fields[i_cell]:
            in_field_vals = cell_pf[field_mask]
            in_mean = np.nanmean(in_field_vals)
            out_mean = np.nanmean(cell_pf[~field_mask])

            widths.append(float(field_mask.sum() * bin_size))
            peaks.append(float(np.nanmax(in_field_vals)))
            ratios.append(float(in_mean / out_mean) if (out_mean > 0) else float('nan'))
            if occupancy_per_bin is not None:
                n_total = cell_occ[field_mask].sum()
                fracs.append(float(cell_transient[field_mask].sum() / n_total)
                             if n_total > 0 else float('nan'))
            else:
                fracs.append(np.nan)

        width_all.append(widths)
        peak_all.append(peaks)
        ratio_all.append(ratios)
        frac_all.append(fracs)

    return {
        'width_cm': width_all,
        'peak_dff': peak_all,
        'in_out_ratio': ratio_all,
        'transient_fraction': frac_all,
    }


def filter_tentative_fields(tentative_fields,
                            field_metrics=None,
                            min_width_cm=18,
                            min_peak_dff=0.10,
                            min_in_out_ratio=2.5,
                            min_transient_fraction=0.1):
    """
    Step (b): apply Dombeck-style criteria (i)-(iv) using precomputed metrics.

    Each criterion uses `not (metric > threshold)` so NaN metrics correctly
    reject (e.g. when the out-of-field mean is non-positive or the field
    has zero occupancy).

      (i)   width_cm           >= min_width_cm
      (ii)  peak_dff           >= min_peak_dff
      (iii) in_out_ratio       >  min_in_out_ratio   (paper: "more than 3x")
      (iv)  transient_fraction >= min_transient_fraction

    Parameters
    ----------
    tentative_fields : list[list[ndarray(bool, n_bins)]] OR pd.DataFrame
        Either the raw per-cell tentative-field list (paired with `field_metrics`),
        or a place-cell DataFrame containing the columns `tentative_field`,
        `tentative_field_width_cm`, `tentative_field_peak_dff`,
        `tentative_field_in_out_ratio`, `tentative_field_transient_fraction`.
        When a DataFrame is passed, `field_metrics` is ignored.
    field_metrics : dict[str, list[list[float]]], optional
        Output of `compute_field_metrics`. Required when `tentative_fields` is
        a list (not a DataFrame).
    """
    if isinstance(tentative_fields, pd.DataFrame):
        df = tentative_fields
        tentative_fields = [
            [np.asarray(m, dtype=bool) for m in row]
            for row in df['tentative_field']
        ]
        field_metrics = {
            k: df[f'tentative_field_{k}'].tolist()
            for k in ('width_cm', 'peak_dff', 'in_out_ratio', 'transient_fraction')
        }
    elif field_metrics is None:
        raise TypeError("field_metrics is required when tentative_fields is not a DataFrame")

    width_cm = field_metrics['width_cm']
    peak_dff = field_metrics['peak_dff']
    in_out_ratio = field_metrics['in_out_ratio']
    transient_fraction = field_metrics['transient_fraction']

    n_cells = len(tentative_fields)
    final_fields = []
    for i_cell in range(n_cells):
        accepted = []
        for j, field_mask in enumerate(tentative_fields[i_cell]):
            # Each criterion: if threshold is None → skip that check (no filter).
            # Otherwise reject the field if it doesn't meet the threshold.
            # NaN metrics also fail (NaN >= x is False), correctly rejecting.
            if min_width_cm is not None:
                if not (width_cm[i_cell][j] >= min_width_cm):
                    continue
            if min_peak_dff is not None:
                if not (peak_dff[i_cell][j] >= min_peak_dff):
                    continue
            if min_in_out_ratio is not None:
                if not (in_out_ratio[i_cell][j] > min_in_out_ratio):
                    continue
            if min_transient_fraction is not None:
                if not (transient_fraction[i_cell][j] >= min_transient_fraction):
                    continue
            accepted.append(field_mask)
        final_fields.append(accepted)
    return final_fields


def select_significant_cells(df, config=None,
                             min_peak_dff=0.1,
                             min_in_out_ratio=2.0,
                             min_width_cm=12.0,
                             min_transient_fraction=0.15):
    """Mark significant cells using field criteria from Dombeck, 2010, Nat. Neurosci.

    A cell is significant if it has at least one tentative field where
    all four pre-computed metrics pass the given thresholds.
    """

    df = df.copy()

    # Re-filter tentative fields using current thresholds
    final = filter_tentative_fields(
        df,
        min_peak_dff=min_peak_dff,
        min_in_out_ratio=min_in_out_ratio,
        min_width_cm=min_width_cm,
        min_transient_fraction=min_transient_fraction,
    )
    df['is_significant'] = [len(f) > 0 for f in final]
    if config is not None:
        label = config['cell_type_label']
        non_label = config['non_cell_type_label']
        df['cell_type'] = np.where(df['is_significant'], label, non_label)
    return df


def detect_place_field(event_count_valid, occupancy_valid,
                       per_lap_profile=None,
                       valid_trials_mask=None,
                       sigma=1.5,  # bins
                       kernel_size=5,  # bins
                       bin_size=4,  # cm
                       transient_count_per_bin=None,
                       ):
    """
    Detect place fields and compute spatial information for each ROI.

    Parameters:
    -----------
    event_count_valid : ndarray (n_cells, n_bins)
        Sum of dF/F values per spatial bin for each cell, accumulated only
        over valid laps (output of compute_valid_event_rate).
    occupancy_valid : ndarray (n_cells, n_bins)
        Per-cell frame count per spatial bin from valid laps only.
    per_lap_profile : ndarray (n_cells, n_laps, n_bins), optional
        Event rate per spatial bin for each cell and lap (full, with NaN
        invalid laps). Stored as-is in the output DataFrame.
    valid_trials_mask : ndarray (n_cells, n_laps) bool, optional
        Per-cell boolean mask of valid laps. Stored as a column and used
        to compute perc_valid_laps.
    sigma : float
        Gaussian kernel sigma in bins
    kernel_size : int
        Gaussian kernel size in bins
    bin_size : float
        Spatial bin size in cm
    transient_count_per_bin : ndarray (n_cells, n_bins), optional
        Per-cell binned count of significant-transient frames (output of
        `compute_transient_count_per_bin`). When provided, criterion (iv)
        of `filter_tentative_fields` is evaluated and `final_field` is
        populated; otherwise `final_field` is empty for every cell. Saved
        as a column so the filter can be re-run offline with different
        thresholds.

    Returns:
    --------
    df_place_field : DataFrame
        DataFrame containing place field info for each ROI
    """
    n_cells, n_bins = event_count_valid.shape

    # Create truncated Gaussian kernel
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    gaussian_kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # normalize

    # Smooth per-cell occupancy and event counts with Gaussian kernel
    occupancy_smoothed = convolve1d(occupancy_valid, gaussian_kernel, axis=1, mode='constant')
    event_count_smoothed = convolve1d(event_count_valid, gaussian_kernel, axis=1, mode='constant')

    # Compute place field map (smoothed event rate)
    occupancy_smoothed_safe = occupancy_smoothed.copy()
    occupancy_smoothed_safe[occupancy_smoothed_safe == 0] = np.nan
    place_field_map = event_count_smoothed / occupancy_smoothed_safe

    # Normalize place field by maximum value
    place_field_max = np.nanmax(place_field_map, axis=1, keepdims=True)
    place_field_max[place_field_max == 0] = 1  # avoid division by zero
    place_field_normalized = place_field_map / place_field_max

    # Find place field position (bin with peak value)
    # Handle cells where all bins are NaN (no events)
    all_nan_mask = np.all(np.isnan(place_field_map), axis=1)
    place_field_peak_bin = np.zeros(n_cells, dtype=int)
    place_field_peak_bin[~all_nan_mask] = np.nanargmax(place_field_map[~all_nan_mask], axis=1)
    place_field_position_cm = (place_field_peak_bin + 0.5) * bin_size  # center of bin
    place_field_position_cm[all_nan_mask] = np.nan

    # Peak amplitude (max of smoothed place field)
    place_field_peak_amplitude = np.nanmax(place_field_map, axis=1)

    # Compute total events per cell (valid laps only)
    total_events = np.nansum(event_count_valid, axis=1)

    if valid_trials_mask is not None:
        valid_trials_mask_col = [valid_trials_mask[i] for i in range(n_cells)]
        perc_valid_laps = valid_trials_mask.sum(axis=1) / valid_trials_mask.shape[1]
    else:
        valid_trials_mask_col = [None] * n_cells
        perc_valid_laps = np.full(n_cells, np.nan)

    # Tentative fields (step a)
    tentative_fields = detect_tentative_fields(place_field_map)
    # Per-field metrics + final fields (step b) — only if transient_count_per_bin provided
    if transient_count_per_bin is not None:
        metrics = compute_field_metrics(
            tentative_fields, place_field_map,
            transient_count_per_bin, occupancy_valid,
            bin_size=bin_size,
        )
        final_fields = filter_tentative_fields(tentative_fields, metrics)
        transient_count_col = [transient_count_per_bin[i] for i in range(n_cells)]
    else:
        metrics = {k: [[] for _ in range(n_cells)]
                   for k in ('width_cm', 'peak_dff', 'in_out_ratio', 'transient_fraction')}
        final_fields = [[] for _ in range(n_cells)]
        transient_count_col = [None] * n_cells

    # Create DataFrame with place_field_map as list of 1D arrays
    df_place_field = pd.DataFrame({
        'cell_id': np.arange(n_cells),
        'per_lap_profile': [per_lap_profile[i] if per_lap_profile is not None else None for i in range(n_cells)],
        'valid_trials_mask': valid_trials_mask_col,
        'perc_valid_laps': perc_valid_laps,
        'place_field_map': [place_field_map[i] for i in range(n_cells)],
        'place_field_map_norm': [place_field_normalized[i] for i in range(n_cells)],
        'place_field_peak_bin': place_field_peak_bin,
        'place_field_position_cm': place_field_position_cm,
        'place_field_peak_amplitude': place_field_peak_amplitude,
        'total_events': total_events,
        'tentative_field': tentative_fields,
        'final_field': final_fields,
        'tentative_field_width_cm': metrics['width_cm'],
        'tentative_field_peak_dff': metrics['peak_dff'],
        'tentative_field_in_out_ratio': metrics['in_out_ratio'],
        'tentative_field_transient_fraction': metrics['transient_fraction'],
        'transient_count_per_bin': transient_count_col,
        'occupancy_per_bin': [occupancy_valid[i] for i in range(n_cells)],
    })

    return df_place_field


    
def align_run_frame_calcium(dff, beh):
    
    frame_times = beh['frame_times']
    n_frames = dff.shape[-1]
    # covert time dimension to distance
    if (len(frame_times)<n_frames) or ((len(frame_times)-n_frames)>100):
    # if (n_frames-(len(frame_times))>200) or ((len(frame_times)-n_frames)>200):
        running_time_map_img, running_distance_map_img, running_calcium_map_img, running_time_map_img_abs = [], [], [], []
        print(f'frame times and actual frames not matching!!!/ntimes: {len(frame_times)}/frames: {n_frames}')
    else:
        
        frame_times = frame_times[:n_frames]
        
        upsampled_distance_cm = beh['upsampled_distance_cm']
        upsampled_speed_cm_s = beh['upsampled_speed_cm_s']
        upsampled_timestamps_ms = beh['upsampled_timestamps_ms']
    
        # define running time map: speed > 10 cm/s
        #                          distance: 0~180 cm  
        running_time_idx = (upsampled_speed_cm_s>10)&(0<upsampled_distance_cm)&(upsampled_distance_cm<180)
        running_time_map = upsampled_timestamps_ms[running_time_idx]
        running_distance_map = upsampled_distance_cm[running_time_idx]
        
        # find frame stamps for running_time_map
        mapp_frame_idx = nearest_mapping(running_time_map, frame_times)
        
        # binning for maps per frame
        n_frames = len(frame_times)
        # counts per frame
        counts = np.bincount(mapp_frame_idx, minlength=n_frames)
        
        # running_time_map
        # sum of running_time_map per frame
        sums = np.bincount(mapp_frame_idx, weights=running_time_map, minlength=n_frames)
        # mean running_time_map per frame (NaN where no samples)
        running_time_binned = np.full(n_frames, np.nan, dtype=float)
        nonempty = counts > 0
        running_time_binned[nonempty] = sums[nonempty] / counts[nonempty]
        
        # running_distance
        # sum of running_time_map per frame
        sums = np.bincount(mapp_frame_idx, weights=running_distance_map, minlength=n_frames)
        # mean running_time_map per frame (NaN where no samples)
        running_distance_binned = np.full(n_frames, np.nan, dtype=float)
        nonempty = counts > 0
        running_distance_binned[nonempty] = sums[nonempty] / counts[nonempty]
        
        # aligned running time, distance and calcium map (30 Hz)
        runing_idx_img = ~np.isnan(running_time_binned) # frame index when animal runs
        running_frames = np.arange(n_frames)[runing_idx_img]
        n_running_frames = running_frames.shape[0]
        
        if n_running_frames < 9000:
            running_time_map_img, running_distance_map_img, running_calcium_map_img, running_time_map_img_abs = [], [], [], []
            print('no enough running farme!!!')
        else:
            running_time_map_img_abs = running_time_binned[runing_idx_img]  # absolute ms
            running_time_map_img = running_time_map_img_abs - running_time_binned[0] # relative ms
            running_distance_map_img = running_distance_binned[runing_idx_img]
            running_calcium_map_img = dff[:, runing_idx_img]  # (n_cells, n_running_frames)
            
    return running_time_map_img, running_distance_map_img, running_calcium_map_img, running_time_map_img_abs

def dff_thresh(a, hard_thresh=1000, factor=8):
    a = a.flatten()
    aa = a[np.abs(a)<hard_thresh]
    mean = np.nanmean(aa)
    std = np.nanstd(aa)
    # print(mean+8*std)
    return mean+factor*std
#%%
tmp_lst = [{'anm': 'AC989', 'date': '20250711'},]
# test script
if __name__ == '__main__':
    
    # from drug_infusion import rec_lst_infusion as recs
    # rec_ctrl = recs.rec_SCH_ctrl
    # rec_SCH  = recs.rec_SCH
    
    OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
    OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
    
    # example rec
    # rec = rec_ctrl.iloc[3]
    # rec = rec_SCH.iloc[4]
    df_place_field_all = pd.DataFrame()
    
    # for _, rec in rec_SCH.head(3).iterrows():
    for rec in tmp_lst:
        anm = rec['anm']
        date = rec['date']
        print(f'\n{anm}-{date}')
        data_path = OUT_DIR_RAW_DATA/'raw_signals'/f'{anm}-{date}'
        ops = np.load(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\02\suite2p_func_detec\plane0\ops.npz")
        
        p_beh_ss1 = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{anm}-{date}-02.pkl'
        beh_ss1 = pd.read_pickle(p_beh_ss1)
        beh = beh_ss1
        # !!!!
        n_frames = ops['nframes']
        frame_times = len(beh['frame_times'])
        
        is_active_soma = np.load(data_path/r'soma_class.npz')['is_soma']
        dff_ss1 = np.load(data_path/f'{anm}-{date}-02_dFF.npy')[is_active_soma]
        n_dff_frames = dff_ss1.shape[-1]
        print(f'{anm}-{date}-02: nframes: {n_frames}, frame_times: {frame_times}, len_dff: {n_dff_frames}')
        
        #%%
        thresh = dff_thresh(dff_ss1)
        dff_sd = robust_filter_along_axis(dff_ss1, factor=2.5)
        dff_sd[abs(dff_sd)>thresh]=np.nan
        dff = dff_sd
        
        running_time_map_img, running_distance_map_img, running_calcium_map_img, running_time_map_img_abs = align_run_frame_calcium(dff, beh)
        
        # Run analysis using the functions
        # Parameters
        track_length = 180  # cm
        bin_size = 4  # cm
        n_bins = int(track_length / bin_size)  # 45 bins
    
        # Spatial binning
        event_count_raw, event_rate_raw, occupancy_raw, per_lap_profile, per_lap_occupancy = spatial_binning(
            running_calcium_map_img, running_distance_map_img,
            track_length=track_length, bin_size=bin_size
        )

        # Per-cell event_rate and occupancy from valid laps only (drop laps
        # with any NaN bin — e.g. dFF thresholded out)
        valid_trials_mask = ~np.any(np.isnan(per_lap_profile), axis=-1)
        event_count_valid, event_rate_valid, occupancy_valid = compute_valid_event_rate(
            per_lap_profile, per_lap_occupancy, valid_trials_mask
        )

        # Per-cell binned count of significant-transient frames (valid laps
        # only) — drives criterion (iv) and is saved for offline retuning.
        transient_count_per_bin = compute_transient_count_per_bin(
            running_calcium_map_img, running_distance_map_img,
            valid_trials_mask=valid_trials_mask,
            track_length=track_length, bin_size=bin_size,
        )

        # Detect place fields (place_field_map uses valid laps only)
        df_place_field = detect_place_field(
            event_count_valid, occupancy_valid,
            per_lap_profile=per_lap_profile,
            valid_trials_mask=valid_trials_mask,
            sigma=1.5, kernel_size=5, bin_size=bin_size,
            transient_count_per_bin=transient_count_per_bin,
        )

        total_active_frames = np.sum(running_calcium_map_img>0, axis=-1)
        df_place_field['total_active_frames'] = total_active_frames
        # Compute spatial information (valid trials only)
        spatial_information = calculate_spatial_info(event_rate_valid, occupancy_valid)
        df_place_field['spatial_information_bits'] = spatial_information

        n_running_frames = running_calcium_map_img.shape[-1]
        # Only compute shuffled SI for ROIs with SI > threshold
        SI_threshold_for_shuffle = 0.05
        candidate_mask = ((total_active_frames>0.1*n_running_frames)&
                          (spatial_information > SI_threshold_for_shuffle)&
                          (~np.isnan(spatial_information))
                          )
        candidate_indices = np.where(candidate_mask)[0]

        # Initialize shuffled_SI column with None
        df_place_field['shuffled_SI'] = None

        if len(candidate_indices) > 0:
            print(f"Computing shuffled SI for {len(candidate_indices)} ROIs with SI > {SI_threshold_for_shuffle}")
            candidate_indices, shuffled_SI = (
                calculate_spatial_info_shuffle_by_valid_laps(
                    running_calcium_map_img, running_distance_map_img,
                    valid_trials_mask, candidate_indices,
                    min_valid_laps=80, shuff_times=1000, min_shift=450,
                    gpu=True, track_length=track_length, bin_size=bin_size,
                )
            )
            for i, idx in enumerate(candidate_indices):
                df_place_field.at[idx, 'shuffled_SI'] = shuffled_SI[i]
        # Extract values from DataFrame for convenience
        n_cells = len(df_place_field)
        place_field_position_cm = df_place_field['place_field_position_cm'].values
        total_events = df_place_field['total_events'].values
    
        print(f"\nPlace cell analysis complete:")
        print(f"  Total cells: {n_cells}")
        print(f"  Cells with >5 events: {np.sum(total_events > 5)}")
        print(f"  Mean spatial information: {np.nanmean(spatial_information):.3f} bits/event")

        df_place_field_all = pd.concat((df_place_field_all, df_place_field))
    #%% Visualization
    import matplotlib.pyplot as plt
    
    # Reset index for proper indexing after concatenation
    df_place_field_all = df_place_field_all.reset_index(drop=True)
    
    # Define place cells: SI > threshold and not NaN
    SI_threshold = 0.2
    shuff_SI_thresh = 99
    df_place_field_all['shuffle_SI_thresh'] = df_place_field_all['shuffled_SI'].apply(
        lambda x: np.nanpercentile(x, shuff_SI_thresh) if x is not None else np.nan
    )
    # Get place cell indices as numpy array
    df_place_field_all['is_place_cell'] = ((df_place_field_all['spatial_information_bits'] > SI_threshold) &
                                            (df_place_field_all['spatial_information_bits'] > df_place_field_all['shuffle_SI_thresh']))
    
    place_cell_indices = np.where(df_place_field_all['is_place_cell'])[0]
    
    print(f"  Place cells (SI > {SI_threshold}): {len(place_cell_indices)}")
    
    # Get data from DataFrame
    place_field_position_cm_all = df_place_field_all['place_field_position_cm'].values
    spatial_information_all = df_place_field_all['spatial_information_bits'].values
    
    # Sort place cells by place field position
    sort_order = np.argsort(place_field_position_cm_all[place_cell_indices])
    sorted_indices = place_cell_indices[sort_order]
    
    # Get normalized place fields for sorted cells (stack from DataFrame)
    place_fields_sorted = np.vstack(df_place_field_all.loc[sorted_indices, 'place_field_map_norm'].values)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Place cell sequence heatmap (sorted by peak position)
    ax1 = axes[0, 0]
    bin_centers = np.arange(0, track_length, bin_size) + bin_size / 2
    imshow_trace = normalize(place_fields_sorted)
    im1 = ax1.imshow(imshow_trace, aspect='auto', cmap='viridis',
                     extent=[0, track_length, len(sorted_indices), 0],
                     interpolation='nearest')
    ax1.set_xlabel('Position (cm)')
    ax1.set_ylabel('Cell # (sorted by place field)')
    ax1.set_title('Place Cell Sequence')
    plt.colorbar(im1, ax=ax1, label='Normalized activity')
    
    # 2. Distribution of place field positions (place cells only)
    ax2 = axes[0, 1]
    ax2.hist(place_field_position_cm_all[place_cell_indices], bins=n_bins // 2,
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
    shuffle_thresh_all = df_place_field_all['shuffle_SI_thresh'].values
    valid_mask = ~np.isnan(shuffle_thresh_all)
    ax4.scatter(spatial_information_all[valid_mask], shuffle_thresh_all[valid_mask],
                alpha=0.3, s=10)
    ax4.plot([0, np.nanmax(spatial_information_all)], [0, np.nanmax(spatial_information_all)],
             'r--', label='SI = shuffle threshold')
    ax4.set_xlabel('Spatial information (bits)')
    ax4.set_ylabel('99th percentile shuffle SI')
    ax4.set_title('SI vs Shuffle Threshold')
    ax4.legend()
    
    plt.suptitle(f'All recordings (n={len(df_place_field_all)} cells)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    #%% Example place cells (top 6 by spatial information, from place cells only)
    if len(place_cell_indices) >= 6:
        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6))
        axes2 = axes2.flatten()
    
        # Get top 6 place cells by spatial information
        top_si_order = np.argsort(spatial_information_all[place_cell_indices])[::-1][:6]
        top_place_cells = place_cell_indices[top_si_order]
    
        for i, cell_idx in enumerate(top_place_cells):
            ax = axes2[i]
            ax.bar(bin_centers, df_place_field_all.loc[cell_idx, 'place_field_map'], width=bin_size * 0.8, alpha=0.7)
            ax.axvline(place_field_position_cm_all[cell_idx], color='r', linestyle='--', alpha=0.7)
            ax.set_xlabel('Position (cm)')
            ax.set_ylabel('Event rate')
            ax.set_title(f'Cell {cell_idx}, SI={spatial_information_all[cell_idx]:.2f} bits')
            ax.set_xlim([0, track_length])
    
        plt.suptitle(f'Top 6 Place Cells (SI > {SI_threshold})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
#%%
# for i in range(50):
#     df_roi = df_place_field_all.loc[df_place_field_all['is_place_cell']].iloc[i]
#     per_lap_profile = df_roi['per_lap_profile']
#     fig, ax = plt.subplots(figsize=(2.2, 2))
#     ax.imshow(normalize(per_lap_profile),
#               aspect='auto', cmap='viridis',
#               extent=[0, track_length, per_lap_profile.shape[0], 0],
#               interpolation='none')
#     field = df_roi['place_field_position_cm']
#     SI = df_roi['spatial_information_bits']
#     ax.set_title(f'field_cm: {field}, SI: {SI:.3f}')
#     plt.show()
