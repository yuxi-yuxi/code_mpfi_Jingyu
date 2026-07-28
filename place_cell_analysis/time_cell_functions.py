# -*- coding: utf-8 -*-
"""
Time Cell Analysis Functions

Parallel to place_cell_functions.py but for temporal tuning analysis.
Instead of spatial binning (distance), this module bins activity by time within each lap/trial.

Time bin: 0.1s (3 frames at 30 Hz)

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from tqdm import tqdm
from common.utils_basic import nearest_mapping, normalize, vectorized_roll
from common.robust_sd_filter import robust_filter_along_axis

# Import shared trial correlation utilities
from .utils_trial_correlation import (
    normalize_per_lap_profile,
    pearson_corr_rows_gpu,
    calculate_trial_correlations_gpu,
    calculate_all_trial_correlations_gpu,
)


def temporal_binning(running_calcium_map_img,
                     running_time_map_img,
                     running_distance_map_img,
                     track_length=180,
                     time_bin_size=0.1,  # seconds (0.1s = 3 frames at 30 Hz)
                     max_lap_duration_s=4.0,  # fixed time window in seconds
                     frame_rate=30,  # Hz
                     save_lap_profile=True,
                     ):
    """
    Bin calcium activity by time within each lap.

    For time cell analysis, we bin activity by relative time within each lap,
    not by absolute distance. This captures neurons that fire at specific
    times during the trial regardless of position.

    Parameters:
    -----------
    running_calcium_map_img : ndarray (n_cells, n_frames)
        Calcium traces during running for each cell
    running_time_map_img : ndarray (n_frames,)
        Time (ms) at each running frame
    running_distance_map_img : ndarray (n_frames,)
        Distance at each running frame (used for lap detection)
    track_length : float
        Track length in cm (for lap boundary detection)
    time_bin_size : float
        Time bin size in seconds (default 0.1s = 3 frames at 30Hz)
    max_lap_duration_s : float
        Fixed time window for binning in seconds (default 4.0s)
        All laps are binned into this fixed window
    frame_rate : float
        Imaging frame rate in Hz
    save_lap_profile : bool
        Whether to save per-lap temporal profiles

    Returns:
    --------
    event_count_raw : ndarray (n_cells, n_time_bins)
        Sum of dF/F values per time bin for each cell (averaged across laps)
    event_rate_raw : ndarray (n_cells, n_time_bins)
        Mean dF/F per time bin for each cell
    occupancy_raw : ndarray (n_time_bins,)
        Total frame count per time bin (summed across laps)
    per_lap_profile : ndarray (n_cells, n_laps, n_time_bins) or []
        Event rate per time bin for each cell and lap. Bins beyond the
        actual lap end (lap shorter than max_lap_duration_s) are NaN
        because their per-lap occupancy is 0.
    per_lap_occupancy : ndarray (n_laps, n_time_bins) or []
        Frame count per time bin per lap. Trailing bins are 0 for laps
        shorter than max_lap_duration_s.
    n_time_bins : int
        Number of time bins used
    median_lap_duration_s : float
        Median lap duration in seconds
    """
    n_cells = running_calcium_map_img.shape[0]

    # Detect lap boundaries: where distance decreases significantly (lap reset)
    distance_diff = np.diff(running_distance_map_img)
    lap_start_indices = np.where(distance_diff < -track_length / 2)[0] + 1
    lap_start_indices = np.concatenate([[0], lap_start_indices, [len(running_distance_map_img)]])
    n_laps = len(lap_start_indices) - 1

    if n_laps < 2:
        return (np.array([]), np.array([]), np.array([]), [], [], 0, 0)

    # Calculate lap durations for reference
    lap_durations_frames = []
    for i_lap in range(n_laps):
        lap_start = lap_start_indices[i_lap]
        lap_end = lap_start_indices[i_lap + 1]
        lap_durations_frames.append(lap_end - lap_start)

    lap_durations_frames = np.array(lap_durations_frames)
    median_lap_duration_frames = np.median(lap_durations_frames)
    median_lap_duration_s = median_lap_duration_frames / frame_rate

    # Use fixed time window (e.g., 4 seconds) with fixed bin size (0.1s)
    # This gives 40 time bins for 4 seconds
    frames_per_bin = int(time_bin_size * frame_rate)  # 3 frames for 0.1s at 30Hz
    n_time_bins = int(max_lap_duration_s / time_bin_size)  # 60 bins for 6 s at 0.1s bins

    # Initialize accumulators
    occupancy_raw = np.zeros(n_time_bins)
    event_count_raw = np.zeros((n_cells, n_time_bins))

    if save_lap_profile:
        per_lap_profile = np.full((n_cells, n_laps, n_time_bins), np.nan)
        per_lap_occupancy = np.zeros((n_laps, n_time_bins))
    else:
        per_lap_profile = []
        per_lap_occupancy = []

    # Process each lap
    for i_lap in range(n_laps):
        lap_start = lap_start_indices[i_lap]
        lap_end = lap_start_indices[i_lap + 1]
        lap_n_frames = lap_end - lap_start

        if lap_n_frames < frames_per_bin:
            continue

        lap_calcium = running_calcium_map_img[:, lap_start:lap_end]  # (n_cells, lap_n_frames)

        # Assign each frame to a time bin based on actual time within lap
        # Each frame at index i corresponds to time i/frame_rate seconds
        frame_indices_in_lap = np.arange(lap_n_frames)
        time_in_lap_s = frame_indices_in_lap / frame_rate  # actual time in seconds

        # Bin based on absolute time (0.1s bins)
        bin_indices = (time_in_lap_s / time_bin_size).astype(int)
        # Only include frames within the max_lap_duration window
        valid_frames = bin_indices < n_time_bins
        bin_indices = bin_indices[valid_frames]
        lap_calcium_valid = lap_calcium[:, valid_frames]

        if len(bin_indices) == 0:
            continue

        # Occupancy for this lap
        lap_occupancy = np.bincount(bin_indices, minlength=n_time_bins).astype(float)
        occupancy_raw += lap_occupancy

        # Sum dF/F per time bin for each cell
        for i_cell in range(n_cells):
            cell_trace = lap_calcium_valid[i_cell]
            dff_sum = np.bincount(bin_indices, weights=cell_trace, minlength=n_time_bins)
            event_count_raw[i_cell] += dff_sum

        # Per-lap profile
        if save_lap_profile:
            per_lap_occupancy[i_lap] = lap_occupancy
            lap_occupancy_safe = lap_occupancy.copy()
            lap_occupancy_safe[lap_occupancy_safe == 0] = np.nan

            dff_sum_lap = np.zeros((n_cells, n_time_bins))
            np.add.at(dff_sum_lap, (slice(None), bin_indices), lap_calcium_valid)
            per_lap_profile[:, i_lap, :] = dff_sum_lap / lap_occupancy_safe

    # Compute event rate (mean dF/F per time bin)
    occupancy_safe = occupancy_raw.copy()
    occupancy_safe[occupancy_safe == 0] = np.nan
    event_rate_raw = event_count_raw / occupancy_safe

    return event_count_raw, event_rate_raw, occupancy_raw, per_lap_profile, per_lap_occupancy, n_time_bins, median_lap_duration_s


def compute_valid_event_rate(per_lap_profile, per_lap_occupancy, valid_trials_mask=None):
    """
    Recompute per-cell event rate and occupancy from only valid laps.

    Unlike the spatial version, NaN in per_lap_profile has two sources here:
      1. Extreme dFF values in the middle of a lap (occupancy > 0 but dFF was
         masked out) — these laps are invalid and excluded per cell.
      2. Laps shorter than max_lap_duration_s, leaving trailing bins with
         occupancy == 0 — these are NOT treated as invalid; the valid bins
         from such laps still contribute.

    Parameters:
    -----------
    per_lap_profile : ndarray (n_cells, n_laps, n_time_bins)
        Per-lap mean dF/F per time bin
    per_lap_occupancy : ndarray (n_laps, n_time_bins)
        Frame count per time bin per lap (0 in trailing bins for short laps)
    valid_trials_mask : ndarray (n_cells, n_laps) or None
        Boolean mask; True = use that lap for that cell. If None, defaults
        to excluding laps that have a NaN bin where the lap actually had
        occupancy (case 1 above).

    Returns:
    --------
    event_count : ndarray (n_cells, n_time_bins)
    event_rate : ndarray (n_cells, n_time_bins)
    occupancy : ndarray (n_cells, n_time_bins)
    """
    per_lap_profile = np.asarray(per_lap_profile)
    per_lap_occupancy = np.asarray(per_lap_occupancy)

    if valid_trials_mask is None:
        # Only NaN bins with real occupancy (case 1) invalidate a trial
        invalid_middle_nan = np.isnan(per_lap_profile) & (per_lap_occupancy[None, :, :] > 0)
        valid_trials_mask = ~np.any(invalid_middle_nan, axis=-1)

    # per_lap_profile is mean dF/F per bin; multiply back by occupancy
    # to get sum of dF/F per bin per lap. Zero out bins with no occupancy
    # to avoid NaN * 0 propagation from short-lap trailing bins.
    per_lap_count = np.where(
        per_lap_occupancy[None, :, :] > 0,
        per_lap_profile * per_lap_occupancy[None, :, :],
        0.0,
    )

    mask_3d = valid_trials_mask[:, :, None]  # (n_cells, n_laps, 1)

    event_count = np.nansum(np.where(mask_3d, per_lap_count, 0.0), axis=1)
    occupancy = np.sum(np.where(mask_3d, per_lap_occupancy[None, :, :], 0.0), axis=1)

    occupancy_safe = np.where(occupancy > 0, occupancy, np.nan)
    event_rate = event_count / occupancy_safe

    return event_count, event_rate, occupancy


def calculate_temporal_info(event_rate, occupancy, gpu=False):
    """
    Vectorized calculation of temporal information for all ROIs.

    Temporal Information = sum_t [ p_t * (r_t / r_bar) * log2(r_t / r_bar) ]
    where r_t = event rate in time bin t, p_t = probability of time bin t, r_bar = mean event rate

    Parameters:
    -----------
    event_rate : ndarray (n_cells, n_time_bins) or (n_time_bins,)
        Unsmoothed event rate (mean dF/F) per time bin for each cell
    occupancy : ndarray (n_time_bins,) or (n_cells, n_time_bins)
        Frame count per time bin. If 2D, per-cell occupancies are used
        (e.g. after excluding invalid trials per cell).
    gpu : bool
        Whether to use GPU acceleration with CuPy

    Returns:
    --------
    temporal_info : ndarray (n_cells,) or scalar
        Temporal information in bits per event for each cell
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
        event_rate = event_rate[xp.newaxis, :]
        squeeze_output = True

    # Support shared (n_time_bins,) or per-cell (n_cells, n_time_bins) occupancy
    if occupancy.ndim == 1:
        total_frames = xp.sum(occupancy)
        p_t = occupancy / total_frames  # (n_time_bins,)
    else:
        total_frames = xp.sum(occupancy, axis=1, keepdims=True)  # (n_cells, 1)
        total_frames_safe = xp.where(total_frames == 0, xp.nan, total_frames)
        p_t = occupancy / total_frames_safe  # (n_cells, n_time_bins)

    # Mean event rate for each cell: r_bar = sum(r_t * p_t)
    r_bar = xp.nansum(event_rate * p_t, axis=1, keepdims=True)

    # Avoid division by zero
    r_bar_safe = xp.where(r_bar == 0, xp.nan, r_bar)

    # r_ratio = r_t / r_bar for each cell and time bin
    r_ratio = event_rate / r_bar_safe

    # Mask for valid bins: occupancy > 0 and r_t > 0
    valid_mask = (occupancy > 0) & (event_rate > 0)

    # Compute: p_t * (r_t / r_bar) * log2(r_t / r_bar)
    log_term = xp.where(r_ratio > 0, xp.log2(r_ratio), 0)
    ti_per_bin = p_t * r_ratio * log_term

    # Zero out invalid bins and sum
    ti_per_bin = xp.where(valid_mask, ti_per_bin, 0)
    temporal_info = xp.nansum(ti_per_bin, axis=1)

    # Set to NaN for cells with r_bar == 0
    temporal_info = xp.where(r_bar.flatten() == 0, xp.nan, temporal_info)

    if squeeze_output:
        temporal_info = temporal_info[0]

    if gpu:
        temporal_info = cp.asnumpy(temporal_info)

    return temporal_info


def calculate_temporal_info_shuffle(running_calcium_map_img,
                                    running_time_map_img,
                                    running_distance_map_img,
                                    shuff_times=1000,
                                    min_shift=450,  # 15s at 30Hz
                                    gpu=True,
                                    track_length=180,
                                    time_bin_size=0.1,
                                    max_lap_duration_s=4.0,
                                    frame_rate=30):
    """
    Compute shuffled temporal information by circular shifting calcium traces.

    For each shuffle, per-cell event rate / occupancy are recomputed from
    valid laps only (dropping laps where a NaN bin falls on a visited time
    bin — i.e. case 1 / extreme dFF). Trailing NaNs from short laps are
    tolerated.

    Parameters:
    -----------
    running_calcium_map_img : ndarray (n_cells, n_frames)
        Calcium traces during running for each cell
    running_time_map_img : ndarray (n_frames,)
        Time at each running frame
    running_distance_map_img : ndarray (n_frames,)
        Distance at each running frame
    shuff_times : int
        Number of shuffle iterations
    min_shift : int
        Minimum shift in frames (15s default)
    gpu : bool
        Whether to use GPU acceleration
    track_length : float
        Track length in cm
    time_bin_size : float
        Time bin size in seconds
    max_lap_duration_s : float
        Fixed time window in seconds
    frame_rate : float
        Frame rate in Hz

    Returns:
    --------
    shuffled_TI : ndarray (n_cells, shuff_times)
        Shuffled temporal information for each cell and iteration
    """
    n_rois, n_frames = running_calcium_map_img.shape
    shuffled_TI = np.zeros((n_rois, shuff_times))

    # Random shift amounts for each shuffle
    rng = np.random.default_rng()
    shift_amounts = rng.integers(min_shift, n_frames - min_shift, size=(shuff_times, n_rois))

    if gpu:
        import cupy as cp
        calcium_gpu = cp.asarray(running_calcium_map_img)
    else:
        calcium_gpu = running_calcium_map_img

    for i in tqdm(range(shuff_times), desc='calculating shuffled TI...'):
        # Vectorized roll: shift each ROI by different amount
        if gpu:
            import cupy as cp
            shifts_gpu = cp.asarray(shift_amounts[i])
            calcium_shuff = vectorized_roll(calcium_gpu, shifts_gpu, xp=cp)
            calcium_shuff_np = cp.asnumpy(calcium_shuff)
        else:
            calcium_shuff_np = vectorized_roll(calcium_gpu, shift_amounts[i], xp=np)

        # Re-bin shuffled traces, keeping per-lap profile so we can exclude
        # invalid laps after the shift
        _, _, _, per_lap_profile_shuff, per_lap_occupancy_shuff, _, _ = temporal_binning(
            calcium_shuff_np, running_time_map_img, running_distance_map_img,
            track_length=track_length, time_bin_size=time_bin_size,
            max_lap_duration_s=max_lap_duration_s, frame_rate=frame_rate,
            save_lap_profile=True,
        )

        if len(per_lap_profile_shuff) == 0:
            shuffled_TI[:, i] = np.nan
            continue

        # Per-cell event_rate / occupancy from valid laps only
        _, event_rate_valid, occupancy_valid = compute_valid_event_rate(
            per_lap_profile_shuff, per_lap_occupancy_shuff
        )

        shuffled_TI[:, i] = calculate_temporal_info(event_rate_valid, occupancy_valid, gpu=gpu)

    return shuffled_TI


def calculate_trial_stability(per_lap_profile, method='odd_even'):
    """
    Calculate time cell stability from per-lap temporal profiles.

    Stability is quantified as the correlation between split-half averages,
    measuring how consistent the temporal tuning is across laps/trials.

    Parameters:
    -----------
    per_lap_profile : ndarray (n_laps, n_time_bins)
        Event rate per time bin for each lap
    method : str
        Method for calculating stability:
        - 'odd_even': Correlation between odd and even lap averages
        - 'first_second_half': Correlation between first and second half averages

    Returns:
    --------
    stability : float
        Stability metric (correlation coefficient, range [-1, 1])
    """
    if per_lap_profile is None or len(per_lap_profile) == 0:
        return np.nan

    if isinstance(per_lap_profile, list):
        per_lap_profile = np.array(per_lap_profile)

    if per_lap_profile.ndim != 2:
        return np.nan

    n_laps, n_time_bins = per_lap_profile.shape

    if n_laps < 2 or n_time_bins < 3:
        return np.nan

    # Remove laps with all NaN values
    valid_laps = ~np.all(np.isnan(per_lap_profile), axis=1)
    per_lap_profile = per_lap_profile[valid_laps]
    n_laps = per_lap_profile.shape[0]

    if n_laps < 2:
        return np.nan

    if method == 'odd_even':
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


def detect_time_field(event_count_raw, event_rate_raw, occupancy_raw,
                      per_lap_profile=None,
                      n_time_bins=None,
                      median_lap_duration_s=None,
                      max_lap_duration_s=4.0,  # fixed time window
                      sigma=1.5,  # bins
                      kernel_size=5,  # bins
                      time_bin_size=0.1,  # seconds
                      ):
    """
    Detect time fields and compute temporal information for each ROI.

    Parameters:
    -----------
    event_count_raw : ndarray (n_cells, n_time_bins)
        Sum of dF/F values per time bin for each cell
    event_rate_raw : ndarray (n_cells, n_time_bins)
        Mean dF/F per time bin for each cell
    occupancy_raw : ndarray (n_time_bins,)
        Frame count per time bin
    per_lap_profile : ndarray (n_cells, n_laps, n_time_bins), optional
        Event rate per time bin for each cell and lap
    n_time_bins : int
        Number of time bins
    median_lap_duration_s : float
        Median lap duration in seconds (for reference only)
    max_lap_duration_s : float
        Fixed time window in seconds (default 4.0s)
    sigma : float
        Gaussian kernel sigma in bins
    kernel_size : int
        Gaussian kernel size in bins
    time_bin_size : float
        Time bin size in seconds

    Returns:
    --------
    df_time_field : DataFrame
        DataFrame containing time field info for each ROI
    """
    n_cells = event_count_raw.shape[0]
    n_bins = event_count_raw.shape[1] if n_time_bins is None else n_time_bins

    # Create truncated Gaussian kernel
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    gaussian_kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Smooth occupancy with Gaussian kernel
    occupancy_smoothed = convolve1d(occupancy_raw, gaussian_kernel, mode='constant')

    # Smooth event counts with Gaussian kernel
    event_count_smoothed = np.apply_along_axis(
        lambda row: convolve1d(row, gaussian_kernel, mode='constant'),
        axis=1, arr=event_count_raw
    )

    # Compute time field map (smoothed event rate)
    occupancy_smoothed_safe = occupancy_smoothed.copy()
    occupancy_smoothed_safe[occupancy_smoothed_safe == 0] = np.nan
    time_field_map = event_count_smoothed / occupancy_smoothed_safe

    # Normalize time field by maximum value
    time_field_max = np.nanmax(time_field_map, axis=1, keepdims=True)
    time_field_max[time_field_max == 0] = 1
    time_field_normalized = time_field_map / time_field_max

    # Find time field position (bin with peak value)
    time_field_peak_bin = np.nanargmax(time_field_map, axis=1)

    # Convert to normalized time (0-1) and actual time in seconds
    # Use fixed time window (max_lap_duration_s) for time_field_position_s
    time_field_position_norm = (time_field_peak_bin + 0.5) / n_bins  # center of bin, normalized
    time_field_position_s = (time_field_peak_bin + 0.5) * time_bin_size  # actual time in seconds

    # Peak amplitude
    time_field_peak_amplitude = np.nanmax(time_field_map, axis=1)

    # Total events per cell
    total_events = np.nansum(event_count_raw, axis=1)

    # Create DataFrame
    df_time_field = pd.DataFrame({
        'cell_id': np.arange(n_cells),
        'per_lap_profile': [per_lap_profile[i] if per_lap_profile is not None and len(per_lap_profile) > 0 else None for i in range(n_cells)],
        'time_field_map': [time_field_map[i] for i in range(n_cells)],
        'time_field_map_norm': [time_field_normalized[i] for i in range(n_cells)],
        'time_field_peak_bin': time_field_peak_bin,
        'time_field_position_norm': time_field_position_norm,  # 0-1 normalized within lap
        'time_field_position_s': time_field_position_s,  # actual time in seconds (0 to max_lap_duration_s)
        'time_field_peak_amplitude': time_field_peak_amplitude,
        'total_events': total_events,
    })

    return df_time_field


def align_run_frame_calcium(dff, beh):
    """
    Align calcium imaging frames to behavior data during running.

    Same as place_cell_functions.align_run_frame_calcium but kept here
    for completeness and potential modifications.

    Parameters:
    -----------
    dff : ndarray (n_cells, n_frames)
        dF/F calcium traces
    beh : dict
        Behavior data containing frame_times, upsampled_distance_cm,
        upsampled_speed_cm_s, upsampled_timestamps_ms

    Returns:
    --------
    running_time_map_img : ndarray
        Time (ms) at each running frame
    running_distance_map_img : ndarray
        Distance (cm) at each running frame
    running_calcium_map_img : ndarray (n_cells, n_running_frames)
        Calcium traces during running
    """
    frame_times = beh['frame_times']
    n_frames = dff.shape[-1]

    if (len(frame_times) < n_frames) or ((len(frame_times) - n_frames) > 100):
        print('frame times and actual frames not matching!!!')
        return [], [], []

    frame_times = frame_times[:n_frames]

    upsampled_distance_cm = beh['upsampled_distance_cm']
    upsampled_speed_cm_s = beh['upsampled_speed_cm_s']
    upsampled_timestamps_ms = beh['upsampled_timestamps_ms']

    # Define running time map: speed > 10 cm/s, distance: 0~180 cm
    running_time_idx = (upsampled_speed_cm_s > 10) & (0 < upsampled_distance_cm) & (upsampled_distance_cm < 180)
    running_time_map = upsampled_timestamps_ms[running_time_idx]
    running_distance_map = upsampled_distance_cm[running_time_idx]

    # Find frame stamps for running_time_map
    mapp_frame_idx = nearest_mapping(running_time_map, frame_times)

    # Binning for maps per frame
    n_frames = len(frame_times)
    counts = np.bincount(mapp_frame_idx, minlength=n_frames)

    # Running time map
    sums = np.bincount(mapp_frame_idx, weights=running_time_map, minlength=n_frames)
    running_time_binned = np.full(n_frames, np.nan, dtype=float)
    nonempty = counts > 0
    running_time_binned[nonempty] = sums[nonempty] / counts[nonempty]

    # Running distance
    sums = np.bincount(mapp_frame_idx, weights=running_distance_map, minlength=n_frames)
    running_distance_binned = np.full(n_frames, np.nan, dtype=float)
    nonempty = counts > 0
    running_distance_binned[nonempty] = sums[nonempty] / counts[nonempty]

    # Aligned running time, distance and calcium map (30 Hz)
    runing_idx_img = ~np.isnan(running_time_binned)
    running_frames = np.arange(n_frames)[runing_idx_img]
    n_running_frames = running_frames.shape[0]

    if n_running_frames < 9000:
        print('not enough running frames!!!')
        return [], [], []

    running_time_map_img = running_time_binned - running_time_binned[0]  # ms
    running_time_map_img = running_time_map_img[runing_idx_img]
    running_distance_map_img = running_distance_binned[runing_idx_img]
    running_calcium_map_img = dff[:, runing_idx_img]

    return running_time_map_img, running_distance_map_img, running_calcium_map_img


#%%
# Test script
if __name__ == '__main__':

    from drug_infusion import rec_lst_infusion as recs
    rec_ctrl = recs.rec_SCH_ctrl
    rec_SCH = recs.rec_SCH

    OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
    OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"

    # Parameters
    time_bin_size = 0.1  # seconds (3 frames at 30Hz)
    frame_rate = 30  # Hz

    df_time_field_all = pd.DataFrame()

    for _, rec in rec_SCH.head(1).iterrows():
        anm = rec['anm']
        date = rec['date']
        print(f'\n{anm}-{date}')
        data_path = OUT_DIR_RAW_DATA / 'raw_signals' / f'{anm}-{date}'

        p_beh_ss1 = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{anm}-{date}-02.pkl'
        beh_ss1 = pd.read_pickle(p_beh_ss1)
        beh = beh_ss1

        is_active_soma = np.load(data_path / r'soma_class.npz')['is_soma']
        dff_ss1 = np.load(data_path / f'{anm}-{date}-02_dFF.npy')[is_active_soma]

        dff = robust_filter_along_axis(dff_ss1, factor=2.5)
        running_time_map_img, running_distance_map_img, running_calcium_map_img = align_run_frame_calcium(dff, beh)

        if len(running_calcium_map_img) == 0:
            continue

        # Temporal binning
        event_count_raw, event_rate_raw, occupancy_raw, per_lap_profile, per_lap_occupancy, n_time_bins, median_lap_duration_s = temporal_binning(
            running_calcium_map_img, running_time_map_img, running_distance_map_img,
            time_bin_size=time_bin_size, frame_rate=frame_rate
        )

        print(f"  n_time_bins: {n_time_bins}, median_lap_duration: {median_lap_duration_s:.2f}s")

        # Per-cell event rate / occupancy from valid laps only (drop laps
        # with a NaN bin where occupancy > 0 — i.e. extreme dFF mid-lap).
        valid_trials_mask = ~np.any(
            np.isnan(per_lap_profile) & (per_lap_occupancy[None, :, :] > 0),
            axis=-1,
        )
        _, event_rate_valid, occupancy_valid = compute_valid_event_rate(
            per_lap_profile, per_lap_occupancy, valid_trials_mask
        )

        # Detect time fields
        df_time_field = detect_time_field(
            event_count_raw, event_rate_raw, occupancy_raw,
            per_lap_profile=per_lap_profile,
            n_time_bins=n_time_bins,
            median_lap_duration_s=median_lap_duration_s,
            sigma=1.5, kernel_size=5, time_bin_size=time_bin_size
        )

        total_active_frames = np.sum(running_calcium_map_img > 0, axis=-1)
        df_time_field['total_active_frames'] = total_active_frames

        # Compute temporal information from valid trials only
        temporal_information = calculate_temporal_info(event_rate_valid, occupancy_valid)
        df_time_field['temporal_information_bits'] = temporal_information

        # Store metadata
        df_time_field['n_time_bins'] = n_time_bins
        df_time_field['median_lap_duration_s'] = median_lap_duration_s

        n_cells = len(df_time_field)
        print(f"\nTime cell analysis complete:")
        print(f"  Total cells: {n_cells}")
        print(f"  Mean temporal information: {np.nanmean(temporal_information):.3f} bits/event")

        df_time_field_all = pd.concat((df_time_field_all, df_time_field))

    #%% Visualization
    import matplotlib.pyplot as plt

    df_time_field_all = df_time_field_all.reset_index(drop=True)

    # Define time cells: TI > threshold
    TI_threshold = 0.15
    time_cell_mask = df_time_field_all['temporal_information_bits'] > TI_threshold
    time_cell_indices = np.where(time_cell_mask)[0]

    print(f"  Time cells (TI > {TI_threshold}): {len(time_cell_indices)}")

    # Get data
    time_field_position_norm = df_time_field_all['time_field_position_norm'].values
    temporal_information_all = df_time_field_all['temporal_information_bits'].values

    # Sort time cells by time field position
    sort_order = np.argsort(time_field_position_norm[time_cell_indices])
    sorted_indices = time_cell_indices[sort_order]

    # Get normalized time fields for sorted cells
    time_fields_sorted = np.vstack(df_time_field_all.loc[sorted_indices, 'time_field_map_norm'].values)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Time cell sequence heatmap (sorted by peak time)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(time_fields_sorted, aspect='auto', cmap='viridis',
                     extent=[0, 1, len(sorted_indices), 0],
                     interpolation='nearest')
    ax1.set_xlabel('Normalized Time in Lap')
    ax1.set_ylabel('Cell # (sorted by time field)')
    ax1.set_title('Time Cell Sequence')
    plt.colorbar(im1, ax=ax1, label='Normalized activity')

    # 2. Distribution of time field positions
    ax2 = axes[0, 1]
    ax2.hist(time_field_position_norm[time_cell_indices], bins=20,
             range=(0, 1), edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Time field position (normalized)')
    ax2.set_ylabel('Number of time cells')
    ax2.set_title(f'Time Field Distribution (TI > {TI_threshold})')
    ax2.set_xlim([0, 1])

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

    # 4. Example time cells
    ax4 = axes[1, 1]
    if len(time_cell_indices) >= 3:
        top_ti_order = np.argsort(temporal_information_all[time_cell_indices])[::-1][:3]
        for i, idx in enumerate(time_cell_indices[top_ti_order]):
            time_bins = np.linspace(0, 1, len(df_time_field_all.loc[idx, 'time_field_map']))
            ax4.plot(time_bins, df_time_field_all.loc[idx, 'time_field_map_norm'],
                     label=f'Cell {idx}, TI={temporal_information_all[idx]:.2f}')
        ax4.set_xlabel('Normalized Time in Lap')
        ax4.set_ylabel('Normalized Activity')
        ax4.set_title('Top 3 Time Cells')
        ax4.legend()

    plt.suptitle(f'Time Cell Analysis (n={len(df_time_field_all)} cells)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
