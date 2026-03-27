#!/usr/bin/env python3
"""
Simplified Grid Response Detection with Shuffle Test and GPU Support

This script analyzes calcium imaging data to detect grids that show significant responses to behavioral events.
It performs statistical testing using both traditional t-tests and a permutation-based shuffle test to identify
responsive regions while controlling for false positives.

Key Features:
============
1. Traditional statistical analysis (paired t-test between baseline and response periods)
2. Shuffle test for robust significance testing (permutation-based null distribution)
3. Sustained response detection (identifies prolonged responses beyond transient peaks)
4. GPU acceleration support via CuPy for faster processing
5. Per-trial response analysis

Shuffle Test Methodology:
========================
The shuffle test creates a null distribution by circularly shifting trial data within a time window,
preserving temporal autocorrelation while breaking the relationship with event timing. This provides
a robust estimate of chance-level responses.

For each grid location:
1. Extract trial segments around each behavioral event (-3s to +5s window)
2. For each shuffle iteration (default 500):
   - Randomly shift each trial segment by a different amount (circular shift)
   - Calculate baseline and response values from shifted data
   - Compute the average response across trials
3. Compare real response to shuffle distribution:
   - p-value: proportion of shuffled responses >= real response
   - 95th percentile: threshold for significance (real response must exceed this)

Sustained Response Detection:
============================
Beyond amplitude-based tests, the script identifies sustained responses where the signal
remains elevated above the shuffle 95th percentile for extended periods (default >0.5s).
This helps distinguish true sustained responses from brief noise fluctuations.

Output Statistics:
=================
For each grid, the analysis provides:
- response_amplitude: Mean response magnitude (response - baseline)
- response_zscore: Standardized response (amplitude / baseline std)
- p_value: Traditional t-test p-value
- shuffle_p_value: Proportion of shuffles exceeding real response
- shuffle_significant: Whether response exceeds shuffle 95th percentile (amplitude OR sustained)
- sustained_significant: Whether grid shows sustained response >0.5s
- max_sustained_sec: Longest duration of sustained response
- response_reliability: Fraction of trials showing positive response
- n_sustained_trials: Number of trials with sustained responses

The shuffle test provides more conservative significance estimates than traditional t-tests,
helping to control false positive rates in high-dimensional imaging data.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
# import matplotlib.pyplot as plt
# plt.ioff()  # Turn off interactive plotting
import cupy as cp

# try:
#     import cupy as cp
#     CUPY_AVAILABLE = True
#     from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
# except ImportError:
#     CUPY_AVAILABLE = False
#     from scipy.ndimage import gaussian_filter1d
#     print("CuPy not available. Shuffle test will run on CPU.")

# from scipy.ndimage import gaussian_filter1d

def align_trials(data, event_frames, 
                 bef=2, aft=4, fs=30, gpu=0):
    win_frames = int((bef+aft)*fs)
    tot_roi = data.shape[0]
    tot_trial = len(event_frames)
    if gpu:
        data = cp.array(data)
        aligned_signal = cp.zeros((tot_roi, tot_trial, win_frames))
        nan=cp.nan
    else:
        aligned_signal = np.zeros((tot_roi, tot_trial, win_frames))
        nan=np.nan
    for t in range(tot_trial):
        curr_trace = data[:, event_frames[t]-int(bef*fs):event_frames[t]+int(aft*fs)]
        if curr_trace.shape[1]<win_frames or event_frames[t]==0:
            aligned_signal[:,t,:]=nan
        else:
            aligned_signal[:,t,:]=curr_trace
    
    return aligned_signal

# def calculate_zscore_f_trace(corrected_traces, event_frames,
#                             baseline_window=(-1, 0), response_window=(0.5, 1.5),
#                             pre_event_window=2, post_event_window=4,
#                             imaging_rate=30.0):

#     orig_shape = corrected_traces.shape
#     orig_ndim  = corrected_traces.ndim

#     # ---------- normalize to ROI-format for computation ----------
#     if orig_ndim == 2:
#         # [n_rois, n_frames]
#         n_rois, n_frames = corrected_traces.shape
#         corrected_2d = corrected_traces
#         is_roi_format = True
#     elif orig_ndim == 3:
#         # [gy, gx, n_frames] -> flatten to [n_rois, n_frames]
#         gy, gx, n_frames = corrected_traces.shape
#         corrected_2d = corrected_traces.reshape(gy * gx, n_frames)
#         is_roi_format = False
#     else:
#         raise ValueError(f"Unsupported corrected_traces shape: {orig_shape}")

#     n_trials = len(event_frames)
#     if n_trials < 2:
#         return None

#     # ---------- build baseline segments ----------
#     baseline_frames = (int(baseline_window[0] * imaging_rate),
#                        int(baseline_window[1] * imaging_rate))

#     trial_aligned_traces = align_trials(
#         corrected_2d,
#         event_frames=event_frames,
#         bef=pre_event_window,
#         aft=post_event_window,
#         gpu=1
#     )  # expected: cupy array [n_rois, n_trials, trial_len]

#     keep_trial_idx = cp.any(cp.isnan(trial_aligned_traces), axis=-1)  # [n_rois, n_trials]
#     trial_aligned_traces[keep_trial_idx] = cp.nan

#     pre_event_frames = int(pre_event_window * imaging_rate)
#     baseline_start = pre_event_frames + baseline_frames[0]
#     baseline_end   = pre_event_frames + baseline_frames[1]

#     # keep on GPU until the end (recommended)
#     baseline_segment_all = trial_aligned_traces[:, :, baseline_start:baseline_end]  # cp, [n_rois, n_trials, len_baseline]

#     # pooled_std_all should end up shape [n_rois,]
#     pooled_std_all = calculate_pooled_std_cp(baseline_segment_all, None)  # <-- depends on your function signature
#     baseline_all_mean = cp.nanmean(baseline_segment_all, axis=(1, 2))     # cp, [n_rois,]

#     # ---------- z-score in 2D ----------
#     denom = pooled_std_all[:, None]                      # [n_rois, 1]
#     corrected_2d = cp.array(corrected_2d)
#     z_2d = (corrected_2d - baseline_all_mean[:, None]) / denom
#     z_2d = cp.where(denom == 0, cp.nan, z_2d)            # avoid div-by-zero

#     # ---------- reshape back to match input ----------
#     if is_roi_format:
#         zscored_traces = z_2d                             # (n_rois, n_frames)
#     else:
#         zscored_traces = z_2d.reshape(orig_shape)         # (gy, gx, n_frames)
        
#     zscored_traces = zscored_traces.get()
#     pooled_std_all = pooled_std_all.get()
#     return zscored_traces, pooled_std_all

import matplotlib.pyplot as plt


def calculate_zscore_f_trace(corrected_traces, event_frames,
                            baseline_window=(-1, 0), response_window=(0.5, 1.5),
                            pre_event_window=2, post_event_window=4,
                            imaging_rate=30.0,
                            debug_plot=False,
                            debug_roi_idx=0):
    """
    Z-score traces using pooled baseline from aligned trials for each ROI.

    Parameters
    ----------
    corrected_traces : np.ndarray
        Shape [n_rois, n_frames] or [gy, gx, n_frames]
    event_frames : array-like
        Event onset frames
    baseline_window : tuple
        Baseline window in seconds relative to event, e.g. (-1, 0)
    response_window : tuple
        Not used here yet, kept for compatibility
    pre_event_window : float
        Seconds before event included in aligned traces
    post_event_window : float
        Seconds after event included in aligned traces
    imaging_rate : float
        Frames/sec
    debug_plot : bool
        If True, make sanity-check plot(s)
    debug_roi_idx : int
        ROI index to plot if ROI-format input; for 3D input this is flattened ROI index

    Returns
    -------
    zscored_traces : np.ndarray
        Same shape as input
    pooled_std_all : np.ndarray
        Shape [n_rois,]
    """

    orig_shape = corrected_traces.shape
    orig_ndim  = corrected_traces.ndim

    # ---------- normalize to ROI-format for computation ----------
    if orig_ndim == 2:
        # [n_rois, n_frames]
        n_rois, n_frames = corrected_traces.shape
        corrected_2d = corrected_traces
        is_roi_format = True
    elif orig_ndim == 3:
        # [gy, gx, n_frames] -> flatten to [n_rois, n_frames]
        gy, gx, n_frames = corrected_traces.shape
        corrected_2d = corrected_traces.reshape(gy * gx, n_frames)
        is_roi_format = False
    else:
        raise ValueError(f"Unsupported corrected_traces shape: {orig_shape}")

    n_trials = len(event_frames)
    if n_trials < 2:
        return None

    # ---------- build baseline segments ----------
    baseline_frames = (int(baseline_window[0] * imaging_rate),
                       int(baseline_window[1] * imaging_rate))

    trial_aligned_traces = align_trials(
        corrected_2d,
        event_frames=event_frames,
        bef=pre_event_window,
        aft=post_event_window,
        gpu=1
    )  # expected: cupy array [n_rois, n_trials, trial_len]

    keep_trial_idx = cp.any(cp.isnan(trial_aligned_traces), axis=-1)  # [n_rois, n_trials]
    trial_aligned_traces = cp.where(
        keep_trial_idx[:, :, None],
        cp.nan,
        trial_aligned_traces
    )

    pre_event_frames = int(pre_event_window * imaging_rate)
    baseline_start = pre_event_frames + baseline_frames[0]
    baseline_end   = pre_event_frames + baseline_frames[1]

    baseline_segment_all = trial_aligned_traces[:, :, baseline_start:baseline_end]  # [n_rois, n_trials, len_baseline]

    pooled_std_all = calculate_pooled_std_cp(baseline_segment_all, None)   # [n_rois,]
    baseline_all_mean = cp.nanmean(baseline_segment_all, axis=(1, 2))      # [n_rois,]

    # ---------- z-score in 2D ----------
    corrected_2d_cp = cp.asarray(corrected_2d)
    denom = pooled_std_all[:, None]                                        # [n_rois, 1]

    z_2d = (corrected_2d_cp - baseline_all_mean[:, None]) / denom
    z_2d = cp.where((denom == 0) | cp.isnan(denom), cp.nan, z_2d)

    # ---------- optional sanity check ----------
    if debug_plot:
        # Re-align z-scored traces using the same events/windows
        z_aligned = align_trials(
            z_2d,
            event_frames=event_frames,
            bef=pre_event_window,
            aft=post_event_window,
            gpu=1
        )  # [n_rois, n_trials, trial_len]

        z_aligned = cp.where(
            keep_trial_idx[:, :, None],
            cp.nan,
            z_aligned
        )

        # 1) strongest mathematical sanity check:
        #    exact pooled baseline samples used for normalization
        baseline_z_direct = z_aligned[:, :, baseline_start:baseline_end]
        baseline_mean_direct = cp.nanmean(baseline_z_direct, axis=(1, 2))   # [n_rois]

        print("Direct pooled baseline mean after z-score:")
        print(f"  max abs across ROIs = {cp.nanmax(cp.abs(baseline_mean_direct)).get():.6g}")
        print(f"  mean abs across ROIs = {cp.nanmean(cp.abs(baseline_mean_direct)).get():.6g}")

        # 2) plot one ROI mean ± SEM across trials
        debug_roi_idx = int(np.clip(debug_roi_idx, 0, z_aligned.shape[0] - 1))
        roi_trials = z_aligned[debug_roi_idx].get()  # [n_trials, trial_len]

        t = np.arange(roi_trials.shape[1]) / imaging_rate - pre_event_window
        mean_trace = np.nanmean(roi_trials, axis=0)
        sem_trace = np.nanstd(roi_trials, axis=0) / np.sqrt(np.sum(~np.isnan(roi_trials), axis=0))

        baseline_mask = (t >= baseline_window[0]) & (t < baseline_window[1])
        roi_baseline_mean = np.nanmean(mean_trace[baseline_mask])

        plt.figure(figsize=(6, 4))
        plt.plot(t, mean_trace, linewidth=2, label=f'ROI {debug_roi_idx}')
        plt.fill_between(t, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3)
        plt.axvline(0, linestyle='--')
        plt.axhline(0, linestyle=':')
        plt.axvspan(baseline_window[0], baseline_window[1], alpha=0.15, label='baseline')
        plt.xlabel('Time from event (s)')
        plt.ylabel('Z-score')
        plt.title(f'Mean aligned z-trace | ROI {debug_roi_idx}\nBaseline mean = {roi_baseline_mean:.4f}')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        # 3) population-average plot across all ROIs
        pop_mean_trace = cp.nanmean(z_aligned, axis=(0, 1)).get()  # pooled over ROI and trial
        pop_baseline_mean = np.nanmean(pop_mean_trace[baseline_mask])

        plt.figure(figsize=(6, 4))
        plt.plot(t, pop_mean_trace, linewidth=2)
        plt.axvline(0, linestyle='--')
        plt.axhline(0, linestyle=':')
        plt.axvspan(baseline_window[0], baseline_window[1], alpha=0.15)
        plt.xlabel('Time from event (s)')
        plt.ylabel('Z-score')
        plt.title(f'Population mean aligned z-trace\nBaseline mean = {pop_baseline_mean:.4f}')
        plt.tight_layout()
        plt.show()

    # ---------- reshape back to match input ----------
    if is_roi_format:
        zscored_traces = z_2d
    else:
        zscored_traces = z_2d.reshape(orig_shape)

    zscored_traces = zscored_traces.get()
    pooled_std_all = pooled_std_all.get()
    return zscored_traces, pooled_std_all    
        
def calculate_pooled_std_cp(pre_segements, post_segements=None):
    if post_segements is not None:
        pre_data = cp.array(pre_segements)
        post_data = cp.array(post_segements)
        n_pre  = pre_data.shape[-1]
        n_post = post_data.shape[-1]
        if pre_data.ndim == 2: # single roi (n_trials, pre/post frames)
            pre_flat = pre_data.flatten()
            post_flat = post_data.flatten()
            # pooled_std = np.sqrt(((n_pre - 1) * np.std(pre_data, ddof=1)**2 + (n_post - 1) * np.std(post_data, ddof=1)**2) / (n_pre + n_post - 2))
        elif pre_data.ndim == 3: # (n_rois, n_trials, pre/post frames)
            # Flatten trials × frames into one axis
            pre_flat  = pre_data.reshape(pre_data.shape[0], -1)   # (n_rois, n_pre_total)
            post_flat = post_data.reshape(post_data.shape[0], -1) # (n_rois, n_post_total)
            # Std per ROI (sample std, ddof=1)
            std_pre  = cp.nanstd(pre_flat,  axis=-1, ddof=1)  # (n_rois,)
            std_post = cp.nanstd(post_flat, axis=-1, ddof=1)  # (n_rois,)
            
        # Vectorized pooled std
        pooled_std = cp.sqrt(
            ((n_pre - 1) * std_pre**2 + (n_post - 1) * std_post**2) / (n_pre + n_post - 2)
        )  # shape (n_rois,) or a single value for single roi
    else:
        pre_data = cp.array(pre_segements)
        n_pre  = pre_data.shape[-1]
        
        if pre_data.ndim == 2: # single roi (n_trials, pre/post frames)
            pre_flat = pre_data.flatten()

        elif pre_data.ndim == 3: # (n_rois, n_trials, pre/post frames)
            # Flatten trials × frames into one axis
            pre_flat  = pre_data.reshape(pre_data.shape[0], -1)   # (n_rois, n_pre_total)
            # Std per ROI (sample std, ddof=1)
            std_pre  = cp.nanstd(pre_flat,  axis=-1, ddof=1)  # (n_rois,)
            
            pooled_std = std_pre

        
    return pooled_std


def quantify_event_response(corrected_traces, event_frames,
                            baseline_window=(-1, 0), response_window=(0, 1.5), # seconds
                            dilation_k = 0,
                            imaging_rate=30.0, shuffle_test=True,
                            shuffle_params={'times': 1000,
                                            'pre_event_window':  2, # seconds
                                            'post_event_window': 4 }
                            
                            ):
    """ calculate event response and optional shuffle test with defined windows
    Args:
        corrected_traces: Either 3D array [grid_y, grid_x, n_frames] or 2D array [n_rois, n_frames]
        baseline_window: (start, end) in seconds relative to event
        response_window: (start, end) in seconds relative to event
    """
    if corrected_traces.ndim == 2:
        # ROI format: [n_rois, n_frames] -> reshape to [n_rois, 1, n_frames]
        n_rois, n_frames = corrected_traces.shape
        # corrected_traces = corrected_traces.reshape(n_rois, 1, n_frames)
        n_grids_y, n_grids_x = n_rois, 1
        is_roi_format = True
        print(f"Detected ROI format: {n_rois} ROIs, {n_frames} frames")
    elif corrected_traces.ndim == 3:
        # Grid format: [grid_y, grid_x, n_frames]
        n_grids_y, n_grids_x, n_frames = corrected_traces.shape
        # for convinience, covert to 2D for calculating response
        corrected_traces = corrected_traces.reshape(n_grids_y*n_grids_x, n_frames)
        n_rois = corrected_traces.shape[0]
        is_roi_format = False
        print(f"Detected grid format: {n_grids_y}x{n_grids_x} grids, {n_frames} frames")
    else:
        raise ValueError(f"Unsupported corrected_traces shape: {corrected_traces.shape}")
    
    n_trials = len(event_frames)
    if n_trials < 2:
        print('no enough valid_events for statistics')
        return None
    
    ## coverting time window to frame window
    baseline_frames = (int(baseline_window[0] * imaging_rate), int(baseline_window[1] * imaging_rate))
    response_frames = (int(response_window[0] * imaging_rate), int(response_window[1] * imaging_rate))
    len_baseline = baseline_frames[1]-baseline_frames[0]
    len_response = response_frames[1]-response_frames[0]
    ## calculate event response
    # extract baseline window segements and response_segment
    # [n_rois, n_trials, window_length]
    baseline_segment_all = np.zeros((n_rois, n_trials, len_baseline))
    response_segment_all = np.zeros((n_rois, n_trials, len_response))
    
    n_valid_event = 0
    
    # calculate roi trial mean profile aligned to event
    trial_aligned_traces = align_trials(corrected_traces,
                                        event_frames = event_frames,
                                        bef=shuffle_params['pre_event_window'],
                                        aft=shuffle_params['post_event_window'],
                                        gpu=1) # [n_rois, n_trials, trial_len]

    # Apply same extreme value filtering to trial_aligned_traces for shuffle test
    exclude_trial_idx = cp.any(cp.isnan(trial_aligned_traces), axis=-1) # [n_rois, n_trials]
    trial_aligned_traces[exclude_trial_idx] = cp.nan
    n_keep_trial = cp.sum(~exclude_trial_idx, axis=-1).get()
    event_aligned_mean = np.nanmean(trial_aligned_traces.get(), axis=1) # [n_rois, n_trials, trial_len]
    
    # Extract baseline and response segments directly from trial_aligned_traces
    # trial_aligned_traces is aligned with event at index = shuffle_pre_frames
    shuffle_pre_frames = int(shuffle_params['pre_event_window'] * imaging_rate)
    baseline_start = shuffle_pre_frames + baseline_frames[0]
    baseline_end = shuffle_pre_frames + baseline_frames[1]
    response_start = shuffle_pre_frames + response_frames[0]
    response_end = shuffle_pre_frames + response_frames[1]

    # Extract segments from trial_aligned_traces (already filtered by keep_trial_idx)
    baseline_segment_all = trial_aligned_traces[:, :, baseline_start:baseline_end].get()  # [n_rois, n_trials, len_baseline]
    response_segment_all = trial_aligned_traces[:, :, response_start:response_end].get()  # [n_rois, n_trials, len_response]

    n_valid_event = int(cp.sum(~exclude_trial_idx[0]).get())  # count valid trials (same for all ROIs in terms of event validity)

    ## calculate std across all baseline or response segments for all rois
    pooled_std_all = calculate_pooled_std_cp(baseline_segment_all, response_segment_all).get()
    baseline_all_mean = np.nanmean(baseline_segment_all, axis=(1,2)) # [n_rois,]
    response_all_mean = np.nanmean(response_segment_all,  axis=(1,2)) # [n_rois,]
    response_amp_all = response_all_mean - baseline_all_mean # [n_rois,]
    response_effect_size_all = response_amp_all/pooled_std_all
    response_ratio_all = response_all_mean / baseline_all_mean

    if  shuffle_test:
        n_shuffle = shuffle_params['times']
        shuffle_pre_sec = shuffle_params['pre_event_window']
        shuffle_post_sec = shuffle_params['post_event_window']

        shuffle_pre_frames = int(shuffle_pre_sec * imaging_rate)
        shuffle_post_frames = int(shuffle_post_sec * imaging_rate)
        trial_length = shuffle_pre_frames + shuffle_post_frames
        # baseline and response indices for trial segments
        baseline_start = shuffle_pre_frames + baseline_frames[0]
        baseline_end   = shuffle_pre_frames + baseline_frames[1]
        response_start = shuffle_pre_frames + response_frames[0]
        response_end   = shuffle_pre_frames + response_frames[1]

        # Filter out ROIs with all-NaN traces to speed up shuffle
        # Check which ROIs have at least some valid data (not all NaN)
        if isinstance(trial_aligned_traces, cp.ndarray):
            roi_has_valid_data = ~cp.all(cp.isnan(trial_aligned_traces), axis=(1, 2))
            valid_roi_indices = cp.where(roi_has_valid_data)[0]
            valid_roi_indices_np = cp.asnumpy(valid_roi_indices)
        else:
            roi_has_valid_data = ~np.all(np.isnan(trial_aligned_traces), axis=(1, 2))
            valid_roi_indices = np.where(roi_has_valid_data)[0]
            valid_roi_indices_np = valid_roi_indices

        n_valid_rois = len(valid_roi_indices)
        print(f"Shuffle test: processing {n_valid_rois}/{n_rois} ROIs with valid data")

        # Extract only valid ROIs for shuffle computation
        valid_trial_aligned = trial_aligned_traces[valid_roi_indices, :, :]

        # shuffle containers - only for valid ROIs
        shuffle_pooled_stds_valid = cp.zeros((n_valid_rois, n_shuffle))
        shuffle_amps_valid = cp.zeros((n_valid_rois, n_shuffle))
        shuffle_ratios_valid = cp.zeros((n_valid_rois, n_shuffle))

        # calculate shuffle for response_amp and response_effect_size_all
        cp.random.seed(42)

        for shuffle_idx in tqdm(range(n_shuffle), desc='Using GPU for shuffle...'):
            # Generate random shifts for all trials at once
            shifts = cp.random.randint(1, trial_length, size=n_trials)

            # Vectorized circular shift via index mapping
            indices = (cp.arange(trial_length)[None, None, :] - shifts[None, :, None]) % trial_length  # (1, n_trials, trial_length)
            indices = cp.broadcast_to(indices, valid_trial_aligned.shape)  # (n_valid_rois, n_trials, trial_length)
            shuffled_segments_valid = cp.take_along_axis(
                valid_trial_aligned,
                indices,
                axis=2
            )  # (n_valid_rois, n_trials, trial_length)

            # Extract baseline and response segments
            shuffled_baseline_segments = shuffled_segments_valid[:, :, baseline_start:baseline_end]
            shuffled_response_segments = shuffled_segments_valid[:, :, response_start:response_end]

            # Calculate means across time dimension
            shuffled_baseline = cp.nanmean(shuffled_baseline_segments, axis=-1)  # (n_valid_rois, n_trials)
            shuffled_response = cp.nanmean(shuffled_response_segments, axis=-1)  # (n_valid_rois, n_trials)
            mean_baseline = cp.nanmean(shuffled_baseline, axis=1)  # (n_valid_rois,)
            mean_response = cp.nanmean(shuffled_response, axis=1)  # (n_valid_rois,)
            # shuffle response amplitude
            shuffle_amps_valid[:, shuffle_idx] = mean_response - mean_baseline  # (n_valid_rois, n_shuffle)
            shuffle_ratios_valid[:, shuffle_idx] = mean_response/mean_baseline

            # Pooled std expects (n_rois, n_trials, frames)
            shuffle_pooled_stds_valid[:, shuffle_idx] = calculate_pooled_std_cp(
                shuffled_baseline_segments,
                shuffled_response_segments
            )  # (n_valid_rois, n_shuffle)

        # Map results back to full ROI arrays with NaN for invalid ROIs
        #containers
        shuffle_amps = np.full((n_rois, n_shuffle), np.nan)
        shuffle_ratios = np.full((n_rois, n_shuffle), np.nan)
        shuffle_pooled_stds = np.full((n_rois, n_shuffle), np.nan)
        
        shuffle_amps[valid_roi_indices_np, :] = cp.asnumpy(shuffle_amps_valid)
        shuffle_ratios[valid_roi_indices_np, :] = cp.asnumpy(shuffle_ratios_valid)
        shuffle_pooled_stds[valid_roi_indices_np, :] = cp.asnumpy(shuffle_pooled_stds_valid)
        shuffle_effect_sizes = shuffle_amps/shuffle_pooled_stds # (n_rois, n_shuffle)
    
    
    roi_info = []
    if is_roi_format:
        for r in range(n_rois):
            if shuffle_test:
                shuff_response_amp = shuffle_amps[r, :]
                shuff_effect_size  = shuffle_effect_sizes[r, :]
                shuffle_ratio = shuffle_ratios[r, :]
            else:
                shuff_response_amp = []
                shuff_effect_size  = []
                shuffle_ratio = []
            
            # for roi format, make sure save all rois'f info to align with the is_soma index    
            # if not np.all(np.isnan(event_aligned_mean[r])):
            roi_dic = {
                        'roi_id': r,
                        'dilation_k': dilation_k,
                        'n_valid_event': n_valid_event,
                        'n_keep_trial': n_keep_trial[r],
                        'response_amplitude': response_amp_all[r],
                        'baseline_mean': baseline_all_mean[r],
                        'response_mean': response_all_mean[r],
                        'effect_size': response_effect_size_all[r],
                        'response_ratio': response_ratio_all[r],
                        'shuff_response_amplitude': shuff_response_amp,
                        'shuff_effect_size': shuff_effect_size,
                        'shuff_response_ratio': shuffle_ratio,
                        'mean_profile': event_aligned_mean[r]
                    }
            
            roi_info.append(roi_dic)
    
    else:
        # reshape reshults back to original roi order
        event_aligned_mean = event_aligned_mean.reshape(n_grids_y, n_grids_x, 
                                                        event_aligned_mean.shape[-1])
        response_amp_all = response_amp_all.reshape(n_grids_y, n_grids_x) # [n_rois,]
        response_effect_size_all = response_effect_size_all.reshape(n_grids_y, n_grids_x)
        response_ratio_all = response_ratio_all.reshape(n_grids_y, n_grids_x)
        n_keep_trial = n_keep_trial.reshape(n_grids_y, n_grids_x)
        
        if shuffle_test:
            shuffle_amps = shuffle_amps.reshape(n_grids_y, n_grids_x, n_shuffle)
            shuffle_effect_sizes = shuffle_effect_sizes.reshape(n_grids_y, n_grids_x, n_shuffle)
            shuffle_ratios = shuffle_ratios.reshape(n_grids_y, n_grids_x, n_shuffle)
        for y in range(n_grids_y):
            for x in range(n_grids_x):
                roi_id = (y, x)  # For grid format, use tuple (grid_y, grid_x) 
                if shuffle_test:
                    shuff_response_amp = shuffle_amps[y, x, :]
                    shuff_effect_size  = shuffle_effect_sizes[y, x, :]
                    shuffle_ratio = shuffle_ratios[y, x, :]
                else:
                    shuff_response_amp = []
                    shuff_effect_size  = []
                    shuffle_ratio = []
                # if not np.all(np.isnan(event_aligned_mean[y, x])): #only save valid rois
                roi_dic = {
                            'roi_id': roi_id,
                            'dilation_k': dilation_k,
                            'n_valid_event': n_valid_event,
                            'n_keep_trial': n_keep_trial[y, x],
                            'response_amplitude': response_amp_all[y, x],
                            'baseline_mean': baseline_all_mean[y, x],
                            'response_mean': response_all_mean[y, x],
                            'effect_size': response_effect_size_all[y, x],
                            'response_ratio': response_ratio_all[y, x],
                            'shuff_response_amplitude': shuff_response_amp,
                            'shuff_effect_size': shuff_effect_size,
                            'shuff_response_ratio': shuffle_ratio,
                            'mean_profile': event_aligned_mean[y, x]
                            
                        }
                roi_info.append(roi_dic)
    
    return pd.DataFrame(roi_info)           
        