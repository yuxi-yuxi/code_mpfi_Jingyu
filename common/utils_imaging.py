# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:31:53 2026

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter as sci_gaussian_filter
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d, percentile_filter
from tqdm import tqdm

def rolling_min(array,win):
    length = array.shape[-1]
    half_win = int(np.ceil(win/2))
    array_padding = np.hstack((array[:,:half_win], array, array[:,-half_win:length]))
    output = np.array([np.min(array_padding[:,i:i+win], axis=1) for i in range(half_win,length+half_win)]).T
    return output

def rolling_max(array,win):
    length = array.shape[-1]
    half_win = int(np.ceil(win/2))
    array_padding = np.hstack((array[:,:half_win], array, array[:,-half_win:length]))
    output = np.array([np.max(array_padding[:,i:i+win], axis=1) for i in range(half_win,length+half_win)]).T
    return output

def get_dff(F_array, win=1800, sigma=300, gpu=0, return_baseline=0, time_axis=-1): # by default (what has been used in matlab pipeline, window=60s, sigma=10s)
    '''
    calculate dFF for each ROI
    
    F: raw F signal, whole session
    win: window used for calculate baseline
    sigma: for gaussian filtering
    
    return dFF
    
    '''
    # F_array = np.asarray(F_array)
    if F_array.ndim == 1:
        F_array = F_array[np.newaxis, :] 
        
    if gpu ==0:
        baseline = sci_gaussian_filter(F_array, sigma, axes=time_axis)
        # baseline = gaussian_filter1d(F, sigma)
        baseline = rolling_min(baseline,win)
        baseline = rolling_max(baseline,win)
    elif gpu ==1:
        F_array = cp.array(F_array)
        # baseline = gaussian_filter(F_array, sigma, axis=time_axis, gpu=1)
        # baseline = rolling_filter(baseline, 'min', win, axis=time_axis, gpu=1)
        # baseline = rolling_filter(baseline, 'max', win, axis=time_axis, gpu=1)
        baseline = gaussian_filter1d(F_array, sigma=sigma, axis=time_axis)
        baseline = minimum_filter1d(baseline, size=win, axis=time_axis)
        baseline = maximum_filter1d(baseline, size=win, axis=time_axis)
    dff = (F_array-baseline)/baseline
    
    if return_baseline:
        return dff, baseline
    else:    
        return dff

def percentile_dff_abs(F_array, window_size=1800, q=10, t_axis=-1, return_baseline=0):
    """
    Calculate dF/F using percentile baseline with absolute value normalization.

    Parameters
    ----------
    F_array : array_like
        Raw fluorescence array, shape (n_rois, n_frames) or (n_frames,)
    window_size : int
        Window size for percentile filter (default: 1800 frames = 60s at 30Hz)
    q : float
        Percentile for baseline (default: 10)
    t_axis : int
        Time axis (default: -1, last axis)
    return_baseline : bool
        Whether to return baseline along with dF/F

    Returns
    -------
    dff : ndarray
        Delta F over F (normalized by absolute baseline)
    baseline : ndarray (optional)
        Baseline fluorescence if return_baseline=True
    """
    array = cp.asarray(F_array)

    # For 2D arrays, size must match array dimensions
    # size=(1, window_size) applies filter only along time axis
    if array.ndim == 2:
        if t_axis == -1 or t_axis == 1:
            filter_size = (1, window_size)
        else:
            filter_size = (window_size, 1)
    else:
        filter_size = window_size

    baseline = percentile_filter(array, q, size=filter_size)
    epsilon = 1e-9

    dff = (array - baseline) / (cp.abs(baseline) + epsilon)

    return (dff.get(), baseline.get()) if return_baseline else dff.get()


# def percentile_dff(F_array, window_size=1800, q=10, t_axis=-1, return_baseline=0):
#     """
#     Calculate dF/F using percentile baseline.

#     Parameters
#     ----------
#     F_array : array_like
#         Raw fluorescence array, shape (n_rois, n_frames) or (n_frames,)
#     window_size : int
#         Window size for percentile filter (default: 1800 frames = 60s at 30Hz)
#     q : float
#         Percentile for baseline (default: 10)
#     t_axis : int
#         Time axis (default: -1, last axis)
#     return_baseline : bool
#         Whether to return baseline along with dF/F

#     Returns
#     -------
#     dff : ndarray
#         Delta F over F
#     baseline : ndarray (optional)
#         Baseline fluorescence if return_baseline=True
#     """
#     array = cp.asarray(F_array)

#     # For 2D arrays, size must match array dimensions
#     # size=(1, window_size) applies filter only along time axis
#     if array.ndim == 2:
#         if t_axis == -1 or t_axis == 1:
#             filter_size = (1, window_size)
#         else:
#             filter_size = (window_size, 1)
#     else:
#         filter_size = window_size

#     baseline = percentile_filter(array, q, size=filter_size)
#     epsilon = 0.1
    
#     baseline = baseline + epsilon
#     dff = (array - baseline) / baseline

#     return (dff.get(), baseline.get()) if return_baseline else dff.get()


# def percentile_dff(
#     F_array,
#     window_size=9000,  # changed from 1800, 22 Jan 2026
#     q=20,
#     t_axis=-1,
#     return_baseline=False,
# ):
#     """
#     Calculate dF/F using a running percentile baseline, with a per-ROI denominator floor.

#     Pipeline
#     --------
#     1) baseline b(t) = running percentile_filter(F, q, window_size) along the time axis
#     2) per-trace/ROI floor Ffloor = floor_q-th percentile of baseline over time,
#        computed ONLY over baseline > 0
#     3) baseline <- max(baseline, Ffloor) (elementwise clamp), per ROI
#     4) dff = (F - baseline) / baseline
#     5) If an ROI has NO positive baseline values at all, Ffloor becomes NaN;
#        in that case, the ENTIRE dff trace (and baseline, if returned) is set to NaN.

#     Parameters
#     ----------
#     F_array : array_like
#         Fluorescence array. Shape can be (n_rois, n_frames) or (n_frames,).
#     window_size : int
#         Window size (in frames) for the percentile filter.
#     q : float
#         Percentile for baseline estimation (e.g., 10).
#     t_axis : int
#         Time axis for 2D input. Use 1 or -1 for (n_rois, n_frames),
#         or 0 for (n_frames, n_rois). Ignored for 1D input.
#     return_baseline : bool
#         If True, return (dff, baseline). Otherwise return dff only.
#     floor_q : float
#         Percentile used to compute the floor from positive baseline values (default 1).

#     Returns
#     -------
#     dff : ndarray
#         dF/F, same shape as F_array.
#     baseline : ndarray, optional
#         Baseline used for normalization, same shape as F_array (if return_baseline=True).
#     """
#     array = cp.asarray(F_array)
    
#     # filter negative values
#     array_inf_filled = cp.where(array<0, cp.inf, array) # only for baseline calculation
#     array[array<0] = 0 # set original array negative values to zero for dff calculation
    
    
#     # Determine filter size so the percentile filter runs only along time axis
#     if array.ndim == 1:
#         # t_axis_eff = 0
#         filter_size = window_size
#     elif array.ndim == 2:
#         if t_axis in (-1, 1):
#             # t_axis_eff = 1
#             filter_size = (1, window_size)
#         elif t_axis == 0:
#             # t_axis_eff = 0
#             filter_size = (window_size, 1)
#         else:
#             raise ValueError("For 2D input, t_axis must be 0, 1, or -1.")
#     else:
#         raise ValueError("F_array must be 1D or 2D.")
        
#     baseline = percentile_filter(array_inf_filled, q, size=filter_size)
#     epsilon = 0.1
    
#     # baseline = cp.maximum(1, baseline)
    
#     dff = (array - baseline) / (baseline + epsilon)
    
#     if return_baseline:
#         return dff.get(), baseline.get()

#     return dff.get()

def percentile_dff(
    F_array,
    window_size=9000,  # changed from 1800, 22 Jan 2026
    q=20,
    t_axis=-1,
    return_baseline=False,
):
    """
    Calculate dF/F using a running percentile baseline, with a per-ROI denominator floor.

    Pipeline
    --------
    1) baseline b(t) = running percentile_filter(F, q, window_size) along the time axis
    2) per-trace/ROI floor Ffloor = floor_q-th percentile of baseline over time,
       computed ONLY over baseline > 0
    3) baseline <- max(baseline, Ffloor) (elementwise clamp), per ROI
    4) dff = (F - baseline) / baseline
    5) If an ROI has NO positive baseline values at all, Ffloor becomes NaN;
       in that case, the ENTIRE dff trace (and baseline, if returned) is set to NaN.

    Parameters
    ----------
    F_array : array_like
        Fluorescence array. Shape can be (n_rois, n_frames) or (n_frames,).
    window_size : int
        Window size (in frames) for the percentile filter.
    q : float
        Percentile for baseline estimation (e.g., 10).
    t_axis : int
        Time axis for 2D input. Use 1 or -1 for (n_rois, n_frames),
        or 0 for (n_frames, n_rois). Ignored for 1D input.
    return_baseline : bool
        If True, return (dff, baseline). Otherwise return dff only.
    floor_q : float
        Percentile used to compute the floor from positive baseline values (default 1).

    Returns
    -------
    dff : ndarray
        dF/F, same shape as F_array.
    baseline : ndarray, optional
        Baseline used for normalization, same shape as F_array (if return_baseline=True).
    """
    F_array = np.asarray(F_array)
    original_shape = F_array.shape
    was_1d = F_array.ndim == 1

    # Ensure 2D array for consistent processing
    if was_1d:
        F_array = F_array[np.newaxis, :]

    # Normalize t_axis for 2D
    if t_axis == -1:
        t_axis = 1

    # Identify valid ROIs (not all NaN along time axis)
    if t_axis == 1:
        valid_mask = ~np.all(np.isnan(F_array), axis=1)
    else:  # t_axis == 0
        valid_mask = ~np.all(np.isnan(F_array), axis=0)

    # Initialize output arrays with NaN
    dff_out = np.full(F_array.shape, np.nan, dtype=np.float32)
    baseline_out = np.full(F_array.shape, np.nan, dtype=np.float32)

    # Only process valid ROIs
    if np.any(valid_mask):
        if t_axis == 1:
            valid_data = F_array[valid_mask, :]
        else:
            valid_data = F_array[:, valid_mask]

        array = cp.asarray(valid_data)

        # Determine filter size so the percentile filter runs only along time axis
        if array.ndim == 1:
            filter_size = window_size
        elif array.ndim == 2:
            if t_axis == 1:
                filter_size = (1, window_size)
            elif t_axis == 0:
                filter_size = (window_size, 1)
            else:
                raise ValueError("For 2D input, t_axis must be 0, 1, or -1.")
        else:
            raise ValueError("F_array must be 1D or 2D.")

        baseline = percentile_filter(array, q, size=filter_size, mode='reflect')
        epsilon = 1e-8

        dff = (array - baseline) / (baseline + epsilon)

        # Place results back into output arrays
        if t_axis == 1:
            dff_out[valid_mask, :] = dff.get()
            baseline_out[valid_mask, :] = baseline.get()
        else:
            dff_out[:, valid_mask] = dff.get()
            baseline_out[:, valid_mask] = baseline.get()

    # Restore original shape if input was 1D
    if was_1d:
        dff_out = dff_out.squeeze(axis=0)
        baseline_out = baseline_out.squeeze(axis=0)

    if return_baseline:
        return dff_out, baseline_out

    return dff_out


def nanpercentile_filter(input, percentile, size, output=None, mode="reflect", cval=0.0, origin=0):
    """
    NaN-aware percentile filter on GPU. NaN values are ignored in percentile calculation.

    Args:
        input: cupy.ndarray
        percentile: float (0-100)
        size: int or tuple of int
        output: output array (optional)
        mode: boundary mode ('reflect', 'constant', 'nearest', 'mirror', 'wrap')
        cval: constant value for 'constant' mode
        origin: filter origin

    Returns:
        cupy.ndarray: filtered array
    """
    from cupyx.scipy.ndimage import _filters_core

    percentile = float(percentile)
    if percentile < 0.0:
        percentile += 100.0
    if percentile < 0.0 or percentile > 100.0:
        raise RuntimeError('invalid percentile')

    # Create footprint from size
    input = cp.asarray(input)
    if np.isscalar(size):
        size = (size,) * input.ndim
    footprint = cp.ones(size, dtype=bool)

    origins, int_type = _filters_core._check_nd_args(input, footprint, mode, origin, 'footprint')
    if footprint.size == 0:
        return cp.zeros_like(input)

    filter_size = int(footprint.sum())
    offsets = _filters_core._origins_to_offsets(origins, footprint.shape)

    kernel = _get_nan_percentile_kernel(filter_size, percentile, mode, footprint.shape,
                                        offsets, float(cval), int_type)
    return _filters_core._call_kernel(kernel, input, footprint, output, weights_dtype=bool)


_NAN_SHELL_SORT = '''
__device__ void sort(X *array, int size) {
    int gap = %d;
    while (gap > 1) {
        gap /= 3;
        for (int i = gap; i < size; ++i) {
            X value = array[i];
            int j = i - gap;
            while (j >= 0 && value < array[j]) {
                array[j + gap] = array[j];
                j -= gap;
            }
            array[j + gap] = value;
        }
    }
}'''


def _get_shell_gap(filter_size):
    gap = 1
    while gap < filter_size:
        gap = 3 * gap + 1
    return gap


@cp._util.memoize(for_each_device=True)
def _get_nan_percentile_kernel(filter_size, percentile, mode, w_shape, offsets, cval, int_type):
    from cupyx.scipy.ndimage import _filters_core

    gap = _get_shell_gap(filter_size)
    sorter = _NAN_SHELL_SORT % gap

    # Kernel: collect non-NaN values, sort them, compute percentile
    pre = 'int iv = 0;\nX values[%d];' % filter_size

    # Only collect non-NaN values - {value} is a placeholder replaced by _generate_nd_kernel
    found = 'if (!isnan({value})) {{ values[iv++] = {value}; }}'

    # After collection: sort valid values and compute percentile
    pct_frac = percentile / 100.0
    post = '''if (iv == 0) {
    y = (Y)(0.0f / 0.0f);
} else {
    sort(values, iv);
    float idx_f = %.6ff * (float)(iv - 1);
    int idx_lo = (int)idx_f;
    int idx_hi = idx_lo + 1;
    if (idx_hi >= iv) idx_hi = iv - 1;
    float frac = idx_f - (float)idx_lo;
    y = (Y)(values[idx_lo] + frac * (values[idx_hi] - values[idx_lo]));
}''' % pct_frac

    return _filters_core._generate_nd_kernel(
        'nan_percentile_%d_%d' % (filter_size, int(percentile * 100)),
        pre, found, post,
        mode, w_shape, int_type, offsets, cval, preamble=sorter)


def nanpercentile_dff(
    F_array,
    window_size=9000,
    q=10,
    t_axis=-1,
    return_baseline=False,
):
    """
    Calculate dF/F using a running percentile baseline, ignoring NaN values.

    GPU-based using custom nanpercentile_filter.

    Parameters
    ----------
    F_array : array_like
        Fluorescence array. Shape can be (n_rois, n_frames) or (n_frames,).
    window_size : int
        Window size (in frames) for the percentile filter.
    q : float
        Percentile for baseline estimation (e.g., 10).
    t_axis : int
        Time axis. Use 1 or -1 for (n_rois, n_frames), or 0 for (n_frames, n_rois).
    return_baseline : bool
        If True, return (dff, baseline). Otherwise return dff only.

    Returns
    -------
    dff : ndarray
        dF/F, same shape as F_array.
    baseline : ndarray, optional
        Baseline used for normalization (if return_baseline=True).
    """
    array = cp.asarray(F_array, dtype=cp.float32)

    # Filter negative values to NaN
    array = cp.where(array < 0, cp.nan, array)

    # Determine filter size
    if array.ndim == 1:
        filter_size = window_size
    elif array.ndim == 2:
        if t_axis in (-1, 1):
            filter_size = (1, window_size)
        elif t_axis == 0:
            filter_size = (window_size, 1)
        else:
            raise ValueError("For 2D input, t_axis must be 0, 1, or -1.")
    else:
        raise ValueError("F_array must be 1D or 2D.")

    baseline = nanpercentile_filter(array, q, size=filter_size)

    epsilon = 0.1
    dff = (array - baseline) / (baseline + epsilon)

    if return_baseline:
        return dff.get(), baseline.get()

    return dff.get()


def align_trials(data, alignment, beh, bef=2, aft=4, gpu=0, fs=30):
    if gpu:
       data = cp.array(data) 
    if data.ndim == 1:
        data = data[np.newaxis, :] 
    
    win_frames = int((bef+aft)*fs)
    tot_roi = data.shape[0]
        
    if alignment ==  'run':
        run_onsets = beh['run_onset_frames']
        tot_trial = len(run_onsets)
        if gpu:
            aligned_signal = cp.zeros((tot_roi, tot_trial, win_frames))
            for t in range(tot_trial):
                curr_trace = data[:, run_onsets[t]-int(bef*fs):run_onsets[t]+int(aft*fs)]
                if curr_trace.shape[1]<win_frames or run_onsets[t]==0:
                    aligned_signal[:,t,:]=cp.nan
                else:
                    aligned_signal[:,t,:]=curr_trace
        else:
            aligned_signal = np.zeros((tot_roi, tot_trial, win_frames))
            for t in range(tot_trial):
                curr_trace = data[:, run_onsets[t]-int(bef*fs):run_onsets[t]+int(aft*fs)]
                if curr_trace.shape[1]<win_frames or run_onsets[t]==0:
                    aligned_signal[:,t,:]=np.nan
                else:
                    aligned_signal[:,t,:]=curr_trace
    
    if alignment == 'rew':
        # # temp: 20250417 
        rewards = beh['reward_frames']
        # rewards = beh.blackout_frames # for none rewarded trials, use black out on time as alignment
        tot_trial = len(rewards)
        if gpu:
            aligned_signal = cp.zeros((tot_roi, tot_trial, win_frames))
            for t in range(tot_trial):
                curr_trace = data[:, rewards[t]-int(bef*fs):rewards[t]+int(aft*fs)]
                if curr_trace.shape[1]<win_frames or rewards[t]==0:
                    aligned_signal[:,t,:]=cp.nan
                else:
                    aligned_signal[:,t,:]=curr_trace
        else:
            aligned_signal = np.zeros((tot_roi, tot_trial, win_frames))
            for t in range(tot_trial):
                curr_trace = data[:, rewards[t]-int(bef*fs):rewards[t]+int(aft*fs)]
                if curr_trace.shape[1]<win_frames or rewards[t]==0:
                    aligned_signal[:,t,:]=np.nan
                else:
                    aligned_signal[:,t,:]=curr_trace
                
    if alignment == 'cue':
        cues = beh.start_cue_frames # for none rewarded trials, use black out on time as alignment
        tot_trial = len(cues)
        aligned_signal = np.zeros((tot_roi, tot_trial, win_frames))
        for t in range(tot_trial):
            curr_trace = data[:, cues[t]-int(bef*fs):cues[t]+int(aft*fs)]
            if curr_trace.shape[1]<win_frames or cues[t]==0:
                aligned_signal[:,t,:]=np.nan
            else:
                aligned_signal[:,t,:]=curr_trace
    
    return aligned_signal.squeeze()

def F_extraction(movie, chunk_size, rois=None, gpu=1):
    """
    Extracts fluorescence traces by averaging pixel values inside ROIs.

    Parameters
    ----------
    movie : np.ndarray (n_frames, height, width)
        Movie in CPU memory (np.memmap is fine).
    suite2p_ops : dict
        Dict produced by Suite2p (must contain 'xrange' and 'yrange' if you did
        registration).
    chunk_size : int
        Number of frames to move through the movie at once.
    rois : np.ndarray | None
        Boolean mask(s) with shape (n_roi, H, W) *or* (H, W).
        If None or empty, the whole field of view is treated as one ROI.
    gpu : bool (1/0)
        Whether to use CuPy.

    Returns
    -------
    roi_traces : cp.ndarray or np.ndarray (n_roi, n_frames)
        Sum of pixel values in each ROI for every frame, divided by pixel count
        so that each trace is a mean‑fluorescence signal.
    """
    
    if rois is None or (isinstance(rois, np.ndarray) and rois.size == 0):
        # Whole‑field mask → a single ROI
        rois = np.ones((1, movie.shape[1], movie.shape[2]), dtype=bool)
    else:
        rois = np.asarray(rois, dtype=bool)
    
    if rois.ndim ==2:
        rois  = rois[np.newaxis, :]
        
    # if using gpu
    if gpu:
        rois = cp.asarray(rois)
        # extraction
        roi_flat = rois.reshape(rois.shape[0], -1).astype(np.float32)
        roi_pixel_counts = roi_flat.sum(axis=1)
        
        nframes = movie.shape[0]
        n_rois = roi_flat.shape[0]
        # roi_flat = cp.array(roi_flat)
        # roi_pixel_counts = cp.array(roi_pixel_counts)
        roi_signal_sum_gpu = cp.empty((n_rois, nframes), dtype=cp.float32)
        
        # Process the movie in chunks with a progress bar
        for start in tqdm(range(0, nframes, chunk_size), desc="Processing chunks"):
            
            end = min(start + chunk_size, nframes)
            # Load a chunk from the memmap and reshape it to 2D (frames, 512*512)
            chunk = movie[start:end].reshape(end - start, -1)
            # Transfer the chunk to the GPU
            chunk_gpu = cp.asarray(chunk)
            # Compute the dot product: result shape is (chunk_frames, n_rois)
            result_chunk = cp.dot(chunk_gpu, roi_flat.T)
            # Synchronize to ensure GPU operations are complete before writing
            cp.cuda.Stream.null.synchronize()
            # Write the computed results directly into the pre-allocated container
            roi_signal_sum_gpu[:, start:end] = result_chunk.T
            
    #if not use gpu, then use dask parallel iteration to accelarate    
    return roi_signal_sum_gpu/roi_pixel_counts[:, None]

def fov_trace_extraction(mov, roi_mask, path_result, edge_pixels = 16):
    # extract FOV traces
    T, H, W = mov.shape
    roi_mask = roi_mask.astype(bool)
    valid_area_mask = np.zeros((H, W), dtype=bool)
    valid_area_mask[edge_pixels : H - edge_pixels, 
                    edge_pixels : W - edge_pixels] = True
    final_mask = roi_mask & valid_area_mask
    # fov_trace = mov[:, final_mask].mean(axis=1)
    fov_trace = F_extraction(mov[:, :, :], chunk_size=5000, rois=final_mask, gpu=1).get()
    fov_trace = fov_trace.squeeze()
    np.save(path_result, fov_trace)
    
    return fov_trace

    
def load_bin_file(data_path, file_name, n_frames=2000, height=512, width=512):
    """"
    Load a registered dLight imaging movie from a binary file.

    This function loads a 3D imaging stack stored in raw binary format (typically
    produced by Suite2p or similar tools). It uses memory mapping for efficient
    access and converts the data to float32 for further analysis.

    Parameters:
        data_path (str): Directory containing the binary movie file.
        file_name (str): Name of the binary movie file (e.g., 'data.bin').
        n_frames (int, optional): Number of frames in the movie. Default is 2000.
        height (int, optional): Image height in pixels. Default is 512.
        width (int, optional): Image width in pixels. Default is 512.

    Returns:
        np.ndarray: Loaded movie as a NumPy array of shape (n_frames, height, width),
                    with dtype float32.
    """
    
    # Build the full path to the binary data file.
    data_bin = Path(data_path)/file_name
    
    # Use a memory map to load the binary data.
    reg_data = np.memmap(data_bin, mode='r', dtype='int16', shape=(n_frames, height, width))
    
    # Convert to float32.
    mov = reg_data.astype('float32')
    return mov