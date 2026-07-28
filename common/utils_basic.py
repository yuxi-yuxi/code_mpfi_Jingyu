# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:31:53 2026

@author: Jingyu Cao
"""
import numpy as np
from scipy.stats import sem, median_abs_deviation
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d, percentile_filter
except:
    print('No cupy avaliable')


import smtplib
from email.mime.text import MIMEText
# Email configuration
to_email = "Jingyu.Cao@mpfi.org"
from_email = "midaure@gmail.com"
smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_username = "midlaure@gmail.com"
smtp_password = r"bksb vqmz dcqb mseb"
def send_notification(subject, message, to_email=to_email, from_email=from_email, 
                      smtp_server=smtp_server, smtp_port=smtp_port, smtp_username=smtp_username, smtp_password=smtp_password):
    """
    Sends an email using the provided SMTP server settings.
    """
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Secure the connection
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, [to_email], msg.as_string())
        
def vectorized_roll(arr, shifts, xp=np):
    """
    Vectorized circular roll: roll each row by a different shift amount.

    Parameters:
    -----------
    arr : ndarray (n_rows, n_cols)
        2D array to roll
    shifts : ndarray (n_rows,)
        Shift amount for each row
    xp : module
        numpy or cupy

    Returns:
    --------
    rolled : ndarray (n_rows, n_cols)
        Array with each row circularly shifted
    """
    n_rows, n_cols = arr.shape
    # Create index mapping: (arange - shifts) % n_cols
    indices = (xp.arange(n_cols)[None, :] - shifts[:, None]) % n_cols  # (n_rows, n_cols)
    # Use take_along_axis for efficient gathering
    rolled = xp.take_along_axis(arr, indices, axis=1)
    return rolled

def run_of_ones(mask, axis=-1):
    """
    mask: array-like, shape (n_rois, n_frames) (or any 2D)
          treated as boolean (nonzero/True = 1)
    axis: the axis along which contiguity is defined (default: -1)

    Returns
    -------
    lengths_by_roi : list of 1D np arrays
        lengths_by_roi[i] = lengths of all contiguous-1 runs in ROI i
    """
    x = (mask != 0).astype(np.int8)
    axis = np.core.numeric.normalize_axis_index(axis, x.ndim)

    # move target axis to last
    y = np.moveaxis(x, axis, -1)            # shape: (n_other, L) for 2D
    if y.ndim != 2:
        raise ValueError("This helper expects a 2D array after moveaxis.")

    # diff on padded int array -> transitions are -1/0/+1
    d = np.diff(np.pad(y, ((0,0),(1,1)), constant_values=0), axis=1)

    starts = np.argwhere(d == 1)   # columns: [roi, start]
    ends   = np.argwhere(d == -1)  # columns: [roi, end]
    lengths = ends[:, 1] - starts[:, 1]

    # group lengths by roi (starts/ends are already row-major sorted)
    n_rois = y.shape[0]
    counts = np.bincount(starts[:, 0], minlength=n_rois)  # #runs per roi
    splits = np.cumsum(counts)[:-1]
    lengths_by_row = np.split(lengths, splits)

    return lengths_by_row

def nearest_mapping(A, B):
    '''
    Map each value in A to the index of the closest value in B.

    This function returns, for every element A[i], the index j such that
    |B[j] - A[i]| is minimized (ties are broken in favor of the left neighbor,
    because the comparison uses '<' rather than '<=').
    
    Implementation details:
    - Sorts B once and uses `np.searchsorted` to find insertion positions for A.
    - Compares the candidate neighbor on the left and right to choose the nearest.
    - Converts indices back to the original (unsorted) indexing of B.
    
    Parameters
    ----------
    A : array_like, shape (N,)
        Query values to be matched (e.g., running time stamps).
    B : array_like, shape (M,)
        Reference values to search within (e.g., frame time stamps). Can be unsorted.
    
    Returns
    -------
    nearest_idx : ndarray, shape (N,), dtype=int
        For each A[i], `nearest_idx[i]` is the index into the original B array
        of the element closest to A[i].
    
    Notes
    -----
    - Time complexity is O(M log M + N log M) due to sorting B and searchsorted.
      If you will call this repeatedly with the same B, consider sorting B once
      outside the function and reusing the sorted arrays for speed.

    '''
    # make sure A and B are np arrays
    A = np.array(A)
    B = np.array(B)
    
    # 1) Ensure B is sorted (do this once per session)
    order = np.argsort(B)
    Bs = B[order]
    
    # 2) Insertion positions of each A in sorted B
    j = np.searchsorted(Bs, A, side="left")   # shape (N,), values in [0, M]
    
    # 3) Compare neighbor on the left vs right to get the nearest
    j0 = np.clip(j - 1, 0, len(Bs) - 1)       # left candidate
    j1 = np.clip(j,     0, len(Bs) - 1)       # right candidate
    
    choose_right = np.abs(Bs[j1] - A) < np.abs(Bs[j0] - A)
    nearest_sorted_idx = np.where(choose_right, j1, j0)  # indices into Bs
    
    # 4) Map back to original indices of frame_times
    nearest_idx = order[nearest_sorted_idx]   # indices into original B
    
    return nearest_idx

# def normalize(arr, axis=-1, gpu=0, **kwargs):
#     if gpu :
#         if 'max_perc' in kwargs:
#             arr_max = cp.nanpercentile(arr, kwargs['max_perc'], axis=axis, keepdims=True)
#         else:
#             arr_max = cp.nanmax(arr, axis=axis, keepdims=True)
        
#         if 'min_perc' in kwargs:
#             arr_min = cp.nanpercentile(arr, kwargs['min_perc'], axis=axis, keepdims=True)
#         else:
#             arr_min = cp.nanmin(arr, axis=axis, keepdims=True)
        
#         # Avoid division by zero in case all elements are the same
#         arr_range = arr_max - arr_min
#         arr_range[arr_range == 0] = cp.nan
#     else:   
#         if 'max_perc' in kwargs:
#             arr_max = np.nanpercentile(arr, kwargs['max_perc'], axis=axis, keepdims=True)
#         else:
#             arr_max = np.nanmax(arr, axis=axis, keepdims=True)
        
#         if 'min_perc' in kwargs:
#             arr_min = np.nanpercentile(arr, kwargs['min_perc'], axis=axis, keepdims=True)
#         else:
#             arr_min = np.nanmin(arr, axis=axis, keepdims=True)
    
#         # Avoid division by zero in case all elements are the same
#         arr_range = arr_max - arr_min
#         arr_range[arr_range == 0] = np.nan
    
#     normalized_arr = (arr - arr_min) / arr_range
    
#     return normalized_arr

def normalize(arr, axis=-1, gpu=0, **kwargs):
    xp = cp if gpu else np

    # max
    if 'max_perc' in kwargs:
        arr_max = xp.nanpercentile(arr, kwargs['max_perc'], axis=axis, keepdims=True)
    else:
        arr_max = xp.nanmax(arr, axis=axis, keepdims=True)

    # min
    if 'min_perc' in kwargs:
        arr_min = xp.nanpercentile(arr, kwargs['min_perc'], axis=axis, keepdims=True)
    else:
        arr_min = xp.nanmin(arr, axis=axis, keepdims=True)

    arr_range = arr_max - arr_min
    const_mask = (arr_range == 0)                      # constant slices (including all-zeros)
    all_zero_mask = const_mask & (arr_min == 0) & (arr_max == 0)

    # Make division safe ONLY for all-zero slices; keep NaN behavior for other constant slices
    safe_range = xp.where(all_zero_mask, 1, arr_range)  # all-zero -> divide by 1; others unchanged
    safe_range = xp.where(const_mask & ~all_zero_mask, xp.nan, safe_range)  # non-zero constant -> NaN

    normalized_arr = (arr - arr_min) / safe_range

    # For all-zero slices, force output to 0
    normalized_arr = xp.where(all_zero_mask, 0, normalized_arr)

    return normalized_arr

def zero_padding(vector, max_length):
    new_vector = np.zeros(max_length)
    if len(vector)<max_length:
        l = len(vector)
        new_vector[:l] = vector
    else:
        new_vector = vector[:max_length]
    return new_vector

def trace_filter(trace_ori, n_sd=5, fix_thresh=None, return_outlier=False):
    """
    Filter extreme values in a 1D trace based on standard deviation threshold.

    Replaces values exceeding mean ± n_sd * std with interpolated neighbors.
    Also handles NaN and Inf values.

    Parameters
    ----------
    trace : 1D array
        Input trace (must be 1D)
    n_sd : float
        Number of standard deviations for threshold (default: 5)

    Returns
    -------
    1D array
        Filtered trace with extreme values replaced
    """
    if trace_ori.ndim != 1:
        raise ValueError("trace_filter_sd only accepts 1D arrays")

    trace = trace_ori.copy().astype(float)
    
    # Get valid (finite) values for statistics
    valid_mask = np.isfinite(trace)
    if valid_mask.sum() < 2:
        if return_outlier:
            return trace, np.nan
        else:
            return trace
    
    if fix_thresh is None: # use n_std thresholding
        mean_val = np.nanmean(trace)
        std_val = np.nanstd(trace)
    
        if std_val == 0:
            return trace
    
        # Define threshold bounds
        lower_bound = mean_val - n_sd * std_val
        upper_bound = mean_val + n_sd * std_val
    
        # Find outliers: NaN, Inf, or beyond threshold
        outlier_mask = ~valid_mask | (trace < lower_bound) | (trace > upper_bound)
        outlier_indices = np.where(outlier_mask)[0]
    else: # ignore n_std, use a fix value for thresholding
        outlier_mask = ~valid_mask | (np.abs(trace) > fix_thresh)
        outlier_indices = np.where(outlier_mask)[0]

    # Replace outliers with interpolated neighbor values
    # for idx in outlier_indices:
        # # Find left valid neighbor
        # left_val = np.nan
        # for i in range(idx - 1, -1, -1):
        #     if not outlier_mask[i]:
        #         left_val = trace[i]
        #         break

        # # Find right valid neighbor
        # right_val = np.nan
        # for i in range(idx + 1, len(trace)):
        #     if not outlier_mask[i]:
        #         right_val = trace[i]
        #         break

        # # Replace with average of neighbors, or single neighbor if only one exists
        # if np.isfinite(left_val) and np.isfinite(right_val):
        #     trace[idx] = (left_val + right_val) / 2
        # elif np.isfinite(left_val):
        #     trace[idx] = left_val
        # elif np.isfinite(right_val):
        #     trace[idx] = right_val
        
        # Replace with np.nan
        # trace[idx] = np.nan
    trace[outlier_mask] = np.nan
    if return_outlier:
        return trace, trace_ori[outlier_mask]
    else:
        return trace


# def trace_filter(dFF, thresh=99.999):
#     """
#     One-pass filter for invalid outliers in a trace using only valid neighbors.

#     Invalid = NaN, Inf, or extreme values beyond ±threshold.
#     Replacements use left/right neighbors only if they are valid.

#     Note: Consider using trace_filter_sd() for simpler SD-based filtering.
#     """

#     dFF = dFF.copy()
#     if dFF.ndim == 1:
#         dFF = dFF[np.newaxis, :]

#     n_rois, n_frames = dFF.shape

#     # Compute symmetric extreme threshold
#     finite_vals = dFF[np.isfinite(dFF)]
#     if finite_vals.size == 0:
#         return dFF  # avoid crash if all NaNs/inf

#     extreme_threshold = np.nanpercentile(finite_vals, thresh)

#     # Mark invalid values
#     invalid_mask = (
#         np.isnan(dFF) | np.isinf(dFF) |
#         (dFF > extreme_threshold) | (dFF < -extreme_threshold)
#     )

#     # Prepare left and right neighbors
#     left = np.roll(dFF, shift=1, axis=1)
#     right = np.roll(dFF, shift=-1, axis=1)
#     left[:, 0] = np.nan
#     right[:, -1] = np.nan

#     # Check neighbor validity
#     left_valid = (
#         np.isfinite(left) &
#         (left <= extreme_threshold) &
#         (left >= -extreme_threshold)
#     )
#     right_valid = (
#         np.isfinite(right) &
#         (right <= extreme_threshold) &
#         (right >= -extreme_threshold)
#     )

#     # Build replacement array
#     replacement = np.full_like(dFF, np.nan)
#     both_valid = left_valid & right_valid
#     only_left = left_valid & ~right_valid
#     only_right = right_valid & ~left_valid

#     replacement[both_valid] = (left[both_valid] + right[both_valid]) / 2.0
#     replacement[only_left] = left[only_left]
#     replacement[only_right] = right[only_right]

#     # Apply replacements to invalid entries
#     dFF[invalid_mask] = replacement[invalid_mask]

#     return dFF.squeeze()

def nd_to_list(arr):
    """
    Convert a numpy / cupy / xarray array to nested Python lists of float32.

    - If `arr` is None (or you deliberately pass something falsey), return [].
    - Otherwise, cast to float32 to keep Parquet footprint small and call .tolist().
    """
    if arr is None:
        return []          # no data → empty list
    return arr.astype("float32").tolist()

def cp_sem(arr, axis=1, nan_policy="propagate"):
    """
    Standard error of the mean along a given axis for a CuPy array.

    Parameters
    ----------
    arr : cp.ndarray
    axis : int
        Axis along which to compute the SEM (default 1).
    nan_policy : {'propagate', 'omit'}
        - 'propagate': return NaN if any NaNs are present in the slice
        - 'omit'     : ignore NaNs (matches scipy.stats.sem)

    Returns
    -------
    cp.ndarray
        SEM values with the specified axis collapsed.
    """
    if nan_policy == "omit":
        # sample s.d. with ddof=1, ignoring NaNs
        sd   = cp.nanstd(arr, axis=axis, ddof=1)
        n    = cp.count_nonzero(~cp.isnan(arr), axis=axis)
    else:  # 'propagate'
        sd   = cp.std(arr, axis=axis, ddof=1)
        n    = arr.shape[axis]

    return sd / cp.sqrt(n)

def robust_zscore(data, axis=-1, keepdims=False, gpu=1):
    """
    Compute robust z-scores using MAD along an axis.

    Parameters
    ----------
    data : array-like (NumPy or CuPy)
    axis : int or tuple, default None
        Axis along which to compute z-scores.
    keepdims : bool
        If True, retains reduced dimensions in the result.
    gpu : 1 = use CuPy, 0 = use NumPy

    Returns
    -------
    z : NumPy or CuPy array
        Robust z-score normalized array.
    """
    if gpu == 1:
        xp = cp
        data = xp.asarray(data)

        # always keepdims=True internally so broadcasting works
        median = xp.nanmedian(data, axis=axis, keepdims=True)

        mad = xp.nanmedian(xp.abs(data - median), axis=axis, keepdims=True)
        # scale='normal' → std ≈ MAD / 0.67449
        mad = mad / 0.67449
    else:
        xp = np
        data = xp.asarray(data)

        # same idea: keepdims=True internally
        median = xp.nanmedian(data, axis=axis, keepdims=True)
        mad = median_abs_deviation(
            data, axis=axis, scale='normal', nan_policy='omit', keepdims=True
        )

    # avoid division by zero *per slice*
    mad_safe = xp.where(mad == 0, xp.nan, mad)
    z = (data - median) / mad_safe
    # replace NaNs (where mad==0) with 0
    # z = xp.nan_to_num(z, nan=0.0)

    # now handle keepdims for the *output*
    # if not keepdims and axis is not None:
    #     z = xp.squeeze(z, axis=axis)

    return z
