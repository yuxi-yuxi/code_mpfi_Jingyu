# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 21:24:47 2025

@author: Jingyu Cao
"""
from scipy.stats import sem, median_abs_deviation
from scipy.ndimage import median_filter, gaussian_filter1d
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
except:
    print('CUPY is not installed')
import matplotlib.pyplot as plt

def cp_median_abs_deviation(x, axis=None, nan_policy='omit'):
    """
    Compute the Median Absolute Deviation (MAD) of a CuPy array.

    Parameters
    ----------
    x : cp.ndarray
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes along which the MAD is computed.
        If None, the MAD is computed over the entire array.
    nan_policy : {'propagate', 'omit'}, optional
        Defines how to handle NaNs:
        - 'propagate': if NaN is present, result will be NaN
        - 'omit': ignore NaNs in the computation

    Returns
    -------
    mad : cp.ndarray
        The median absolute deviation of the array elements.
    """
    if not isinstance(x, cp.ndarray):
        x = cp.array(x)

    if nan_policy not in {'propagate', 'omit'}:
        raise ValueError("nan_policy must be 'propagate' or 'omit'")

    if nan_policy == 'omit':
        med = cp.nanmedian(x, axis=axis, keepdims=True)
        abs_dev = cp.abs(x - med)
        mad = cp.nanmedian(abs_dev, axis=axis)
    else:  # propagate
        med = cp.median(x, axis=axis, keepdims=True)
        abs_dev = cp.abs(x - med)
        mad = cp.median(abs_dev, axis=axis)

    return mad

def std_filter_perROI(vector, factor=2, return_sigma=False, gpu=0):
    """
    Suppress values below a robust threshold (based on MAD) in a CuPy or NumPy array,
    optionally Gaussian-smoothing the signal beforehand.

    Parameters
    ----------
    vector : np.ndarray or cp.ndarray
        Input 2D or 3D array. Can be on CPU or GPU.
        - 1D: shape (,T)
        - 2D: shape (trial, T) → flattened and filtered.
        - 3D: shape (R, trial, T) → flatten last two dims per R.
    factor : float
        Multiplier for robust threshold: threshold = MAD / 0.6745 * factor
    return_rsd : bool
        If True, also return the robust sigma (MAD / 0.6745).
    gpu : bool
        If True or if input is CuPy array, use GPU.

    Returns
    -------
    new_vector : same shape as input
        Denoised array with sub-threshold values set to zero.
    robust_sigma : cp.ndarray or float (optional)
        Returned only if `return_rsd=True`. Shape: (R,) for 3D input.
    """
    # CuPy or NumPy mode
    if gpu or isinstance(vector, cp.ndarray):
        vector = cp.array(vector)
        gaussian_filter1d = cp_gaussian_filter1d
        std_func = cp.nanstd
    else:
        vector = np.array(vector)
        from scipy.ndimage import gaussian_filter1d
        std_func = np.nanstd
    
    if vector.ndim==1: 
        vector_sm = gaussian_filter1d(vector, sigma=1)
        sigma = std_func(vector_sm)
        thresh = sigma*factor
        vector_sm[vector<thresh]=0
        new_vector = vector_sm
        
    elif vector.ndim == 2:
        # vector_flatten = vector
        vector_flatten = vector.flatten()
        vector_flatten_sm = gaussian_filter1d(vector_flatten, sigma=1)
        sigma = std_func(vector_flatten_sm, axis=-1)
        thresh = sigma * factor
        vector_flatten_sm[vector_flatten_sm<thresh]=0
        new_vector = vector_flatten_sm.reshape(vector.shape)
        
    elif vector.ndim == 3:
        R = vector.shape[0]
        vector_flatten = vector.reshape(R, -1)
        vector_flatten_sm = gaussian_filter1d(vector_flatten, sigma=1, axis=-1)
        
        sigma = std_func(vector_flatten_sm, axis=-1)
        thresh = sigma * factor  # shape (R,)
        
        mask = vector_flatten_sm < thresh[:, None]
        vector_flatten_sm[mask] = 0


        # Row-wise thresholding, in-place
        # for i in range(R):
        #        vector_flatten_sm[i][vector_flatten_sm[i]<thresh[i]] = 0
        new_vector = vector_flatten_sm.reshape(vector.shape)
    else:
        raise ValueError("Only 1D, 2D or 3D arrays are supported.")
    
    # if gpu:
    #     new_vector = new_vector.get()
    return (new_vector, sigma) if return_sigma else new_vector

def robust_sd_filter_perROI(vector, factor=2, return_rsd=False, gpu=0):
    """
    Suppress values below a robust threshold (based on MAD) in a CuPy or NumPy array,
    optionally Gaussian-smoothing the signal beforehand.

    Parameters
    ----------
    vector : np.ndarray or cp.ndarray
        Input 2D or 3D array. Can be on CPU or GPU.
        - 1D: shape (,T)
        - 2D: shape (trial, T) → flattened and filtered.
        - 3D: shape (R, trial, time) → flatten last two dims per R.
    factor : float
        Multiplier for robust threshold: threshold = MAD / 0.6745 * factor
    return_rsd : bool
        If True, also return the robust sigma (MAD / 0.6745).
    gpu : bool
        If True or if input is CuPy array, use GPU.

    Returns
    -------
    new_vector : same shape as input
        Denoised array with sub-threshold values set to zero.
    robust_sigma : cp.ndarray or float (optional)
        Returned only if `return_rsd=True`. Shape: (R,) for 3D input.
    """
    # CuPy or NumPy mode
    if gpu or isinstance(vector, cp.ndarray):
        vector = cp.array(vector)
        gaussian_filter1d = cp_gaussian_filter1d
        mad_func = cp_median_abs_deviation
    else:
        vector = np.array(vector)
        from scipy.ndimage import gaussian_filter1d
        from scipy.stats import median_abs_deviation as mad_func
    
    if vector.ndim==1: 
        vector_sm = gaussian_filter1d(vector, sigma=1)
        robust_sigma = mad_func(vector_sm, nan_policy='omit')/0.6745
        thresh = robust_sigma*factor
        vector_sm[vector<thresh]=0
        new_vector = vector_sm
        
    elif vector.ndim == 2:
        # vector_flatten = vector
        vector_flatten = vector.flatten()
        vector_flatten_sm = gaussian_filter1d(vector_flatten, sigma=1)
        robust_sigma = mad_func(vector_flatten_sm) / 0.6745
        thresh = robust_sigma * factor
        vector_flatten_sm[vector_flatten_sm<thresh]=0
        new_vector = vector_flatten_sm.reshape(vector.shape)
        
    elif vector.ndim == 3:
        R = vector.shape[0]
        vector_flatten = vector.reshape(R, -1)
        vector_flatten_sm = gaussian_filter1d(vector_flatten, sigma=1, axis=-1)
        
        robust_sigma = mad_func(vector_flatten_sm, axis=1) / 0.6745  # shape (R,)
        thresh = robust_sigma * factor  # shape (R,)
        
        mask = vector_flatten_sm < thresh[:, None]
        vector_flatten_sm[mask] = 0


        # Row-wise thresholding, in-place
        # for i in range(R):
        #        vector_flatten_sm[i][vector_flatten_sm[i]<thresh[i]] = 0
        new_vector = vector_flatten_sm.reshape(vector.shape)
    else:
        raise ValueError("Only 1D, 2D or 3D arrays are supported.")
    
    # if gpu:
    #     new_vector = new_vector.get()
    return (new_vector, robust_sigma) if return_rsd else new_vector

def robust_filter_along_axis(vector, factor=2.0, axis=-1, gpu=False, sigma=1.0):
    """
    Gaussian-smooth, compute MAD along `axis`, and zero values below (factor * MAD/0.6745)
    on that axis. Uses your existing `cp_median_abs_deviation` for GPU path without changes.

    Parameters
    ----------
    vector : array-like (np.ndarray or cp.ndarray)
    factor : float
        Threshold multiplier for robust sigma (MAD/0.6745).
    axis : int
        Axis along which to compute MAD and threshold.
    gpu : bool
        If True (or if `vector` is a CuPy array), use CuPy path.
    sigma : float
        Gaussian sigma for smoothing along `axis`.

    Returns
    -------
    new_vector : same type as input
    """
    # --- Select backend & funcs ---
    if gpu or isinstance(vector, cp.ndarray):
        xp = cp
        gaussian_filter1d = cp_gaussian_filter1d
        mad_func = cp_median_abs_deviation    
    else:
        xp = np
        from scipy.ndimage import gaussian_filter1d  # CPU version
        from scipy.stats import median_abs_deviation as _np_mad
        # mirror your cp_median_abs_deviation behavior (omit NaNs)
        mad_func = lambda x, axis=None: _np_mad(x, axis=axis, nan_policy='omit')

    # --- To array ---
    vector = xp.array(vector)

    # --- Smooth along the chosen axis ---
    vector_sm = gaussian_filter1d(vector, sigma=sigma, axis=axis)

    # --- Robust sigma (per-slice along axis) ---
    robust_sigma = mad_func(vector_sm, axis=axis) / 0.6745 # equivalent to using 'normal'
    # as scale in median_abs_deviation from scipy
    thresh = robust_sigma * factor

    # --- Broadcast threshold along `axis` ---
    if xp.isscalar(thresh) or getattr(thresh, "ndim", 0) == 0:
        thresh_b = thresh
    else:
        ax = axis if axis >= 0 else vector_sm.ndim + axis
        shape = list(vector_sm.shape)
        shape[ax] = 1
        thresh_b = thresh.reshape(shape)

    # --- Apply threshold on the smoothed data, keep type & shape ---
    new_vector = xp.where(vector_sm < thresh_b, 0, vector_sm)
    return new_vector

def calculate_rsd_along_axis(vector, axis=-1, gpu=False, sigma=1.0):
    """
    Gaussian-smooth, compute MAD along `axis`, and zero values below (factor * MAD/0.6745)
    on that axis. Uses your existing `cp_median_abs_deviation` for GPU path without changes.

    Parameters
    ----------
    vector : array-like (np.ndarray or cp.ndarray)
    factor : float
        Threshold multiplier for robust sigma (MAD/0.6745).
    axis : int
        Axis along which to compute MAD and threshold.
    gpu : bool
        If True (or if `vector` is a CuPy array), use CuPy path.
    sigma : float
        Gaussian sigma for smoothing along `axis`.

    Returns
    -------
    new_vector : same type as input
    """
    # --- Select backend & funcs ---
    if gpu or isinstance(vector, cp.ndarray):
        xp = cp
        gaussian_filter1d = cp_gaussian_filter1d
        mad_func = cp_median_abs_deviation    
    else:
        xp = np
        from scipy.ndimage import gaussian_filter1d  # CPU version
        from scipy.stats import median_abs_deviation as _np_mad
        # mirror your cp_median_abs_deviation behavior (omit NaNs)
        mad_func = lambda x, axis=None: _np_mad(x, axis=axis, nan_policy='omit')

    # --- To array ---
    vector = xp.array(vector)

    # --- Smooth along the chosen axis ---
    vector_sm = gaussian_filter1d(vector, sigma=sigma, axis=axis)

    # --- Robust sigma (per-slice along axis) ---
    robust_sigma = mad_func(vector_sm, axis=axis) / 0.6745 # equivalent to using 'normal'
    
    if gpu:
        return robust_sigma.get()
    else:
        return robust_sigma
# def robust_sd_filter(vector, factor=2, return_rsd=0):
#     vector = np.array(vector)
#     if vector.ndim==2:
#        vector_flatten = vector.flatten()
#        vector_flatten_sm = gaussian_filter1d(vector_flatten, sigma=1)
#        robust_sigma = median_abs_deviation(vector_flatten_sm, nan_policy='omit')/0.6745
#        thresh = robust_sigma*factor
#        vector_flatten_sm[vector_flatten_sm<thresh]=0
#        new_vector = vector_flatten_sm.reshape(vector.shape)
#     # thresh = np.nanmean(vector)+1*np.nanstd(vector)
#     # thresh = median_abs_deviation(vector)/0.6745
#     elif vector.ndim==1: 
#         vector_sm = gaussian_filter1d(vector, sigma=1)
#         robust_sigma = median_abs_deviation(vector_sm, nan_policy='omit')/0.6745
#         thresh = robust_sigma*factor
#         vector_sm[vector<thresh]=0
#         new_vector = vector_sm
#     if return_rsd:
#         return robust_sigma
#     else:
#         return new_vector

def detect_active_trial(session_array, return_perc=False):
    '''
    Parameters
    ----------
    session_array : TYPE
        DESCRIPTION.
    factor : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    '''
    nan_mask = np.isnan(session_array).any(axis=1)
    session_array = session_array[~nan_mask]
    n_trial = session_array.shape[0]
    act_sum = np.sum(session_array>0, axis=1)
    act_trials = act_sum>15
    perc_act_trials = np.sum(act_trials)/n_trial
    if return_perc:
        return act_trials, perc_act_trials
    else:
        return act_trials
# def ativity_filter_rsd(session_array, factor=2):
#     session_array = np.vstack(session_array)
#     nan_mask = np.isnan(session_array).any(axis=1)
#     session_array = session_array[~nan_mask]
#     session_array_thresh = robust_sd_filter(session_array,factor)
#     n_trial = session_array.shape[0]
#     act_sum = np.sum(session_array_thresh>0, axis=1)
#     n_act_trial = np.sum(act_sum>15)
#     return n_act_trial/n_trial

    
# def std_filter(vector, factor=3, return_std=0):
#     vector = np.array(vector)
#     if vector.ndim==2:
#        vector_flatten = vector.flatten()
#        vector_flatten_sm = gaussian_filter1d(vector_flatten, sigma=1)
#        std = np.nanstd(vector_flatten_sm)
#        thresh = std*factor
#        vector_flatten_sm[vector_flatten_sm<thresh]=0
#        new_vector = vector_flatten_sm.reshape(vector.shape)
#     # thresh = np.nanmean(vector)+1*np.nanstd(vector)
#     # thresh = median_abs_deviation(vector)/0.6745
#     elif vector.ndim==1:
#         vector_sm = gaussian_filter1d(vector, sigma=1)
#         std = np.nanstd(vector_sm)
#         thresh = std(vector_sm)*factor
#         vector_sm[vector_sm<thresh]=0
#         new_vector = vector_sm
#     if return_std:
#         return std
#     else:
#         return new_vector
    
# def ativity_filter_std(session_array, factor=2):
#     session_array = np.vstack(session_array)
#     nan_mask = np.isnan(session_array).any(axis=1)
#     session_array = session_array[~nan_mask]
#     session_array_thresh = std_filter(session_array,factor)
#     n_trial = session_array.shape[0]
#     act_sum = np.sum(session_array_thresh>0, axis=1)
#     n_act_trial = np.sum(act_sum>15)
#     return n_act_trial/n_trial

#%% test
if __name__ == "__main__":
    test_data = np.load(r"Z:\Jingyu\2P_Recording\AC302\AC302-20250727\02\suite2p\plane0\F.npy")
    test_roi = test_data[279:281]
    mad = median_abs_deviation(test_roi, nan_policy='omit', axis=-1)
    mad_cp = cp_median_abs_deviation(test_roi, nan_policy='omit', axis=-1).get()
    print(mad, mad_cp)
    
    filtered = robust_sd_filter_perROI(test_roi, factor=1.5)
    # filtered_new = robust_sd_filter_new(test_roi, factor=1.5)
    
    F1 = test_roi[0, 0:3000]
    F2 = filtered[0, 0:3000]
    # F3 = filtered_new[0, 0:3000]
    fig, ax = plt.subplots()
    ax.plot(F1, color='tab:blue')
    ax.plot(F2, color='tab:green')
    
    # fig, ax = plt.subplots()
    # ax.plot(F1, color='tab:blue')
    # ax.plot(F3, color='tab:orange')
    #%% test
    # p_beh = r"Z:\Jingyu\2P_Recording\all_session_info\{}\all_anm_behaviour\{}.pkl".format(exp, anm_id+'-'+date+'-'+ss[i])
    import utils_Jingyu as utl
    p_beh = r"Z:\Jingyu\2P_Recording\all_session_info\GCaMP8s_infusion\all_anm_behaviour\AC302-20250727-02.pkl"
    import pandas as pd
    beh = pd.read_pickle(p_beh)
    test_data_sub = test_data[:10, :]
    test_data_dff = utl.get_dff(test_data_sub)
    rsd_filtered = robust_filter_along_axis(test_data_dff, gpu=1).get()
    
    # test_roi = test_data[279:281]
    # test_roi = utl.get_dff(test_roi)
    
    # run_aligned = utl.align_trials(test_roi, 'run', beh, 2, 4)
    # run_aligned_filtered = robust_sd_filter_perROI(run_aligned[0], factor=2, gpu=1).get()
    
    F1 = test_data_dff[2,:5000]
    F2 = rsd_filtered[2, :5000]
    # F3 = filtered_new[0, 0:3000]
    fig, ax = plt.subplots()
    ax.plot(F1, color='tab:blue')
    ax.plot(F2, color='tab:green')
    #%% test
    # p_beh = r"Z:\Jingyu\2P_Recording\all_session_info\{}\all_anm_behaviour\{}.pkl".format(exp, anm_id+'-'+date+'-'+ss[i])
    import utils_Jingyu as utl
    p_beh = r"Z:\Jingyu\2P_Recording\all_session_info\GCaMP8s_infusion\all_anm_behaviour\AC302-20250727-02.pkl"
    import pandas as pd
    beh = pd.read_pickle(p_beh)
    test_roi = test_data[279:281]
    test_roi = utl.get_dff(test_roi)
    
    run_aligned = utl.align_trials(test_roi, 'run', beh, 2, 4)
    run_aligned_filtered = robust_sd_filter_perROI(run_aligned[0], factor=2, gpu=1).get()
    
    F1 = run_aligned[0, 6, :]
    F2 = run_aligned_filtered[6, :]
    # F3 = filtered_new[0, 0:3000]
    fig, ax = plt.subplots()
    ax.plot(F1, color='tab:blue')
    ax.plot(F2, color='tab:green')