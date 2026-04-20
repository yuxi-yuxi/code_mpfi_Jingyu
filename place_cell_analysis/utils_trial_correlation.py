# -*- coding: utf-8 -*-
"""
Trial-to-trial correlation utilities for place cell and time cell analysis.

GPU-accelerated functions for calculating various trial correlation metrics.
Shared between place_cell_functions.py and time_cell_functions.py.

@author: Jingyu Cao
"""

import numpy as np


def normalize_per_lap_profile(per_lap):
    """
    Convert per_lap_profile to 2D numpy array.

    Handles various input formats from parquet loading (1D arrays of lists, etc.)

    Parameters:
    -----------
    per_lap : array-like or None
        Per-lap profile data in various formats

    Returns:
    --------
    ndarray or None
        2D numpy array (n_laps, n_bins) or None if invalid
    """
    if per_lap is None:
        return None

    # Handle 1D numpy array of lists (from parquet loading)
    if isinstance(per_lap, np.ndarray):
        if per_lap.ndim == 1 and len(per_lap) > 0:
            if hasattr(per_lap[0], '__len__'):
                per_lap = np.vstack(per_lap)
        elif per_lap.ndim == 0:
            per_lap = per_lap.item()
            if isinstance(per_lap, list) and len(per_lap) > 0:
                per_lap = np.array(per_lap)

    if isinstance(per_lap, list) and len(per_lap) > 0:
        per_lap = np.array(per_lap)

    if not isinstance(per_lap, np.ndarray) or per_lap.ndim != 2:
        return None

    return per_lap


def pearson_corr_rows_gpu(X, xp):
    """
    Compute pairwise Pearson correlations between all rows of X using GPU.

    Parameters:
    -----------
    X : array (n_rows, n_features)
        Data matrix on GPU (cupy) or CPU (numpy)
    xp : module
        Either cupy or numpy

    Returns:
    --------
    corr_matrix : array (n_rows, n_rows)
        Pairwise correlation matrix
    """
    # Center each row
    X_centered = X - xp.nanmean(X, axis=1, keepdims=True)

    # Replace NaN with 0 for correlation computation
    X_centered = xp.nan_to_num(X_centered, nan=0.0)

    # Compute norms
    norms = xp.sqrt(xp.sum(X_centered ** 2, axis=1, keepdims=True))
    norms = xp.where(norms == 0, 1, norms)  # Avoid division by zero

    # Normalize
    X_norm = X_centered / norms

    # Correlation matrix = X_norm @ X_norm.T
    corr_matrix = X_norm @ X_norm.T

    return corr_matrix


def calculate_trial_correlations_gpu(per_lap_profiles, methods=None, gpu=True):
    """
    Calculate trial-to-trial correlations for multiple cells using GPU acceleration.

    Parameters:
    -----------
    per_lap_profiles : list of ndarray
        List of per-lap profiles, each with shape (n_laps, n_bins)
    methods : str, list of str, or None
        Correlation type(s) to calculate. Options:
        - 'odd_even' : Correlation between mean of odd and even trials
        - 'mean_pairwise' : Average correlation between all pairs of trials
        - 'consecutive' : Average correlation between consecutive trials (N vs N+1)
        Can be a single string, a list of strings, or None (calculates all three).
    gpu : bool
        Whether to use GPU acceleration

    Returns:
    --------
    If single method (str): ndarray (n_cells,)
        Correlation values for each cell
    If multiple methods (list or None): dict
        Dictionary with keys corresponding to requested methods, each containing
        ndarray (n_cells,) of correlation values
    """
    # Handle methods parameter
    single_method = False
    if methods is None:
        methods = ['odd_even', 'mean_pairwise', 'consecutive']
    elif isinstance(methods, str):
        single_method = True
        methods = [methods]

    # Validate methods
    valid_methods = {'odd_even', 'mean_pairwise', 'consecutive'}
    for m in methods:
        if m not in valid_methods:
            raise ValueError(f"Unknown method: {m}. Valid options: {valid_methods}")

    n_cells = len(per_lap_profiles)

    # Initialize result arrays for requested methods
    results = {m: np.full(n_cells, np.nan) for m in methods}

    # Try to use GPU
    if gpu:
        try:
            import cupy as cp
            xp = cp
            use_gpu = True
        except ImportError:
            xp = np
            use_gpu = False
    else:
        xp = np
        use_gpu = False

    # Process each cell
    for i, per_lap in enumerate(per_lap_profiles):
        per_lap = normalize_per_lap_profile(per_lap)
        if per_lap is None:
            continue

        n_laps, n_bins = per_lap.shape
        if n_laps < 2 or n_bins < 3:
            continue

        # Remove laps with all NaN values
        valid_laps = ~np.all(np.isnan(per_lap), axis=1)
        per_lap = per_lap[valid_laps]
        n_laps = per_lap.shape[0]

        if n_laps < 2:
            continue

        if use_gpu:
            per_lap_gpu = cp.asarray(per_lap)
        else:
            per_lap_gpu = per_lap

        # Mean pairwise correlation
        if 'mean_pairwise' in methods:
            corr_matrix = pearson_corr_rows_gpu(per_lap_gpu, xp)
            n = corr_matrix.shape[0]
            if use_gpu:
                upper_tri = cp.triu(cp.ones((n, n), dtype=bool), k=1)
                pairwise_corrs = corr_matrix[upper_tri]
                results['mean_pairwise'][i] = float(cp.nanmean(pairwise_corrs).get())
            else:
                upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
                pairwise_corrs = corr_matrix[upper_tri]
                results['mean_pairwise'][i] = float(np.nanmean(pairwise_corrs))

        # Consecutive correlation
        if 'consecutive' in methods:
            X1 = per_lap_gpu[:-1]
            X2 = per_lap_gpu[1:]
            X1_centered = X1 - xp.nanmean(X1, axis=1, keepdims=True)
            X2_centered = X2 - xp.nanmean(X2, axis=1, keepdims=True)
            X1_centered = xp.nan_to_num(X1_centered, nan=0.0)
            X2_centered = xp.nan_to_num(X2_centered, nan=0.0)
            numer = xp.sum(X1_centered * X2_centered, axis=1)
            denom = xp.sqrt(xp.sum(X1_centered**2, axis=1) * xp.sum(X2_centered**2, axis=1))
            denom = xp.where(denom == 0, 1, denom)
            consecutive_corrs = numer / denom
            if use_gpu:
                results['consecutive'][i] = float(cp.nanmean(consecutive_corrs).get())
            else:
                results['consecutive'][i] = float(np.nanmean(consecutive_corrs))

        # Odd-even correlation
        if 'odd_even' in methods:
            odd_mean = xp.nanmean(per_lap_gpu[::2], axis=0)
            even_mean = xp.nanmean(per_lap_gpu[1::2], axis=0)
            odd_centered = odd_mean - xp.nanmean(odd_mean)
            even_centered = even_mean - xp.nanmean(even_mean)
            odd_centered = xp.nan_to_num(odd_centered, nan=0.0)
            even_centered = xp.nan_to_num(even_centered, nan=0.0)
            numer = xp.sum(odd_centered * even_centered)
            denom = xp.sqrt(xp.sum(odd_centered**2) * xp.sum(even_centered**2))
            if denom > 0:
                corr = numer / denom
                if use_gpu:
                    results['odd_even'][i] = float(corr.get())
                else:
                    results['odd_even'][i] = float(corr)

    # Return array if single method, dict if multiple
    if single_method:
        return results[methods[0]]
    return results


# Alias for backward compatibility
calculate_all_trial_correlations_gpu = calculate_trial_correlations_gpu
