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
from common.utils_basic import nearest_mapping, normalize, vectorized_roll
from common.robust_sd_filter import robust_filter_along_axis

# Import shared trial correlation utilities
from .utils_trial_correlation import (
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
    
        for i_lap in range(n_laps):
            lap_start = lap_start_indices[i_lap]
            lap_end = lap_start_indices[i_lap + 1]
    
            lap_bin_indices = bin_indices[lap_start:lap_end]
            lap_calcium = running_calcium_map_img[:, lap_start:lap_end]  # (n_cells, n_lap_frames)
    
            # Occupancy for this lap
            lap_occupancy = np.bincount(lap_bin_indices, minlength=n_bins).astype(float)
            lap_occupancy_safe = lap_occupancy.copy()
            lap_occupancy_safe[lap_occupancy_safe == 0] = np.nan
    
            # Vectorized: sum dF/F per bin for all cells at once
            # Create a sparse-like accumulation using advanced indexing
            dff_sum_lap = np.zeros((n_cells, n_bins))
            np.add.at(dff_sum_lap, (slice(None), lap_bin_indices), lap_calcium)
            per_lap_profile[:, i_lap, :] = dff_sum_lap / lap_occupancy_safe
    else:
        per_lap_profile = []

    return event_count_raw, event_rate_raw, occupancy_raw, per_lap_profile

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
    occupancy : ndarray (n_bins,)
        Frame count per spatial bin
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

    total_frames = xp.sum(occupancy)
    p_i = occupancy / total_frames  # probability per bin (n_bins,)

    # Mean event rate for each cell: r_bar = sum(r_i * p_i)
    r_bar = xp.nansum(event_rate * p_i, axis=1, keepdims=True)  # (n_cells, 1)

    # Avoid division by zero
    r_bar_safe = xp.where(r_bar == 0, xp.nan, r_bar)

    # r_ratio = r_i / r_bar for each cell and bin
    r_ratio = event_rate / r_bar_safe  # (n_cells, n_bins)

    # Mask for valid bins: occupancy > 0 and r_i > 0
    valid_mask = (occupancy > 0) & (event_rate > 0)  # (n_cells, n_bins)

    # Compute: p_i * (r_i / r_bar) * log2(r_i / r_bar)
    log_term = xp.where(r_ratio > 0, xp.log2(r_ratio), 0)
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
                                   min_shift=450, # 15 s
                                   gpu=True):
    """
    Compute shuffled spatial information by circular shifting calcium traces.

    Parameters:
    -----------
    running_calcium_map_img : ndarray (n_cells, n_frames)
        Calcium traces during running for each cell
    running_distance_map_img : ndarray (n_frames,)
        Distance at each running frame
    occupancy : ndarray (n_bins,)
        Frame count per spatial bin
    shuff_times : int
        Number of shuffle iterations
    gpu : bool
        Whether to use GPU acceleration with CuPy

    Returns:
    --------
    shuffled_SI : ndarray (n_cells, shuff_times)
        Shuffled spatial information for each cell and iteration
    """
    n_rois, n_frames = running_calcium_map_img.shape
    shuffled_SI = np.zeros((n_rois, shuff_times))

    # Random shift amounts for each shuffle (different for each ROI)
    # Ensure each shift is at least min_shift frames
    rng = np.random.default_rng()
    shift_amounts = rng.integers(min_shift, n_frames - min_shift, size=(shuff_times, n_rois))

    if gpu:
        import cupy as cp
        calcium_gpu = cp.asarray(running_calcium_map_img)
        distance_gpu = cp.asnumpy(running_distance_map_img) if hasattr(running_distance_map_img, 'get') else running_distance_map_img
    else:
        calcium_gpu = running_calcium_map_img
        distance_gpu = running_distance_map_img

    for i in tqdm(range(shuff_times), desc='calculating shuffled SI...'):
        # Vectorized roll: shift each ROI by different amount
        if gpu:
            shifts_gpu = cp.asarray(shift_amounts[i])
            calcium_shuff = vectorized_roll(calcium_gpu, shifts_gpu, xp=cp)
            calcium_shuff_np = cp.asnumpy(calcium_shuff)
        else:
            calcium_shuff_np = vectorized_roll(calcium_gpu, shift_amounts[i], xp=np)

        # Re-bin the shuffled calcium traces
        _, event_rate_shuff, _, _ = spatial_binning(calcium_shuff_np, distance_gpu, save_lap_profile=False)
        shuffled_SI[:, i] = calculate_spatial_info(event_rate_shuff, occupancy, gpu=gpu)

    return shuffled_SI

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

    # Remove laps with all NaN values
    valid_laps = ~np.all(np.isnan(per_lap_profile), axis=1)
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


def detect_place_field(event_count_raw, event_rate_raw, occupancy_raw,
                       per_lap_profile=None,
                       sigma=1.5,  # bins
                       kernel_size=5,  # bins
                       bin_size=4,  # cm
                       ):
    """
    Detect place fields and compute spatial information for each ROI.

    Parameters:
    -----------
    event_count_raw : ndarray (n_cells, n_bins)
        Sum of dF/F values per spatial bin for each cell
    event_rate_raw : ndarray (n_cells, n_bins)
        Mean dF/F per spatial bin for each cell (event_count / occupancy)
    occupancy_raw : ndarray (n_bins,)
        Frame count per spatial bin
    per_lap_profile : ndarray (n_cells, n_laps, n_bins), optional
        Event rate per spatial bin for each cell and lap
    sigma : float
        Gaussian kernel sigma in bins
    kernel_size : int
        Gaussian kernel size in bins
    bin_size : float
        Spatial bin size in cm

    Returns:
    --------
    df_place_field : DataFrame
        DataFrame containing place field info for each ROI
    """
    n_cells, n_bins = event_count_raw.shape
    
    # Create truncated Gaussian kernel
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    gaussian_kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # normalize

    # Smooth occupancy with Gaussian kernel
    occupancy_smoothed = convolve1d(occupancy_raw, gaussian_kernel, mode='constant')

    # Smooth event counts with Gaussian kernel
    event_count_smoothed = np.apply_along_axis(
        lambda row: convolve1d(row, gaussian_kernel, mode='constant'),
        axis=1, arr=event_count_raw
    )

    # Compute place field map (smoothed event rate)
    occupancy_smoothed_safe = occupancy_smoothed.copy()
    occupancy_smoothed_safe[occupancy_smoothed_safe == 0] = np.nan
    place_field_map = event_count_smoothed / occupancy_smoothed_safe

    # Normalize place field by maximum value
    place_field_max = np.nanmax(place_field_map, axis=1, keepdims=True)
    place_field_max[place_field_max == 0] = 1  # avoid division by zero
    place_field_normalized = place_field_map / place_field_max

    # Find place field position (bin with peak value)
    place_field_peak_bin = np.nanargmax(place_field_map, axis=1)
    place_field_position_cm = (place_field_peak_bin + 0.5) * bin_size  # center of bin

    # Peak amplitude (max of smoothed place field)
    place_field_peak_amplitude = np.nanmax(place_field_map, axis=1)

    # Compute total events per cell
    total_events = np.nansum(event_count_raw, axis=1)

    # Create DataFrame with place_field_map as list of 1D arrays
    df_place_field = pd.DataFrame({
        'cell_id': np.arange(n_cells),
        'per_lap_profile': [per_lap_profile[i] if per_lap_profile is not None else None for i in range(n_cells)],
        'place_field_map': [place_field_map[i] for i in range(n_cells)],
        'place_field_map_norm': [place_field_normalized[i] for i in range(n_cells)],
        'place_field_peak_bin': place_field_peak_bin,
        'place_field_position_cm': place_field_position_cm,
        'place_field_peak_amplitude': place_field_peak_amplitude,
        'total_events': total_events,
    })

    return df_place_field


    
def align_run_frame_calcium(dff, beh):
    
    frame_times = beh['frame_times']
    n_frames = dff.shape[-1]
    # covert time dimension to distance
    if (len(frame_times)<n_frames) or ((len(frame_times)-n_frames)>100):
        running_time_map_img, running_distance_map_img, running_calcium_map_img = [], [], []
        print('frame times and actual frames not matching!!!')
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
            running_time_map_img, running_distance_map_img, running_calcium_map_img = [], [], []
            print('no enough running farme!!!')
        else:
            running_time_map_img = running_time_binned - running_time_binned[0] # ms
            running_time_map_img = running_time_map_img[runing_idx_img]
            running_distance_map_img = running_distance_binned[runing_idx_img]
            running_calcium_map_img = dff[:, runing_idx_img]  # (n_cells, n_running_frames)   
            
    return running_time_map_img, running_distance_map_img, running_calcium_map_img
    
#%%
# test script
if __name__ == '__main__':
    
    from drug_infusion import rec_lst_infusion as recs
    rec_ctrl = recs.rec_SCH_ctrl
    rec_SCH  = recs.rec_SCH
    
    OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
    OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
    
    # example rec
    # rec = rec_ctrl.iloc[3]
    # rec = rec_SCH.iloc[4]
    df_place_field_all = pd.DataFrame()
    for _, rec in rec_SCH.head(3).iterrows():
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

        dff = robust_filter_along_axis(dff_ss1, factor=2.5)
        running_time_map_img, running_distance_map_img, running_calcium_map_img = align_run_frame_calcium(dff, beh)   
        
        # Run analysis using the functions
        # Parameters
        track_length = 180  # cm
        bin_size = 4  # cm
        n_bins = int(track_length / bin_size)  # 45 bins
    
        # Spatial binning
        event_count_raw, event_rate_raw, occupancy_raw, per_lap_profile = spatial_binning(
            running_calcium_map_img, running_distance_map_img,
            track_length=track_length, bin_size=bin_size
        )
    
        # Detect place fields
        df_place_field = detect_place_field(
            event_count_raw, event_rate_raw, occupancy_raw,
            per_lap_profile=per_lap_profile,
            sigma=1.5, kernel_size=5, bin_size=bin_size
        )
        
        total_active_frames = np.sum(running_calcium_map_img>0, axis=-1)
        df_place_field['total_active_frames'] = total_active_frames
        # Compute spatial information
        spatial_information = calculate_spatial_info(event_rate_raw, occupancy_raw)
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
            shuffled_SI = calculate_spatial_info_shuffle(
                running_calcium_map_img[candidate_indices],
                running_distance_map_img,
                occupancy_raw, shuff_times=1000, gpu=True
            )
            # Assign shuffled SI to candidate ROIs
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
