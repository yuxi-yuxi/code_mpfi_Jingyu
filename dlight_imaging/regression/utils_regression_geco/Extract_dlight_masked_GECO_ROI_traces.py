import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import scipy.stats
from pathlib import Path
from scipy.ndimage import binary_dilation

def _extract_roi_trace_worker_intersected(roi_idx, dff, stat_array, global_dlight_mask):
    """
    Worker function. Extracts the mean fluorescence trace for a single ROI, but
    only using the pixels that are also present in the `global_dlight_mask`.

    Args:
        roi_idx (int): The index of the ROI to process.
        dff (np.ndarray): The full dF/F movie.
        stat_array (np.ndarray): The loaded 'stat.npy' data.
        global_dlight_mask (np.ndarray): The pre-computed global mask of valid pixels.

    Returns:
        np.ndarray: The 1D fluorescence trace for the refined ROI.
    """
    T, H, W = dff.shape
    roi_info = stat_array[roi_idx]
    
    suite2p_roi_mask = np.zeros((H, W), dtype=bool)
    ypix, xpix = roi_info['ypix'], roi_info['xpix']
    if len(ypix) == 0: return np.full(T, np.nan)
    suite2p_roi_mask[ypix, xpix] = True

    # The final mask is only the pixels that exist in BOTH the Suite2p ROI
    # AND global dLight mask.
    final_roi_mask = suite2p_roi_mask & global_dlight_mask
    
    # Check if any pixels remain after the intersection
    if np.sum(final_roi_mask) < 10: # Use a threshold to ensure a stable trace
        return np.full(dff.shape[0], np.nan)

    # Instead of using the 2D boolean mask directly, find the
    # (y, x) coordinates of the True pixels.
    ypix_final, xpix_final = np.where(final_roi_mask)

    # use these coordinate arrays for indexing. This is unambiguous
    # and highly memory-efficient. It creates a (T, num_pixels) array directly.
    roi_pixels_over_time = dff[:, ypix_final, xpix_final]
    
    # Calculate the mean across the pixel axis.
    roi_trace = roi_pixels_over_time.mean(axis=1)
    
    return roi_trace


def extract_roi_traces_parallel_intersected(
    dff, 
    stat_array, 
    global_dlight_mask,
    roi_indices_to_extract: np.ndarray = None, # <-- NEW ARGUMENT
    n_jobs=-1
):
    """
    Efficiently extracts traces for a specified subset of ROIs, refining each
    ROI by intersecting it with a provided global mask.

    Args:
        dff (np.ndarray): The dF/F movie (T, H, W).
        stat_array (np.ndarray): The loaded 'stat.npy' data.
        global_dlight_mask (np.ndarray): A 2D boolean mask of valid signal pixels.
        roi_indices_to_extract (np.ndarray, optional): A 1D array of the integer indices
            of the ROIs to extract. If None, traces for ALL ROIs will be extracted.
            Defaults to None.
        n_jobs (int): The number of jobs to run in parallel.
    """
    # --- Step 1: Load necessary files ---
    num_total_rois = len(stat_array)

    # --- THIS IS THE KEY CHANGE ---
    # Determine which ROI indices to process
    if roi_indices_to_extract is None:
        # If no subset is specified, process all ROIs
        tasks = np.arange(num_total_rois)
        print(f"No specific ROIs provided. Extracting traces for all {num_total_rois} ROIs...")
    else:
        # If a subset is specified, use that as our task list
        tasks = roi_indices_to_extract
        print(f"Extracting traces for the {len(tasks)} selected ROIs...")

    # --- Step 2: Run the parallel extraction on the specified tasks ---
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_extract_roi_trace_worker_intersected)(roi_idx, dff, stat_array, global_dlight_mask) 
        for roi_idx in tqdm(tasks, desc="Extracting selected ROI traces")
    )
    
    # --- Step 3: Unpack results and select corresponding neuropil ---
    geco_traces_selected = np.array(results)
                
    return geco_traces_selected


# def _extract_roi_signal_and_background_worker(roi_idx, dff, stat_array, global_dlight_mask):
#     """
#     Worker function. For a single ROI, it extracts two traces:
#     1. The 'signal' trace from pixels INSIDE the global_dlight_mask.
#     2. The 'background' trace from pixels OUTSIDE the global_dlight_mask.
#     """
#     T, H, W = dff.shape
#     roi_info = stat_array[roi_idx]
    
#     # Create a mask for the current Suite2p ROI's pixels
#     suite2p_roi_mask = np.zeros((H, W), dtype=bool)
#     ypix, xpix = roi_info['ypix'], roi_info['xpix']
#     if len(ypix) == 0:
#         return (np.full(T, np.nan), np.full(T, np.nan))
#     suite2p_roi_mask[ypix, xpix] = True

#     # --- Define the two final masks using logical operations ---
    
#     # 1. Signal Mask: Pixels in the Suite2p ROI AND in the global mask
#     signal_mask = suite2p_roi_mask & global_dlight_mask
    
#     # 2. Background Mask: Pixels in the Suite2p ROI but NOT in the global mask
#     background_mask = suite2p_roi_mask & (~global_dlight_mask)
    
#     # --- Extract traces for both masks ---
#     signal_trace = np.nan
#     background_trace = np.nan

#     if np.sum(signal_mask) >= 10:
#         ypix_sig, xpix_sig = np.where(signal_mask)
#         signal_trace = dff[:, ypix_sig, xpix_sig].mean(axis=1)
    
#     if np.sum(background_mask) >= 10:
#         ypix_bg, xpix_bg = np.where(background_mask)
#         background_trace = dff[:, ypix_bg, xpix_bg].mean(axis=1)
    
#     # Return both traces as a tuple
#     return (signal_trace, background_trace)


def _extract_roi_signal_and_background_worker(
    roi_idx: int, 
    mov: np.ndarray, 
    stat_array: np.ndarray, 
    global_dlight_mask: np.ndarray,
    neuropil_dilation_iterations: int = 3
):
    """
    Worker function. For a single ROI, it extracts two traces:
    1. The 'signal' trace from the intersection of the Suite2p ROI and the global mask.
    2. The 'neuropil' trace from a shell around the Suite2p ROI, which is then
       itself masked by the global_dlight_mask.

    Args:
        (Familiar arguments...)
        neuropil_dilation_iterations (int): How many pixels to expand the ROI mask
                                            to create the neuropil shell.
    """
    T, H, W = mov.shape
    roi_info = stat_array[roi_idx]
    
    # Create a mask for the current Suite2p ROI's pixels
    suite2p_roi_mask = np.zeros((H, W), dtype=bool)
    ypix, xpix = roi_info['ypix'], roi_info['xpix']
    if len(ypix) == 0:
        return (np.full(T, np.nan), np.full(T, np.nan))
    suite2p_roi_mask[ypix, xpix] = True

    # --- Define the two final masks using the new logic ---
    
    # 1. Signal Mask: Pixels in the Suite2p ROI AND in the global mask (same as before)
    signal_mask = suite2p_roi_mask & global_dlight_mask
    
    # --- THIS IS THE NEW, SUPERIOR NEUROPIL DEFINITION ---
    # 2. Neuropil Mask:
    #    a) Create a dilated "shell" around the Suite2p ROI
    dilated_roi = binary_dilation(suite2p_roi_mask, iterations=neuropil_dilation_iterations)
    neuropil_shell = dilated_roi & (~suite2p_roi_mask)
    
    #    b) Intersect this shell with the global dLight mask to get high-quality neuropil pixels
    final_neuropil_mask = neuropil_shell & global_dlight_mask
    
    # --- Extract traces for both masks ---
    signal_trace = np.full(T, np.nan) # Initialize as NaN
    background_trace = np.full(T, np.nan) # Initialize as NaN

    if np.sum(signal_mask) >= 5: # Threshold for stable signal
        ypix_sig, xpix_sig = np.where(signal_mask)
        signal_trace = mov[:, ypix_sig, xpix_sig].mean(axis=1)
    
    if np.sum(final_neuropil_mask) >= 10: # Use a slightly higher threshold for neuropil
        ypix_bg, xpix_bg = np.where(final_neuropil_mask)
        background_trace = mov[:, ypix_bg, xpix_bg].mean(axis=1)
    
    # Return both traces as a tuple
    return (signal_trace, background_trace)


def extract_signal_and_background_traces_parallel(
    mov, 
    stat_array, 
    global_dlight_mask,
    roi_indices_to_extract: np.ndarray = None,
    neuropil_dilation_iterations: int = 3,
    n_jobs=-1
):
    """
    Efficiently extracts both a signal trace and a local background trace for
    a specified subset of ROIs.
    """
    num_total_rois = len(stat_array)

    if roi_indices_to_extract is None:
        tasks = np.arange(num_total_rois)
        print(f"Extracting traces for all {num_total_rois} ROIs...")
    else:
        tasks = roi_indices_to_extract
        print(f"Extracting traces for the {len(tasks)} selected ROIs...")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_extract_roi_signal_and_background_worker)(roi_idx, mov, stat_array, global_dlight_mask, neuropil_dilation_iterations) 
        for roi_idx in tqdm(tasks, desc="Extracting signal/background traces")
    )
    
    # --- Unpack the tuples of results into two separate arrays ---
    # `results` is a list of tuples, e.g., [(trace1_sig, trace1_bg), (trace2_sig, trace2_bg), ...]
    # We can unpack it efficiently:
    signal_traces_selected = np.array([res[0] for res in results])
    background_traces_selected = np.array([res[1] for res in results])
                
    return signal_traces_selected, background_traces_selected


def calculate_trial_average_from_traces(traces_array, run_onset_frames, pre_frames=90, post_frames=120):
    """
    Calculates the trial-averaged mean and SEM from a pre-computed array of traces.
    """
    if np.all(np.isnan(traces_array)):
        print("Warning: Input traces array is all NaN. Returning empty results.")
        return None, None
    
    n_roi, T = traces_array.shape
    window_len = pre_frames + post_frames
    
    all_trials = []
    for onset in run_onset_frames:
        start_frame = onset - pre_frames
        end_frame = onset + post_frames
        
        # Boundary check
        if start_frame >= 0 and end_frame < T:
            trial_data = traces_array[:, start_frame:end_frame]
            all_trials.append(trial_data)
            
    if not all_trials:
        print("Warning: No valid trials found within the movie bounds.")
        return None, None
    
    # Stack trials and compute mean and SEM
    stacked_trials = np.stack(all_trials, axis=0) # Shape: (n_trials, n_roi, window_len)
    mean_trials = np.nanmean(stacked_trials, axis=0)
    sem_trials = scipy.stats.sem(stacked_trials, axis=0, nan_policy='omit')
    
    return mean_trials, sem_trials


def plot_roi_response_map(
    mean_trace_trials: np.ndarray, 
    stat_array: np.ndarray,
    roi_indices_to_extract: np.ndarray,
    pre_frames: int,
    mean_img: np.ndarray,
    imaging_rate: float = 30.0, 
    str_align: str = 'Run Onset',
    save_path: str = None
):
    """
    Visualizes trial-averaged responses by coloring each individual ROI on a map,
    using a pre-computed array of trial-averaged traces.

    Args:
        mean_trace_trials (np.ndarray): 2D array (num_ROIs_extracted, T_trial) of mean traces.
        stat_array (np.ndarray): The full stat array from Suite2p.
        roi_indices_to_extract (np.ndarray): The integer indices of the ROIs that were extracted
                                           and are present in `mean_trace_trials`.
        pre_frames (int): Number of pre-onset frames.
        mean_img (np.ndarray): The anatomical background image.
        imaging_rate (float): Imaging rate in Hz.
        str_align (str): String for the plot title.
    """
    num_rois_extracted, T_trial = mean_trace_trials.shape
    H, W = mean_img.shape
    
    
    if len(roi_indices_to_extract) != num_rois_extracted:
        print("Error: Length of `roi_indices_to_extract` must match the number of rows in `mean_trace_trials`.")
        return

    # --- Step 1: Calculate the response amplitude for each ROI ---
    roi_amplitudes = np.full(num_rois_extracted, np.nan)

    # Define windows based on the trial snippet length
    response_window = slice(pre_frames + int(0 * imaging_rate), pre_frames + int(1.5 * imaging_rate))
    baseline_window = slice(pre_frames - int(1 * imaging_rate), pre_frames - int(0 * imaging_rate))

    for i in range(num_rois_extracted):
        # Correctly slice the 2D mean_trace_trials array for the i-th ROI
        baseline = np.mean(mean_trace_trials[i, baseline_window])
        response = np.mean(mean_trace_trials[i, response_window])
        
        # Add epsilon to prevent division by zero
        if baseline != 0:
            roi_amplitudes[i] = (response - baseline) / (abs(baseline) + 1e-9)

    # --- Step 2: Create "Color-by-Number" response image ---
    response_img = np.full((H, W), np.nan, dtype=np.float32)

    # Loop through the extracted ROIs and paint them
    for i, roi_idx in enumerate(roi_indices_to_extract):
        roi_info = stat_array[roi_idx]
        # "Paint" the pixels of each ROI with its calculated amplitude
        response_img[roi_info['ypix'], roi_info['xpix']] = roi_amplitudes[i]

    # --- Step 3: Visualize the map ---
    # Find a symmetric color scale centered at zero
    if not np.all(np.isnan(roi_amplitudes)):
        vmax = np.nanpercentile(np.abs(roi_amplitudes), 98) # Use 98th percentile for robustness
        vmin = -vmax
    else:
        vmin, vmax = -1, 1 # Default if no data

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    fig.suptitle(f'ROI-Based Response Map Aligned to {str_align}', fontsize=16)

    ax.imshow(mean_img, cmap='gray')
    im = ax.imshow(response_img, cmap='coolwarm', alpha=0.7, vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('dF/F change (Post - Pre) / |Pre|')

    if save_path:
        print(f"   > Saving figure to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()
    plt.close(fig)


## -----------------------------------------------------------------------------
## Batched extraction (for long recordings where the full float32 stack OOMs)
##
## These functions accept an int16 memmap, precompute per-ROI pixel indices once,
## then iterate over frame batches, casting each chunk to float32 just in time.
## Behaviour mirrors the non-batched workers (pixel thresholds, NaN handling).
## -----------------------------------------------------------------------------

def _build_intersected_roi_masks(stat_array, global_dlight_mask, soma_indices):
    """Precompute (ypix, xpix) index arrays for `suite2p_roi & global_dlight_mask`.

    Returns a list aligned with `soma_indices`; entries are None when the final
    mask has fewer than 10 pixels (matching `_extract_roi_trace_worker_intersected`).
    """
    H, W = global_dlight_mask.shape
    masks = []
    for roi_idx in soma_indices:
        roi_info = stat_array[roi_idx]
        ypix, xpix = roi_info['ypix'], roi_info['xpix']
        if len(ypix) == 0:
            masks.append(None)
            continue
        suite2p_roi_mask = np.zeros((H, W), dtype=bool)
        suite2p_roi_mask[ypix, xpix] = True
        final_roi_mask = suite2p_roi_mask & global_dlight_mask
        if np.sum(final_roi_mask) < 10:
            masks.append(None)
            continue
        masks.append(np.where(final_roi_mask))
    return masks


def _build_signal_and_neuropil_masks(stat_array, global_dlight_mask, soma_indices,
                                     neuropil_dilation_iterations=3):
    """Precompute per-ROI (signal_yx, neuropil_yx) index arrays.

    Mirrors `_extract_roi_signal_and_background_worker`: signal threshold >= 5,
    neuropil threshold >= 10; either slot can be None independently.
    """
    H, W = global_dlight_mask.shape
    masks = []
    for roi_idx in soma_indices:
        roi_info = stat_array[roi_idx]
        ypix, xpix = roi_info['ypix'], roi_info['xpix']
        if len(ypix) == 0:
            masks.append((None, None))
            continue
        suite2p_roi_mask = np.zeros((H, W), dtype=bool)
        suite2p_roi_mask[ypix, xpix] = True

        signal_mask = suite2p_roi_mask & global_dlight_mask
        dilated_roi = binary_dilation(suite2p_roi_mask, iterations=neuropil_dilation_iterations)
        neuropil_shell = dilated_roi & (~suite2p_roi_mask)
        final_neuropil_mask = neuropil_shell & global_dlight_mask

        sig = np.where(signal_mask) if np.sum(signal_mask) >= 5 else None
        bg = np.where(final_neuropil_mask) if np.sum(final_neuropil_mask) >= 10 else None
        masks.append((sig, bg))
    return masks


def extract_roi_traces_batched_intersected(
    mov_memmap,
    stat_array,
    global_dlight_mask,
    roi_indices_to_extract: np.ndarray = None,
    batch_size: int = 2000,
):
    """Batched version of `extract_roi_traces_parallel_intersected`.

    `mov_memmap` is a (T, H, W) int16 memmap (or any array with fancy-index
    support); the function never holds more than `batch_size` frames in float32.
    """
    T = mov_memmap.shape[0]
    num_total_rois = len(stat_array)
    if roi_indices_to_extract is None:
        tasks = np.arange(num_total_rois)
        print(f"No specific ROIs provided. Extracting traces for all {num_total_rois} ROIs (batched)...")
    else:
        tasks = roi_indices_to_extract
        print(f"Extracting traces for the {len(tasks)} selected ROIs (batched)...")

    roi_masks = _build_intersected_roi_masks(stat_array, global_dlight_mask, tasks)
    traces = np.full((len(tasks), T), np.nan, dtype=np.float32)

    for start in tqdm(range(0, T, batch_size), desc="Batched ROI trace extraction"):
        end = min(start + batch_size, T)
        mov_chunk = np.asarray(mov_memmap[start:end], dtype=np.float32)
        for i, m in enumerate(roi_masks):
            if m is None:
                continue
            yp, xp = m
            traces[i, start:end] = mov_chunk[:, yp, xp].mean(axis=1)
        del mov_chunk
    return traces


def extract_signal_and_background_traces_batched(
    mov_memmap,
    stat_array,
    global_dlight_mask,
    roi_indices_to_extract: np.ndarray = None,
    neuropil_dilation_iterations: int = 3,
    batch_size: int = 2000,
):
    """Batched version of `extract_signal_and_background_traces_parallel`."""
    T = mov_memmap.shape[0]
    num_total_rois = len(stat_array)
    if roi_indices_to_extract is None:
        tasks = np.arange(num_total_rois)
        print(f"Extracting traces for all {num_total_rois} ROIs (batched)...")
    else:
        tasks = roi_indices_to_extract
        print(f"Extracting traces for the {len(tasks)} selected ROIs (batched)...")

    masks = _build_signal_and_neuropil_masks(
        stat_array, global_dlight_mask, tasks, neuropil_dilation_iterations
    )
    signal_traces = np.full((len(tasks), T), np.nan, dtype=np.float32)
    bg_traces = np.full((len(tasks), T), np.nan, dtype=np.float32)

    for start in tqdm(range(0, T, batch_size), desc="Batched signal/bg extraction"):
        end = min(start + batch_size, T)
        mov_chunk = np.asarray(mov_memmap[start:end], dtype=np.float32)
        for i, (sig, bg) in enumerate(masks):
            if sig is not None:
                yp, xp = sig
                signal_traces[i, start:end] = mov_chunk[:, yp, xp].mean(axis=1)
            if bg is not None:
                yp, xp = bg
                bg_traces[i, start:end] = mov_chunk[:, yp, xp].mean(axis=1)
        del mov_chunk
    return signal_traces, bg_traces


def analyze_all_ROI_traces_batched(
    mov_memmap: np.ndarray,
    stat_array: np.ndarray,
    global_dlight_mask: np.ndarray,
    event_frames: np.ndarray,
    output_dir: str,
    event_name: str,
    trace_name: str,
    soma_indices: np.ndarray = None,
    pre_frames: int = 90,
    post_frames: int = 120,
    imaging_rate: float = 30.0,
    batch_size: int = 2000,
):
    """Batched orchestrator mirroring `analyze_all_ROI_traces`.

    Accepts an int16 memmap movie instead of a float32 array; saves outputs with
    the same schema as the non-batched version for drop-in compatibility.
    """
    print(f"\n--- Starting Soma Trace Analysis (batched) for '{event_name}' ---")
    output_path = Path(output_dir)
    figure_path = output_path / 'soma_response_maps'
    figure_path.mkdir(parents=True, exist_ok=True)

    if soma_indices is None:
        soma_indices = np.arange(len(stat_array))

    print(" > Extracting dLight-masked traces (batched)...")
    soma_traces_dlight_masked = extract_roi_traces_batched_intersected(
        mov_memmap, stat_array, global_dlight_mask,
        roi_indices_to_extract=soma_indices, batch_size=batch_size,
    )

    print("\nStep 2: Calculating trial-averaged traces...")
    mean_soma_dlight_masked, sem_soma_dlight_masked = calculate_trial_average_from_traces(
        soma_traces_dlight_masked, event_frames, pre_frames, post_frames
    )

    data_save_path = output_path / f'{trace_name}.npz'
    print(f"\nStep 3: Saving processed data to {data_save_path}")
    np.savez_compressed(
        data_save_path,
        soma_traces_dlight_masked=soma_traces_dlight_masked,
        mean_soma_dlight_masked=mean_soma_dlight_masked,
        sem_soma_dlight_masked=sem_soma_dlight_masked,
        soma_indices=soma_indices,
        pre_frames=pre_frames,
        post_frames=post_frames,
    )

    print("\nStep 4: Generating and saving summary heatmap...")
    plot_roi_response_map(
        mean_trace_trials=mean_soma_dlight_masked,
        stat_array=stat_array,
        roi_indices_to_extract=soma_indices,
        pre_frames=pre_frames,
        mean_img=global_dlight_mask,
        imaging_rate=imaging_rate,
        str_align=event_name,
        save_path=figure_path / f'{event_name}_soma_response_map.png',
    )

    print(f"\n--- Soma trace analysis (batched) for '{event_name}' complete. ---")
    return {
        'soma_traces_dlight_masked': soma_traces_dlight_masked,
        'mean_soma_dlight_masked': mean_soma_dlight_masked,
    }


def analyze_all_ROI_traces_and_background_batched(
    mov_memmap: np.ndarray,
    stat_array: np.ndarray,
    global_dlight_mask: np.ndarray,
    event_frames: np.ndarray,
    output_dir: str,
    event_name: str,
    trace_name: str,
    soma_indices: np.ndarray = None,
    pre_frames: int = 90,
    post_frames: int = 120,
    imaging_rate: float = 30.0,
    neuropil_dilation_iterations: int = 3,
    batch_size: int = 2000,
):
    """Batched orchestrator mirroring `analyze_all_ROI_traces_and_background`."""
    print(f"\n--- Starting Soma Trace Analysis (batched) for '{event_name}' ---")
    output_path = Path(output_dir)
    figure_path = output_path / 'soma_response_maps'
    figure_path.mkdir(parents=True, exist_ok=True)

    if soma_indices is None:
        soma_indices = np.arange(len(stat_array))

    print(" > Extracting dLight-masked signal + neuropil traces (batched)...")
    soma_traces_dlight_masked, bg_traces_dlight_masked = extract_signal_and_background_traces_batched(
        mov_memmap, stat_array, global_dlight_mask,
        roi_indices_to_extract=soma_indices,
        neuropil_dilation_iterations=neuropil_dilation_iterations,
        batch_size=batch_size,
    )

    print("\nStep 2: Calculating trial-averaged roi traces...")
    mean_soma_dlight_masked, sem_soma_dlight_masked = calculate_trial_average_from_traces(
        soma_traces_dlight_masked, event_frames, pre_frames, post_frames
    )

    print("\nStep 2: Calculating trial-averaged background traces...")
    mean_bg_dlight_masked, sem_bg_dlight_masked = calculate_trial_average_from_traces(
        bg_traces_dlight_masked, event_frames, pre_frames, post_frames
    )

    data_save_path = output_path / f'{trace_name}.npz'
    print(f"\nStep 3: Saving processed data to {data_save_path}")
    np.savez_compressed(
        data_save_path,
        soma_traces_dlight_masked=soma_traces_dlight_masked,
        bg_traces_dlight_masked=bg_traces_dlight_masked,
        mean_soma_dlight_masked=mean_soma_dlight_masked,
        sem_soma_dlight_masked=sem_soma_dlight_masked,
        mean_bg_dlight_masked=mean_bg_dlight_masked,
        sem_bg_dlight_masked=sem_bg_dlight_masked,
        soma_indices=soma_indices,
        pre_frames=pre_frames,
        post_frames=post_frames,
    )

    print("\nStep 4: Generating and saving summary heatmap...")
    plot_roi_response_map(
        mean_trace_trials=mean_soma_dlight_masked,
        stat_array=stat_array,
        roi_indices_to_extract=soma_indices,
        pre_frames=pre_frames,
        mean_img=global_dlight_mask,
        imaging_rate=imaging_rate,
        str_align=event_name,
        save_path=figure_path / f'{event_name}_soma_response_map.png',
    )
    plot_roi_response_map(
        mean_trace_trials=mean_bg_dlight_masked,
        stat_array=stat_array,
        roi_indices_to_extract=soma_indices,
        pre_frames=pre_frames,
        mean_img=global_dlight_mask,
        imaging_rate=imaging_rate,
        str_align=f'{event_name}, background',
        save_path=figure_path / f'{event_name}_background_response_map.png',
    )

    print(f"\n--- Soma trace analysis (batched) for '{event_name}' complete. ---")
    return {
        'soma_traces_dlight_masked': soma_traces_dlight_masked,
        'bg_traces_dlight_masked': bg_traces_dlight_masked,
        'mean_soma_dlight_masked': mean_soma_dlight_masked,
        'bg_soma_dlight_masked': mean_soma_dlight_masked,
    }


def analyze_all_ROI_traces(
    mov: np.ndarray,
    stat_array: np.ndarray,
    global_dlight_mask: np.ndarray,
    event_frames: np.ndarray,
    output_dir: str,
    event_name: str,
    trace_name: str,
    soma_indices: np.ndarray = None,
    pre_frames: int = 90,
    post_frames: int = 120,
    imaging_rate: float = 30.0
):
    """
    Orchestrates the analysis pipeline for a selected subset of ROIs (somas).

    This function performs:
    1. Extracts two types of traces for the selected somas:
        a) Intersected with a global dLight mask.
        b) Full ROI traces (not intersected).
    2. Calculates the trial-averaged response for the intersected traces.
    3. Saves all extracted and processed data to an .npz file.
    4. Generates and saves a summary heatmap of the trial-averaged responses.

    Args:
        mov (np.ndarray): The movie to extract traces from (e.g., movr).
        stat_array (np.ndarray): The full stat array from Suite2p.
        is_soma_mask (np.ndarray): A boolean mask identifying which ROIs are somas.
        global_dlight_mask (np.ndarray): The global mask for intersection.
        event_frames (np.ndarray): The frame numbers for the alignment event.
        output_dir (str): The main directory to save all results and figures.
        event_name (str): A descriptive name for the event (e.g., 'RunOnset').
        pre_frames, post_frames, imaging_rate: Analysis parameters.
    """
    print(f"\n--- Starting Soma Trace Analysis for '{event_name}' ---")
    
    # --- Setup Paths ---
    output_path = Path(output_dir)
    figure_path = output_path / 'soma_response_maps'
    figure_path.mkdir(parents=True, exist_ok=True)

    # # --- Step 1: Select Somas and Extract Traces ---
    # print("\nStep 1: Extracting traces for selected somas...")
    # if soma_indices.size == 0:
    #     print("No somas found in the provided mask. Aborting analysis.")
    #     return

    # print(f" > Found {len(soma_indices)} ROIs classified as somas.")
    if soma_indices is None: # extract all rois
        soma_indices = np.arange(len(stat_array))
    # Extract traces using the intersection with the dLight mask
    print(" > Extracting dLight-masked traces...")
    soma_traces_dlight_masked = extract_roi_traces_parallel_intersected(
        mov, stat_array, global_dlight_mask, soma_indices
    )
    
    # Extract full ROI traces (by intersecting with a mask of all ones)
    # print(" > Extracting full ROI traces...")
    # full_mask = np.ones_like(global_dlight_mask, dtype=bool)
    # soma_traces_full = extract_roi_traces_parallel_intersected(
        # mov, stat_array, full_mask, soma_indices
    # )
    
    # --- Step 2: Calculate Trial-Averaged Responses ---
    print("\nStep 2: Calculating trial-averaged traces...")
    mean_soma_dlight_masked, sem_soma_dlight_masked = calculate_trial_average_from_traces(
        soma_traces_dlight_masked, event_frames, pre_frames, post_frames
    )

    # --- Step 3: Save Processed Data ---
    data_save_path = output_path / f'{trace_name}.npz'
    print(f"\nStep 3: Saving processed data to {data_save_path}")
    np.savez_compressed(
        data_save_path,
        soma_traces_dlight_masked=soma_traces_dlight_masked,
        mean_soma_dlight_masked=mean_soma_dlight_masked,
        sem_soma_dlight_masked=sem_soma_dlight_masked,
        soma_indices=soma_indices,
        pre_frames=pre_frames,
        post_frames=post_frames
    )

    # --- Step 4: Generate and Save Summary Heatmap ---
    print("\nStep 4: Generating and saving summary heatmap...")
    # This assumes your plotting function is modified to accept a save path
    plot_roi_response_map(
        mean_trace_trials=mean_soma_dlight_masked,
        stat_array=stat_array,
        roi_indices_to_extract=soma_indices,
        pre_frames=pre_frames,
        mean_img=global_dlight_mask, 
        imaging_rate=imaging_rate,
        str_align=event_name,
        save_path=figure_path / f'{event_name}_soma_response_map.png'
    )
    
    print(f"\n--- Soma trace analysis for '{event_name}' complete. ---")
    
    # Return the key data for immediate use
    return {
        'soma_traces_dlight_masked': soma_traces_dlight_masked,
        'mean_soma_dlight_masked': mean_soma_dlight_masked
    }



def analyze_all_ROI_traces_and_background(
    mov: np.ndarray,
    stat_array: np.ndarray,
    global_dlight_mask: np.ndarray,
    event_frames: np.ndarray,
    output_dir: str,
    event_name: str,
    trace_name: str,
    soma_indices: np.ndarray = None,
    pre_frames: int = 90,
    post_frames: int = 120,
    imaging_rate: float = 30.0,
    neuropil_dilation_iterations: int = 3
):
    """
    Orchestrates the analysis pipeline for a selected subset of ROIs (somas).

    This function performs:
    1. Extracts two types of traces for the selected somas:
        a) Intersected with a global dLight mask.
        b) Full ROI traces (not intersected).
    2. Calculates the trial-averaged response for the intersected traces.
    3. Saves all extracted and processed data to an .npz file.
    4. Generates and saves a summary heatmap of the trial-averaged responses.

    Args:
        mov (np.ndarray): The movie to extract traces from (e.g., movr).
        stat_array (np.ndarray): The full stat array from Suite2p.
        is_soma_mask (np.ndarray): A boolean mask identifying which ROIs are somas.
        global_dlight_mask (np.ndarray): The global mask for intersection.
        event_frames (np.ndarray): The frame numbers for the alignment event.
        output_dir (str): The main directory to save all results and figures.
        event_name (str): A descriptive name for the event (e.g., 'RunOnset').
        pre_frames, post_frames, imaging_rate: Analysis parameters.
    """
    print(f"\n--- Starting Soma Trace Analysis for '{event_name}' ---")
    
    # --- Setup Paths ---
    output_path = Path(output_dir)
    figure_path = output_path / 'soma_response_maps'
    figure_path.mkdir(parents=True, exist_ok=True)

    # # --- Step 1: Select Somas and Extract Traces ---
    # print("\nStep 1: Extracting traces for selected somas...")
    # soma_indices = np.where(is_soma_mask)[0]
    # if soma_indices.size == 0:
    #     print("No somas found in the provided mask. Aborting analysis.")
    #     return

    # print(f" > Found {len(soma_indices)} ROIs classified as somas.")
    if soma_indices is None: # extract all rois
        soma_indices = np.arange(len(stat_array))
    # Extract roi and background traces using the intersection with the dLight mask
    print(" > Extracting dLight-masked traces...")
    soma_traces_dlight_masked, bg_traces_dlight_masked = extract_signal_and_background_traces_parallel(
        mov, stat_array, global_dlight_mask, soma_indices, neuropil_dilation_iterations
    )
    
    # --- Step 2: Calculate Trial-Averaged Responses ---
    print("\nStep 2: Calculating trial-averaged roi traces...")
    mean_soma_dlight_masked, sem_soma_dlight_masked = calculate_trial_average_from_traces(
        soma_traces_dlight_masked, event_frames, pre_frames, post_frames
    )

    print("\nStep 2: Calculating trial-averaged background traces...")
    mean_bg_dlight_masked, sem_bg_dlight_masked = calculate_trial_average_from_traces(
        bg_traces_dlight_masked, event_frames, pre_frames, post_frames
    )

    # --- Step 3: Save Processed Data ---
    data_save_path = output_path / f'{trace_name}.npz'
    print(f"\nStep 3: Saving processed data to {data_save_path}")
    np.savez_compressed(
        data_save_path,
        soma_traces_dlight_masked=soma_traces_dlight_masked,
        bg_traces_dlight_masked=bg_traces_dlight_masked,
        mean_soma_dlight_masked=mean_soma_dlight_masked,
        sem_soma_dlight_masked=sem_soma_dlight_masked,
        mean_bg_dlight_masked=mean_bg_dlight_masked,
        sem_bg_dlight_masked=sem_bg_dlight_masked,
        soma_indices=soma_indices,
        pre_frames=pre_frames,
        post_frames=post_frames
    )

    # --- Step 4: Generate and Save Summary Heatmap ---
    print("\nStep 4: Generating and saving summary heatmap...")
    # This assumes your plotting function is modified to accept a save path
    plot_roi_response_map(
        mean_trace_trials=mean_soma_dlight_masked,
        stat_array=stat_array,
        roi_indices_to_extract=soma_indices,
        pre_frames=pre_frames,
        mean_img=global_dlight_mask, 
        imaging_rate=imaging_rate,
        str_align=event_name,
        save_path=figure_path / f'{event_name}_soma_response_map.png'
    )

    plot_roi_response_map(
        mean_trace_trials=mean_bg_dlight_masked,
        stat_array=stat_array,
        roi_indices_to_extract=soma_indices,
        pre_frames=pre_frames,
        mean_img=global_dlight_mask, 
        imaging_rate=imaging_rate,
        str_align=f'{event_name}, background',
        save_path=figure_path / f'{event_name}_background_response_map.png'
    )
    
    print(f"\n--- Soma trace analysis for '{event_name}' complete. ---")
    
    # Return the key data for immediate use
    return {
        'soma_traces_dlight_masked': soma_traces_dlight_masked,
        'bg_traces_dlight_masked': bg_traces_dlight_masked,
        'mean_soma_dlight_masked': mean_soma_dlight_masked,
        'bg_soma_dlight_masked': mean_soma_dlight_masked
    }


