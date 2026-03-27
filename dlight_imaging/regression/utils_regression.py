# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:00:18 2026

@author: CaoJ
"""
from pathlib import Path
import numpy as np
import pandas as pd 
from joblib import Parallel, delayed
from tqdm import tqdm
from common.mask.generate_masks import extract_axon_dlight_masks, extract_geco_dlight_masks
from . import align_beh_imaging
from sklearn.linear_model import LinearRegression

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
    data_bin = data_path + file_name
    
    # Use a memory map to load the binary data.
    reg_data = np.memmap(data_bin, mode='r', dtype='int16', shape=(n_frames, height, width))
    
    # Convert to float32.
    mov = reg_data.astype('float32')
    return mov

def load_masks_axon_dlight(rec, OUT_DIR_RAW_REGRESS, dilation_steps = (0, 2, 4, 6, 8, 10),
                           constrain_to_grid = True,
                           visualize_neu = False,
                           visualize_dilation = False,
                           visualize_regressor = False):
    p_masks = OUT_DIR_RAW_REGRESS / rf'{rec}'/'masks'
    if not (p_masks/'dilated_global_axon_k=10.npy').exists():
        if not p_masks.exists():
            p_masks.mkdir(parents=True, exist_ok=False)
        extract_axon_dlight_masks(rec, p_masks, dilation_steps,
                                  constrain_to_grid,
                                  visualize_neu,
                                  visualize_dilation,
                                  visualize_regressor)
    return p_masks

def load_masks_geco_dlight(rec, OUT_DIR_RAW_REGRESS):
    p_masks = OUT_DIR_RAW_REGRESS / rf'{rec}'/'masks'
    if not p_masks.exists():
        p_masks.mkdir(parents=True, exist_ok=False)
        extract_geco_dlight_masks(rec, p_masks,)
    return p_masks

def load_behaviour(rec, OUT_DIR_REGRESS, num_frames):
    p_beh_file = OUT_DIR_REGRESS.parent / 'behaviour_profile' / f'{rec}.pkl'
    ourput_dir = OUT_DIR_REGRESS / f'{rec}'/ f"{rec}_processed_behavior.npz"
    if not ourput_dir.exists():
        align_beh_imaging.process_and_save_behavioral_data(
            input_pkl_path=p_beh_file,
            output_dir=ourput_dir,
            num_imaging_frames=num_frames,
            imaging_rate=30.0,
            do_plots=False
        )
    beh_data = np.load(ourput_dir)
    return beh_data

def get_valid_grids(res_traces):
    """
    Get list of all valid grids (those without NaN traces).

    Args:
        res_traces: Loaded npz file with trace arrays

    Returns:
        list: List of tuples (grid_y, grid_x) for valid grids
    """
    # corrected_dlight = res_traces['corrected_dlight']
    n_blocks_y, n_blocks_x, _ = res_traces.shape

    valid_grids = []
    for gy in range(n_blocks_y):
        for gx in range(n_blocks_x):
            if not np.all(np.isnan(res_traces[gy, gx, :])):
                valid_grids.append((gy, gx))

    print(f"Found {len(valid_grids)} valid grids out of {n_blocks_y * n_blocks_x} total grids")
    return valid_grids

def get_axon_grids(global_axon_mask, n_grid=32):
    """
    Count axon pixels per grid block in a 2D mask (H, W), return valid grids and counts.
    """
    H, W = global_axon_mask.shape
    grid_h = H // n_grid
    grid_w = W // n_grid

    valid_grids = []
    pix_counts = np.zeros((n_grid, n_grid), dtype=np.int32)

    for gy in range(n_grid):
        y0 = gy * grid_h
        y1 = (gy + 1) * grid_h if gy < n_grid - 1 else H  # include remainder in last block

        for gx in range(n_grid):
            x0 = gx * grid_w
            x1 = (gx + 1) * grid_w if gx < n_grid - 1 else W

            block = global_axon_mask[y0:y1, x0:x1]
            cnt = int(np.count_nonzero(block))  # True pixels
            pix_counts[gy, gx] = cnt

            if cnt > 0:
                valid_grids.append((gy, gx))

    print(f"Found {len(valid_grids)} valid grids out of {n_grid * n_grid} total grids")
    return valid_grids, pix_counts

# trace extraction functions
def _trace_extraction_worker(block_idx, mov, movr, 
                             global_mask_green, global_mask_red,
                             dlight_regressor_mask, neuropil_mask,
                             grid_size=16):

    H, W = mov.shape[1], mov.shape[2]
    n_blocks_x = W // grid_size
    row, col = block_idx // n_blocks_x, block_idx % n_blocks_x

    block_slice = tuple(slice(r * grid_size, (r + 1) * grid_size) for r in [row, col])

    # --- 1. Extract Full-Length Traces ---
    local_green_mask = global_mask_green[block_slice]
    n_green_mask = int(np.sum(local_green_mask))
    if n_green_mask < 10: return None
    green_mask_full = np.zeros_like(global_mask_green)
    green_mask_full[block_slice] = local_green_mask
    original_green_trace = mov[:, green_mask_full].mean(axis=1)

    local_dlight_regressor_mask = dlight_regressor_mask[block_slice]
    n_regressor_mask = int(np.sum(local_dlight_regressor_mask))
    if n_regressor_mask < 10: return None
    dlight_regressor_mask_full = np.zeros_like(dlight_regressor_mask)
    dlight_regressor_mask_full[block_slice] = local_dlight_regressor_mask
    dlight_regressor_trace = mov[:, dlight_regressor_mask_full].mean(axis=1)

    local_red_mask = global_mask_red[block_slice]
    n_red_mask = int(np.sum(local_red_mask))
    if n_red_mask < 10: return None
    red_mask_full = np.zeros_like(global_mask_red)
    red_mask_full[block_slice] = local_red_mask
    red_trace = movr[:, red_mask_full].mean(axis=1)
    
    local_neuropil_mask = neuropil_mask[row, col, :, :]
    n_neuropil_mask = int(np.sum(local_neuropil_mask))
    if n_neuropil_mask < 10: return None
    neuropil_trace = mov[:, local_neuropil_mask].mean(axis=1)
    neuropil_trace_red = movr[:, local_neuropil_mask].mean(axis=1)
    
    return {'original_dlight': original_green_trace,
            'red': red_trace, 'dlight_regressor': dlight_regressor_trace,
            'neuropil': neuropil_trace,
            'red_neuropil': neuropil_trace_red,}

# trace extraction functions
def _trace_extraction_worker_grid_free(block_idx, mov, movr,
                                       global_mask_dilated_axon,
                                       global_mask_red, global_mask_dlight,
                                       dlight_regressor_mask, neuropil_mask,
                                       grid_size=16):

    H, W = mov.shape[1], mov.shape[2]
    n_blocks_x = W // grid_size
    row, col = block_idx // n_blocks_x, block_idx % n_blocks_x

    block_slice = tuple(slice(r * grid_size, (r + 1) * grid_size) for r in [row, col])

    # --- 1. Extract Full-Length Traces ---
    local_dilated_axon_mask = global_mask_dilated_axon[row, col, :, :]
    local_green_mask = (local_dilated_axon_mask)&(global_mask_dlight)
    n_green_mask = int(np.sum(local_green_mask))
    if n_green_mask < 10: return None
    # green_mask_full = np.zeros_like(global_mask_green)
    # green_mask_full[block_slice] = local_green_mask
    original_green_trace = mov[:, local_green_mask].mean(axis=1)

    local_dlight_regressor_mask = dlight_regressor_mask[row, col, :, :]
    n_regressor_mask = int(np.sum(local_dlight_regressor_mask))
    if n_regressor_mask < 10: return None
    # dlight_regressor_mask_full = np.zeros_like(dlight_regressor_mask)
    # dlight_regressor_mask_full[block_slice] = local_dlight_regressor_mask
    dlight_regressor_trace = mov[:, local_dlight_regressor_mask].mean(axis=1)

    local_red_mask = global_mask_red[row, col, :, :]
    n_red_mask = int(np.sum(local_red_mask))
    if n_red_mask < 10: return None
    # red_mask_full = np.zeros_like(global_mask_red)
    # red_mask_full[block_slice] = local_red_mask
    red_trace = movr[:, local_red_mask].mean(axis=1)
    
    local_neuropil_mask = neuropil_mask[row, col, :, :]
    n_neuropil_mask = int(np.sum(local_neuropil_mask))
    if n_neuropil_mask < 10: return None
    neuropil_trace = mov[:, local_neuropil_mask].mean(axis=1)
    neuropil_trace_red = movr[:, local_neuropil_mask].mean(axis=1)
    
    return {'original_dlight': original_green_trace,
            'red': red_trace, 'dlight_regressor': dlight_regressor_trace,
            'neuropil': neuropil_trace,
            'red_neuropil': neuropil_trace_red,}

def traces_extraction_parallel(mov, movr, global_mask_dilated_axon, 
                               global_mask_red, global_mask_dlight,
                               dlight_regressor_mask, neuropil_mask,
                               out_dir=None,
                               grid_free=False,
                               **kwargs):
    """
    Main orchestrator that runs single-trial red-channel correction in parallel.
    """
    T, H, W = mov.shape
    grid_size = kwargs.get('grid_size', 16)
    n_blocks_y = H // grid_size
    n_blocks_x = W // grid_size
    num_blocks = n_blocks_y * n_blocks_x
    
    n_jobs = kwargs.pop('n_jobs', -1)
    
    # Extract only the parameters needed by the worker function
    worker_params = {
        'grid_size': kwargs.get('grid_size', 16),
    }
    if grid_free:
        print('!!!!grid-free ROIs')
        all_traces = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_trace_extraction_worker_grid_free)(i, mov, movr, 
                                             global_mask_dilated_axon, 
                                             global_mask_red, global_mask_dlight,
                                             dlight_regressor_mask, neuropil_mask,
                                             **worker_params) 
            for i in tqdm(range(num_blocks), desc="extracting traces from each block..."))
    else:
        global_mask_green = (global_mask_dilated_axon)&(global_mask_dlight)
        all_traces = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_trace_extraction_worker)(i, mov, movr, 
                                             global_mask_green, global_mask_red,
                                             dlight_regressor_mask, neuropil_mask,
                                             **worker_params) 
            for i in tqdm(range(num_blocks), desc="extracting traces from each block..."))
        
    
    # Unpack results into grid arrays
    original_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    red_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    dlight_regressor_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    neuropil_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    red_neuropil_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    
    for i, res_dict in enumerate(all_traces):
        row, col = i // n_blocks_x, i % n_blocks_x
        if res_dict is not None and type(res_dict) is not tuple: # handle some weird exception
            original_traces_grid[row, col, :] = res_dict['original_dlight']
            dlight_regressor_traces_grid[row, col, :] = res_dict['dlight_regressor']
            red_traces_grid[row, col, :] = res_dict['red']
            neuropil_traces_grid[row, col, :] = res_dict['neuropil']
            red_neuropil_traces_grid[row, col, :] = res_dict['red_neuropil']
    

    data_save_path = Path(out_dir)
    print(f"\nSaving extracted traces to {data_save_path}")
    np.savez_compressed(
        data_save_path,
        original_dlight=original_traces_grid,
        dlight_regressor=dlight_regressor_traces_grid,
        red_trace=red_traces_grid,
        neuropil_dlight=neuropil_traces_grid,
        neuropil_red=red_neuropil_traces_grid
    )

def traces_extraction_parallel_roi(mov, movr, global_mask_green, global_mask_red,
                                   dlight_regressor_mask, neuropil_mask,
                                   out_dir=None,
                                   **kwargs):
    """
    Main orchestrator that runs single-trial red-channel correction in parallel.
    """
    T, H, W = mov.shape
    grid_size = kwargs.get('grid_size', 16)
    n_blocks_y = H // grid_size
    n_blocks_x = W // grid_size
    num_blocks = n_blocks_y * n_blocks_x
    
    n_jobs = kwargs.pop('n_jobs', -1)
    
    # Extract only the parameters needed by the worker function
    worker_params = {
        'grid_size': kwargs.get('grid_size', 16),
    }

    all_traces = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_trace_extraction_worker_grid_free)(i, mov, movr, 
                                         global_mask_green, global_mask_red,
                                         dlight_regressor_mask, neuropil_mask,
                                         **worker_params) 
        for i in tqdm(range(num_blocks), desc="extracting traces from each block..."))
    
    # Unpack results into grid arrays
    original_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    red_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    dlight_regressor_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    neuropil_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    red_neuropil_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    
    for i, res_dict in enumerate(all_traces):
        row, col = i // n_blocks_x, i % n_blocks_x
        if res_dict is not None and type(res_dict) is not tuple: # handle some weird exception
            original_traces_grid[row, col, :] = res_dict['original_dlight']
            dlight_regressor_traces_grid[row, col, :] = res_dict['dlight_regressor']
            red_traces_grid[row, col, :] = res_dict['red']
            neuropil_traces_grid[row, col, :] = res_dict['neuropil']
            red_neuropil_traces_grid[row, col, :] = res_dict['red_neuropil']
    

    data_save_path = Path(out_dir)
    print(f"\nSaving extracted traces to {data_save_path}")
    np.savez_compressed(
        data_save_path,
        original_dlight=original_traces_grid,
        dlight_regressor=dlight_regressor_traces_grid,
        red_trace=red_traces_grid,
        neuropil_dlight=neuropil_traces_grid,
        neuropil_red=red_neuropil_traces_grid
    )
    
## single trial trace regression functions
def _single_trial_regression_worker(block_idx,
                                   original_green_traces,
                                   green_regressor_traces,
                                   red_regressor_traces,
                                   event_frames, pre_frames, post_frames):
    
    T = original_green_traces.shape[-1]
    n_blocks_x = original_green_traces.shape[1]
    row, col = block_idx // n_blocks_x, block_idx % n_blocks_x
    
    # Get a list of onsets that are valid within the movie bounds
    valid_event_frames = sorted([e for e in event_frames if (e - pre_frames >= 0 and e + post_frames < T)])
    if not valid_event_frames:
        return None
    
    original_green_trace  = original_green_traces[row, col]
    green_regressor_trace = green_regressor_traces[row, col]
    red_regressor_trace   = red_regressor_traces[row, col]
    
    if np.any((np.isnan(original_green_trace)|
        np.isnan(green_regressor_trace)|
        np.isnan(red_regressor_trace))):
        final_corrected_trace = np.nan
        qc_dict = None
    else:
        # --- 3. Initialize Accumulator Arrays ---
        # We will accumulate sums and then divide by counts for a proper average.
        sum_corrected_trace = np.zeros(T)
        sum_artifact_trace = np.zeros(T)
        overlap_counts = np.zeros(T)
    
        # Lists to collect regression metrics across all events
        r2_list = []
        b1_list = []
        b2_list = []
        event_frame_list = []
        start_frame_list = []
        end_frame_list = []
    
        # --- 4: Loop through trials and perform "Correct-and-Paste" ---    
        prev_trial_end = 0
        for i, onset in enumerate(valid_event_frames):
            # Define the start of the current analysis window
            # For the first trial, it starts at frame 0.
            # For subsequent trials, it starts where the last trial's post-window ended.
            start_frame = prev_trial_end if i > 0 else 0
            
            # The end of the analysis window is the end of the current trial's post-event period
            end_frame = onset + post_frames
            
            # Ensure we don't try to create a window of zero or negative length
            if start_frame >= end_frame:
                prev_trial_end = end_frame
                continue
    
            # --- Extract Slices for this specific window ---
            y_target = original_green_trace[start_frame:end_frame]
            green_regressor = green_regressor_trace[start_frame:end_frame]
            red_regressor = red_regressor_trace[start_frame:end_frame]
    
            # --- Build and Fit a Model for this window ONLY ---
            X_regressors = np.vstack([green_regressor, red_regressor]).T
            
            try:
                # Non-negative least squares
                model = LinearRegression().fit(X_regressors, y_target) 
                # estimated_artifact_slice = model.predict(X_regressors)
                # corrected_slice = y_target - estimated_artifact_slice
                
                # The Fix: Do not subtract the intercept. 
                # Subtract only the time-varying components: 
                # Gcleaned =Graw −[β1* (R(t)−mean(R(t))) + β2 * (N(t)-mean(N(t)))] 
                # Beta 1: neuropil regressor constant; Beta 2: red channel regressor constant
                b1, b2 = model.coef_[0], model.coef_[1]
    
                # Collect regression metrics for this event
                r2_list.append(model.score(X_regressors, y_target))
                b1_list.append(b1)
                b2_list.append(b2)
                event_frame_list.append(onset)
                start_frame_list.append(start_frame)
                end_frame_list.append(end_frame)
                
                # estimated time-varying artifact components
                est_art = b1*(green_regressor - np.nanmean(green_regressor)) \
                            + b2*(red_regressor - np.nanmean(red_regressor))
                
                corrected_slice = y_target - est_art
                
                # --- "Add" the results to the full-length sum arrays ---
                sum_corrected_trace[start_frame:end_frame] += corrected_slice
                sum_artifact_trace[start_frame:end_frame] += est_art
                overlap_counts[start_frame:end_frame] += 1
    
            except ValueError as e:
                print(f"Warning: Could not fit model for trial {i} (frames {start_frame}-{end_frame}). Skipping. Error: {e}")
    
            # Update the end point for the next trial's start
            prev_trial_end = end_frame
    
        # --- 5: Final Averaging and Gap Filling ---
        final_corrected_trace = np.copy(original_green_trace) # Start with original trace
        estimated_full_artifact = np.zeros(T) # Artifact is zero in gaps
    
        # Find the indices where at least one trial contributed
        valid_indices = overlap_counts > 0
        
        # Calculate the average for these valid indices
        final_corrected_trace[valid_indices] = sum_corrected_trace[valid_indices] / overlap_counts[valid_indices]
        estimated_full_artifact[valid_indices] = sum_artifact_trace[valid_indices] / overlap_counts[valid_indices]
    
        # Build QC dictionary for this block
        qc_dict = {
            'grid_y': row,
            'grid_x': col,
            'r2': r2_list,
            'b1': b1_list,
            'b2': b2_list,
            'event_frame': event_frame_list,
            'start_frame': start_frame_list,
            'end_frame': end_frame_list
        }
    return final_corrected_trace, qc_dict

def single_trial_regression_parallel(original_green_traces,
                                     green_regressor_traces,
                                     red_regressor_traces,
                                     event_frames,
                                     out_dir=None,
                                     **kwargs):
    """
    Main orchestrator that runs single-trial red-channel correction in parallel.
    """
    n_blocks_y, n_blocks_x, T = original_green_traces.shape
    num_blocks = n_blocks_y * n_blocks_x
    
    n_jobs = kwargs.pop('n_jobs', -1)
    
    # Extract only the parameters needed by the worker function
    worker_params = {
        'pre_frames': kwargs.get('pre_frames', 90),
        'post_frames': kwargs.get('post_frames', 120)
    }
    
    regression_res = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_single_trial_regression_worker)
        (i, 
         original_green_traces,
         green_regressor_traces,
         red_regressor_traces, event_frames, **worker_params) 
         for i in tqdm(range(num_blocks), desc="single trial regression for each block..."))
    
    # Unpack: each worker returns (corrected_trace, qc_dict)
    corrected_traces = [r[0] if r is not None else None for r in regression_res]
    all_qc_dicts = [r[1] if r is not None else None for r in regression_res]
    
    # Unpack results into grid arrays
    corrected_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    
    for i, corrected_trace in enumerate(corrected_traces):
        row, col = i // n_blocks_x, i % n_blocks_x
        if corrected_trace is not None and type(corrected_trace) is not tuple: # handle some weird exception
            corrected_traces_grid[row, col, :] = corrected_trace
    
    # Build regression QC DataFrame (only available when BG_regressor is None)
    if all_qc_dicts is not None:
        # Filter out None entries and build DataFrame
        valid_qc_dicts = [qc for qc in all_qc_dicts if qc is not None]
        regress_qc_df = pd.DataFrame(valid_qc_dicts)
    else:
        regress_qc_df = None
    
    if out_dir is not None:
        data_save_path = out_dir/'corrected_dlight_trace.npy'
        print(f"\nSaving extracted traces to {data_save_path}")
        np.save(data_save_path, corrected_traces_grid)
        
        # Save regression QC data as CSV (DataFrame format)
        if regress_qc_df is not None:
            qc_save_path = out_dir/'regression_qc.npy'
            regress_qc_df.to_pickle(qc_save_path)
            print(f"  Saved regression QC data to {qc_save_path}")