# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 14:46:49 2026

@author: Jingyu Cao

dLight regression script
"""
import sys
import os
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
if r"Z:\Jingyu\code_mpfi_Jingyu" not in sys.path:
    sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")

from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst    
import dlight_imaging.regression.utils_regression as utl
#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_grid_free_dilation'
OUT_DIR_FIG = ''


# DILATION_STEPS = (0, 2, 4, 6, 8, 10)
# DILATION_STEPS = ( 1, 3, 5, 7, 9)
DILATION_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
# DILATION_STEPS = (0, )

correction_params = {
    'grid_size': 16,
    'pre_frames': 90,
    'post_frames': 120,
    'n_jobs': -1
}
#%%
regression_name ='single_trial_regression'
original_stdout = sys.stdout
error_lst = []
     
for rec in tqdm(rec_lst[:21]):
    # try:
    start_time = time.time()
    print(f'\n{rec}_running regression pipeline...')
    out_path_regression = OUR_DIR_REGRESS / f"{rec}"
    if not os.path.exists(out_path_regression):
        os.makedirs(out_path_regression)
        
    log_file = open(out_path_regression/f'{rec}_regression_processing_log.txt', 'w')
    log_file.write(f"\n{'='*50}\nRecording: {rec}\n{'='*50}\n")
    log_file.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.stdout = log_file
    
    anm, date, ss = rec.split('-')
    p_rec = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\RegOnly\suite2p\plane0"
    # beh = pd.read_pickle(OUT_DIR_RAW_DATA / 'behaviour_profile' /  f'{rec}.pkl')
    
    # load suite2p ops
    suite2p_ops = np.load(p_rec+r'\ops.npy', allow_pickle=True).item()
    # load masks path or generate mask file    
    p_masks = utl.load_masks_axon_dlight(rec, OUR_DIR_REGRESS, dilation_steps=DILATION_STEPS,
                                         constrain_to_grid = False,
                                         visualize_neu = True,
                                         visualize_dilation = True,
                                         visualize_regressor = True)
    # load or generate behaviour file
    beh_data = utl.load_behaviour(rec, OUR_DIR_REGRESS, suite2p_ops['nframes'])
    
    ## run regression

    global_axon_mask = np.load(p_masks / 'dilated_global_axon_k=0.npy')
    global_dlight_mask = np.load(p_masks / 'global_dlight_mask_enhanced.npy')
    
    # load ch1 and ch2 movie
    nframes = suite2p_ops['nframes']
    # dlight
    mov = utl.load_bin_file(p_rec, r'\data.bin', n_frames=nframes, height=512, width=512)
    # red
    movr = utl.load_bin_file(p_rec, r"\data_chan2.bin", n_frames=nframes, height=512, width=512)
    for k_size in DILATION_STEPS:
        print(f'Processing dilation step = {k_size}...')
        p_dilation_results = OUR_DIR_REGRESS / rec / regression_name / r'dilation_k={}'.format(k_size)
        if not p_dilation_results.exists():
            p_dilation_results.mkdir(parents=True)
        if not (out_path_regression/
                f'{rec}_raw_traces_k={k_size}.npz').exists():
            print('\nextracting raw traces...')
            
            # nframes = suite2p_ops['nframes']
            # nframes = 1000
            
            # # load ch1 and ch2 movie
            # # dlight
            # mov = utl.load_bin_file(p_rec, r'\data.bin', n_frames=nframes, height=512, width=512)
            # # red
            # movr = utl.load_bin_file(p_rec, r"\data_chan2.bin", n_frames=nframes, height=512, width=512)
            
            global_mask_dilated_axon = np.load(p_masks / f'dilated_global_axon_k={k_size}.npy')
            # dilated_global_axon_and_dlight_mask = (global_axon_mask_dilated)&(global_dlight_mask)
            dlight_regressor_mask = np.load(p_masks / f'dlight_regressor_fiber_dilation_k={k_size}.npy')
            neuropil_mask = np.load(p_masks / f'fiber_neuropil_masks_dilation_k={k_size}.npy')
            # extract traces
            utl.traces_extraction_parallel(mov, movr, 
                                           global_mask_dilated_axon=global_mask_dilated_axon,
                                           global_mask_red=global_mask_dilated_axon,
                                           global_mask_dlight = global_dlight_mask,
                                           dlight_regressor_mask=dlight_regressor_mask, 
                                           neuropil_mask=neuropil_mask,
                                           grid_free = True,
                                           out_dir=out_path_regression/f'{rec}_raw_traces_k={k_size}.npz',)
         
        print('loadind extraced traces...')
        raw_traces = np.load(out_path_regression/f'{rec}_raw_traces_k={k_size}.npz')
        original_dlight_traces  = raw_traces['original_dlight']
        dlight_neu = raw_traces['neuropil_dlight']
        # !!!subtract neuropil before regression!!!
        dlight_corr = original_dlight_traces -0.2*dlight_neu
        dlight_regressor_traces = raw_traces['dlight_regressor']
        raw_traces.close()
        # always use the original axon red traces as red regressor
        red_regressor_traces = np.load(out_path_regression/
                                       f'{rec}_raw_traces_k=0.npz')['red_trace']
        
        p_res = p_dilation_results / f'{regression_name}_res_traces.npz'
        
        # if not p_res.exists():
        # Run the regression pipeline
        utl.single_trial_regression_parallel(original_green_traces=dlight_corr,
                                             green_regressor_traces=dlight_regressor_traces,
                                             red_regressor_traces=red_regressor_traces,
                                             event_frames=beh_data['run_onset_frames'],
                                             out_dir=p_dilation_results,
                                             **correction_params)
    del mov, movr 
                
       
    
       
        
    print(f"Completed: {rec}")
    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    print(f"Total time: {int(mins)}m {secs:.1f}s")
    log_file.close()
    sys.stdout = original_stdout
    print(f"Completed: {rec} - Time: {int(mins)}m {secs:.1f}s")
    
    # except Exception as e:
    #     sys.stdout = original_stdout
    #     error_lst.append(rec)
    #     print(f"Error processing {rec}: {str(e)}")
    #     continue
        
