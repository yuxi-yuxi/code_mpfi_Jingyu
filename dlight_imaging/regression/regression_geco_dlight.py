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
import pandas as pd
import time
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
# import matplotlib.pyplot as plt
# if r"Z:\Jingyu\code_mpfi_Jingyu" not in sys.path:
#     sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")

from common.utils_imaging import percentile_dff
from dlight_imaging.regression.utils_regression_geco import Extract_dlight_masked_GECO_ROI_traces, Regression_Red_From_Green_ROIs_Single_Trial_geco
# from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst    
import dlight_imaging.regression.utils_regression as utl
from common.robust_sd_filter import calculate_rsd_along_axis
#%%
def extract_traces(mov, movr, stat, soma_indices, global_dlight_mask, event_frames, output_dir,
                   neuropil_dilation_iterations,
                   batch_size
                   ):
    
    # extract GECO traces (batched: casts only `batch_size` frames to float32 at
    # a time, so the full float32 stack is never materialized in RAM)
    Extract_dlight_masked_GECO_ROI_traces.analyze_all_ROI_traces_batched(
        mov_memmap=movr,
        stat_array=stat,
        soma_indices=soma_indices, # set as None to extract all rois
        global_dlight_mask=global_dlight_mask,
        event_frames=event_frames,
        output_dir=output_dir,
        event_name='geco_run_onset_dlight_mask',
        trace_name='geco_roi_dlight_mask_geco_traces',
        pre_frames = 90,
        post_frames = 120,
        imaging_rate = 30.0,
        batch_size = batch_size,
    );
    # extract dlight traces
    Extract_dlight_masked_GECO_ROI_traces.analyze_all_ROI_traces_and_background_batched(
        mov_memmap=mov,
        stat_array=stat,
        soma_indices=soma_indices, # set as None to extract all rois
        global_dlight_mask=global_dlight_mask,
        neuropil_dilation_iterations=neuropil_dilation_iterations,
        event_frames=event_frames,
        output_dir=output_dir,
        event_name='dlight_run_onset_dlight_mask',
        trace_name='geco_roi_dlight_mask_dlight_traces',
        pre_frames = 90,
        post_frames = 120,
        imaging_rate = 30.0,
        batch_size = batch_size,

    )
    
def run_single_trial_regression_geco(p_suite2p_geco, p_suite2p_movie, regression_name,
                                     OUR_DIR_REGRESS, OUT_DIR_FIG,
                                     neuropil_dilation_iterations=3,
                                     calculate_dff=0,
                                     batch_size=2000):
    start_time = time.time()
    original_stdout = sys.stdout
    
    out_path_regression = OUR_DIR_REGRESS / f'{rec}' / regression_name
    if not os.path.exists(out_path_regression):
        os.makedirs(out_path_regression)
    fig_out_dir=OUT_DIR_FIG / f'{rec}' / f'{regression_name}_dashboard_check'
    if not os.path.exists(fig_out_dir):
        os.makedirs(fig_out_dir)
    
    # load suite2p ops
    suite2p_ops = np.load(p_suite2p_geco+r'\ops.npy', allow_pickle=True).item()
    Fneu_geco = np.load(p_suite2p_geco+r'\Fneu.npy')
    Fneu_dlight = np.load(p_suite2p_geco+r'\Fneu_chan2.npy')
    nframes = suite2p_ops['nframes']
    # nframes = 1000
    # load masks path or generate mask file    
    p_masks = utl.load_masks_geco_dlight(rec, OUR_DIR_REGRESS, p_suite2p_geco)
    global_dlight_mask = np.load(p_masks / 
                               'global_dlight_mask_enhanced.npy')
    is_soma =  np.load(p_masks/r'soma_class.npz')['is_soma']
    soma_indices = np.where(is_soma>0)[0]
    
    # load or generate behaviour file
    beh_data = utl.load_behaviour(rec, OUR_DIR_REGRESS, suite2p_ops['nframes'])
    run_onset_frames =beh_data['run_onset_frames']
    
    log_file = open(out_path_regression/'regression_processing_log.txt', 'w')
    log_file.write(f"\n{'='*50}\nRecording: {rec}\n{'='*50}\n")
    log_file.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.stdout = log_file
    
    
    # extract GECO and dlight traces
    p_dlight_trace = out_path_regression / 'geco_roi_dlight_mask_dlight_traces.npz'
    if not p_dlight_trace.exists():
        # load info for GECO ROIs detected by suite2p 
        stat_path = p_suite2p_geco + r"\stat.npy"
        stat = np.load(stat_path, allow_pickle=True)
        # load ch1 and ch2 movie as int16 memmaps (no full float32 copy in RAM;
        # the batched extractors cast per-batch just in time)
        movr = utl.load_bin_file_memmap(p_suite2p_movie, r'\data.bin', n_frames=nframes, height=512, width=512)
        mov = utl.load_bin_file_memmap(p_suite2p_movie, r"\data_chan2.bin", n_frames=nframes, height=512, width=512)
        extract_traces(mov, movr, stat, 
                       soma_indices=None, # set as None to extract all rois 
                       global_dlight_mask=global_dlight_mask, 
                       event_frames=run_onset_frames, 
                       output_dir=out_path_regression,
                       neuropil_dilation_iterations=3,
                       batch_size=batch_size
                       )
        
        del mov, movr
    else:
        print('signal extacted, loading traces...')
    
    ### run single trial regression

    # load extracted traces 
    # Load geco traces
    geco_traces = np.load(out_path_regression / 
                          'geco_roi_dlight_mask_geco_traces.npz')
    geco_soma_traces = geco_traces['soma_traces_dlight_masked']
    geco_traces.close()
    # Load dlight traces
    dlight_traces = np.load(out_path_regression / 
                          'geco_roi_dlight_mask_dlight_traces.npz')
    dlight_soma_traces = dlight_traces['soma_traces_dlight_masked']
    dlight_bg_traces = dlight_traces['bg_traces_dlight_masked']
    # mean_dlight_dlight_masked = dlight_traces['mean_soma_dlight_masked']
    dlight_traces.close()
    
    p_regress_res = out_path_regression/ f'{regression_name}_res_traces.npz'
    if not p_regress_res.exists():
        ### Regression of all the ROIs in green channel using red channel trace (suite2p_roi_mask & global_dlight_mask)
        pre = 90
        post = 120
        dlight_corr = dlight_soma_traces-0.2*Fneu_dlight
        geco_soma_traces_corr = geco_soma_traces-0.7*Fneu_geco
        geco_traces_dff = percentile_dff(geco_soma_traces_corr, q=20)
        # np.save(out_path_regression / 'dff_geco.npy', geco_traces_dff)
        geco_traces_rsd = calculate_rsd_along_axis(geco_traces_dff, axis=-1, gpu=True, sigma=1.0)
        geco_traces_mask = gaussian_filter1d(geco_traces_dff, sigma=1) > (np.median(geco_traces_dff, axis=-1) + 2*geco_traces_rsd)[:, None]
        # test
        # roi = 8
        # fig, ax = plt.subplots()
        # ax.plot(gaussian_filter1d(geco_traces_dff[roi], 1))
        # ax.axhline(np.median(geco_traces_dff[roi]) + 2*geco_traces_rsd[roi], lw=1, c='green')
        # plt.show()
        corrected_dlight = Regression_Red_From_Green_ROIs_Single_Trial_geco.run_and_save_motion_correction_results(
            dlight_traces=dlight_corr, # dlight signal to regress
            dlight_bg_traces=dlight_bg_traces, # dlight regressor
            red_traces=geco_soma_traces_corr, # geco trace regressor
            soma_indices=soma_indices, 
            event_frames=run_onset_frames,
            output_dir=out_path_regression,
            fig_out_dir=fig_out_dir,
            regression_name=regression_name,
            pre_frames=pre,
            post_frames=post,
            smoothing_sigma=0.0,
            red_trace_mask = geco_traces_mask, # mask for True frames with big calcium event to prevent biasing regression results
            num_rois_to_plot = 10 # plot first n ROIs to check regression
        )
        
        if calculate_dff:
            print('calculating dlight dFF trace...') 
            dff_dlight = percentile_dff(corrected_dlight, q=20)
            np.save(out_path_regression / 'dff_corrected_dlight.npy', dff_dlight)
            
            print('calculating soma geco dFF trace...') 
            # correct calcium trace with neuropil signal
            suite2p_geco_trace = np.load(p_suite2p_geco+r'\F.npy')
            geco_trace_corr = suite2p_geco_trace - 0.7*Fneu_geco
            dff_geco_soma = percentile_dff(geco_trace_corr, q=20) 
            np.save(out_path_regression / 'dff_geco.npy', dff_geco_soma)
                
                
        print(f"Completed: {rec}")
        elapsed_time = time.time() - start_time
        mins, secs = divmod(elapsed_time, 60)
        print(f"Total time: {int(mins)}m {secs:.1f}s")
        log_file.close()
        sys.stdout = original_stdout
        print(f"Completed: {rec} - Time: {int(mins)}m {secs:.1f}s")
    else:
        print(f"Session_existed_{rec}")
        log_file.close()
        sys.stdout = original_stdout

#%%
# OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight")
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\geco_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
# OUT_DIR_FIG = Path(r"Z:\Jingyu\Code\dlight_imgaing\dlight_GECO_Ai14_Dbh\raw_data\regression_res")
OUT_DIR_FIG = OUT_DIR_RAW_DATA/'TEST_PLOTS'/'regression_qc'
if not OUT_DIR_FIG.exists():
    OUT_DIR_FIG.mkdir(parents=True)
regression_name ='single_trial_regression_anat_roi'
#%%
exp = r'dlight_GECO_Ai14_Dbh'
f_out_df_selected = r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_corr.pkl".format(exp)
df_selected = pd.read_pickle(f_out_df_selected)
df_selected = df_selected.loc[(df_selected['speed_corr_single_trial_r_median']<0.3)
                              # &(~df_selected.index.str.contains('AC991'))
                              ]
# rec_lst = df_selected.index.tolist()
rec_lst = [
# 'AC327-20260602-02',     
# 'AC330-20260602-02',

# 'AC327-20260603-02',     
# 'AC330-20260603-02', 

# 'AC327-20260604-02',     
# 'AC330-20260604-02', 

# 'AC327-20260605-02',     
# 'AC330-20260605-02', 

# 'AC327-20260606-02',     
# 'AC330-20260606-02', 

# 'AC327-20260607-02',     
# 'AC330-20260607-02',  

# 'AC327-20260608-02',     
# 'AC330-20260608-02',

# 'AC327-20260609-02',     
# 'AC330-20260609-02',

# 'AC327-20260610-02',     
# 'AC330-20260610-02', 

'AC327-20260611-02',     
'AC330-20260611-02', 

'AC327-20260612-02',     
'AC330-20260612-02',          
    ]

error_lst = []
original_stdout = sys.stdout
# rec_lst=['AC953-20240919-02']
for rec in tqdm(rec_lst):
    print(f'\nrunning single trial regression for {rec}...')
    anm, date, ss = rec.split('-')
    # p_suite2p_geco = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\nonrigid_reg_geco\suite2p_anat_detec\plane0"
    # p_suite2p_movie = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\nonrigid_reg_geco\suite2p\plane0"
    
    p_suite2p_geco = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\anat_detect\suite2p\plane0"
    p_suite2p_movie = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\anat_detect\suite2p\plane0"

    # try:
    run_single_trial_regression_geco(p_suite2p_geco, p_suite2p_movie, regression_name,
                                     OUR_DIR_REGRESS, OUT_DIR_FIG,
                                     neuropil_dilation_iterations=3,
                                     calculate_dff=True,
                                     batch_size = 5000)
    

    # except:
    #     sys.stdout = original_stdout
    #     error_lst.append(rec)
    





