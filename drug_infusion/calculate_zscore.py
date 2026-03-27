# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 15:56:19 2026
Used to test saline sessions 31 Jan 2026

@author: Jingyu Cao
@modifier: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# add parent directories to path
directories = [
    'Z:/Jingyu/code_mpfi_Jingyu/common', 
    'Z:/Jingyu/code_mpfi_Jingyu/drug_infusion',
    # 'Z:/Dinghao/code_mpfi_dinghao/utils'
    ]
sys.path.extend(directories)

from rec_lst_infusion import rec_lst
from common.event_response_quantification import calculate_zscore_f_trace
from common.trial_selection import seperate_valid_trial


#%% paths and parameters
image_stem    = Path('Z:/Jingyu/2P_Recording')
raw_data_stem = Path('Z:/Jingyu/LC_HPC_manuscript/raw_data/drug_infusion')
raw_sig_stem  = raw_data_stem / 'raw_signals'

# info_path = raw_data_stem / 'infusion_session_info.parquet'
# info = pd.read_parquet(info_path)
info = rec_lst
# info = info[30:]

# correction index
corr_index = 0.7  # ROI - corr_index * neuropil

pre_window=(-1, 0)
post_window=(0.5, 1.5)
#%% main loop
for recdate, rec in info.iterrows():
    # load keys first  
    animal   = rec['anm']
    sessions = rec['session']
    labels   = rec['label']
    
    # ----------------
    # now the goal is to calculate corrected dFF, separate dFF traces into 
    # ... sessions, and save them to dFF_stem
    # example save file name: AC986-20250606-02.npy
    # ----------------
    
    for sess_idx, sess in enumerate(sessions):
        recname = f'{recdate}-{sess}'
        print(f'\nProcessing {recname}...')
        
        # paths 
        signal_path  = raw_data_stem / 'raw_signals' / recdate
        beh_path     = raw_data_stem / 'behaviour_profile' / f'{recdate}-{sess}.pkl'
        suite2p_path = image_stem / animal / recdate / sess / 'suite2p_func_detec' / 'plane0'
        
        # Suite2p ops 
        ops = np.load(suite2p_path / 'ops.npy', allow_pickle=True).item()
        
        # behaviour stuff 
        beh = pd.read_pickle(beh_path)
        run_onset_frames = np.array(beh['run_onset_frames'])
        valid_trials = (seperate_valid_trial(beh))&(run_onset_frames!=-1)
        run_onset_frames_valid = run_onset_frames[valid_trials]
        
        run_onsets = beh['run_onsets']
        tot_trials = len(run_onsets)

        # signal - all ROIs
        sig_master = xr.open_dataarray(signal_path / 'sig_master_raw.nc')
        F_all      = sig_master.values.squeeze()
        
        # signal - all neuropil
        sig_master_neu = xr.open_dataarray(signal_path / 'sig_master_neu_raw.nc')
        Fneu_all       = sig_master_neu.values.squeeze()
        
        # correction
        nframes = ops['nframes']
        if sess_idx == 0:
            F_start = 0
            F_end   = nframes
        else:
            F_start = F_end
            F_end   = F_start + nframes
        
        F_corr = F_all - corr_index * Fneu_all
        F_corr = F_corr[:, F_start:F_end]
        
        # zscore calculation on GPU
        # zscored_geco, pooled_std_geco = calculate_zscore_f_trace(F_corr, event_frames=run_onset_frames_valid,
        #                             baseline_window=pre_window, response_window=post_window,
        #                             pre_event_window=2, post_event_window=4,
        #                             imaging_rate=30.0)
        zscored_trace, pooled_std = calculate_zscore_f_trace(
            F_corr,
            event_frames=run_onset_frames_valid,
            baseline_window=pre_window,
            response_window=post_window,
            pre_event_window=2,
            post_event_window=4,
            imaging_rate=30.0,
            debug_plot=0,
            debug_roi_idx=0
        )
        import os
        try:
            os.remove(raw_sig_stem / recdate /'{recname}_zscored.npy')
        except:
            print(f'{recname}')
        np.save(raw_sig_stem / recdate / f'{recname}_zscored.npy', zscored_trace)
        np.save(raw_sig_stem / recdate / f'{recname}_baseline{pre_window}_std.npy', pooled_std)
        
        