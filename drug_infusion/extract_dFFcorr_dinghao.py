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
    'Z:/Dinghao/code_mpfi_dinghao/utils'
    ]
sys.path.extend(directories)

from rec_lst_infusion import rec_lst
from imaging_pipeline_functions import calculate_dFF_percentile

from common_functions import get_GPU_availability
_, GPU_AVAILABLE, _ = get_GPU_availability()


#%% paths and parameters
image_stem    = Path('Z:/Jingyu/2P_Recording')
raw_data_stem = Path('Z:/Jingyu/LC_HPC_manuscript/raw_data/drug_infusion')
raw_sig_stem  = raw_data_stem / 'raw_signals'

# info_path = raw_data_stem / 'infusion_session_info.parquet'
# info = pd.read_parquet(info_path)
info = rec_lst
info = info[30:]

# correction index
corr_index = 0.7  # ROI - corr_index * neuropil


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
        
        # dFF calculation (hopefully on GPU)
        dFF, baselines = calculate_dFF_percentile(
            F_corr,
            t_axis=1,
            window_size=9000,
            pct=20,
            GPU_AVAILABLE=GPU_AVAILABLE,
            CHUNK=True, 
            chunk_size=5000, 
            return_baseline=True
            )
        
        # check number of frames matching 
        frametimes = beh['frame_times']
        print(f'length of frame_times = {len(frametimes)}')
        print(f'ops-nframes           = {nframes}')
        print(f'length of dFF         = {dFF.shape[1]}')
        if dFF.shape[1] != ops['nframes']:
            raise ValueError('Issues with dFF calculation')
        
        # save dFF and baselines to disk
        save_path_dFF       = raw_sig_stem / recdate / f'{recname}_dFF.npy'
        save_path_baselines = raw_sig_stem / recdate / f'{recname}_baselines.npy'
        np.save(save_path_dFF, dFF)
        np.save(save_path_baselines, baselines)
        
        # plot baselines for inspection
        figure_stem = raw_data_stem / 'baseline_plots'
        
        mean_baselines = np.nanmean(baselines, axis=-1)

        pct_lt_1 = 100 * np.mean(mean_baselines < 1)
        pct_lt_0 = 100 * np.mean(mean_baselines < 0)
        
        # 2-panel plot
        fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.6), dpi=200)
        
        # left: all values
        ax[0].hist(
            mean_baselines,
            bins=100,
            color='grey',
            alpha=0.8
        )
        ax[0].set(
            xlabel='Mean baseline',
            ylabel='# ROI',
            title='all ROIs'
        )
        
        # right: zoomed
        ax[1].hist(
            mean_baselines,
            bins=np.linspace(-1, 1, 120),  # smaller bin size
            color='grey',
            alpha=0.8
        )
        ax[1].set(
            xlim=(-1, 1),
            xlabel='Mean baseline',
            title='zoom: -1 to 1'
        )
        
        # text stats
        ax[1].text(
            0.98, 0.95,
            f'< 1: {pct_lt_1:.1f}%\n< 0: {pct_lt_0:.1f}%',
            transform=ax[1].transAxes,
            ha='right',
            va='top',
            fontsize=8
        )
        
        for a in ax:
            for s in ['top', 'right']:
                a.spines[s].set_visible(False)
        
        fig.tight_layout()
        fig.savefig(
            figure_stem / f'{recname}_roi_dff_baseline_mean_hist_2panel.png'
        )


# #%%
# F_aligned = np.zeros((F_corr.shape[0], tot_trials, 180))
# for roi in range(F_corr.shape[0]):
#     F_aligned[roi,:,:] = _align_trials(F_corr[roi], beh_ss1, bef=2, aft=4, gpu=0, fs=30)
    
# F_aligned_mean = np.zeros((F_corr.shape[0], 180))
# F_keys = np.zeros(F_corr.shape[0])
# for roi in range(F_corr.shape[0]):
#     F_aligned_mean[roi] = normalise(np.nanmean(F_aligned[roi,:,:], axis=0))
#     F_keys[roi] = np.nanmean(F_aligned_mean[roi][75:90]) / np.nanmean(F_aligned_mean[roi][45:60]) 
    
    
# #%% plotting 
# # sort by F_keys
# sort_idx = np.argsort(F_keys)[::-1]

# F_aligned_mean_sorted = F_aligned_mean[sort_idx, :]
# F_keys_sorted = F_keys[sort_idx]

# # plotting
# fig, ax = plt.subplots(figsize=(3,3))
# ax.imshow(F_aligned_mean_sorted, 
#           cmap='Greys', aspect='auto', extent=[-2, 4, 0, len(F_keys)])

# n_roi = len(F_keys_sorted)
# n_pyrup   = np.sum(F_keys_sorted > 3/2)
# n_pyrdown = np.sum(F_keys_sorted < 2/3)

# xmin, xmax = -2, 4

# y_pyrup = n_roi - n_pyrup
# ax.hlines(
#     y_pyrup,
#     xmin, xmax,
#     colors='firebrick',
#     lw=2,
#     zorder=5
# )

# y_pyrdown = n_pyrdown
# ax.hlines(
#     y_pyrdown,
#     xmin, xmax,
#     colors='purple',
#     lw=2,
#     zorder=5
# )

# # percentages
# pct_pyrup   = 100 * n_pyrup   / n_roi
# pct_pyrdown = 100 * n_pyrdown / n_roi

# x_text = xmax + 0.1

# ax.text(
#     x_text,
#     n_roi - n_pyrup / 2,
#     f'PyrUp:\n{pct_pyrup:.1f}%',
#     color='firebrick',
#     va='center',
#     ha='left',
#     fontsize=8,
#     clip_on=False
# )

# ax.text(
#     x_text,
#     n_pyrdown / 2,
#     f'PyrDown:\n{pct_pyrdown:.1f}%',
#     color='purple',
#     va='center',
#     ha='left',
#     fontsize=8,
#     clip_on=False
# )

# ax.set(xlabel='Time from run onset (s)',
#        ylabel='# ROI',
#        title=f'{rec_id}-{sess_id}')