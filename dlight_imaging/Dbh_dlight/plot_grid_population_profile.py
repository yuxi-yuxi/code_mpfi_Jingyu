# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 17:06:40 2025

@author: Jingyu Cao
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from common import plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()
from common.utils_basic import normalize
#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
# OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_grid_free_dilation'
OUT_DIR_DF = OUT_DIR_RAW_DATA/'processed_dataframe_grid_free_dilation'

OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_dbh_dlight")
# OUT_DIR_FIG = Path(r"Z:\Jingyu\2026_sunposium\fig_dbh_dlight")
dlight_pre  = (-1, 0)
dlight_post = (0, 1)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5

thresh_baseline_dlight = 2
thresh_baseline_red    = 1

regression_name ='single_trial_regression'
save_plot = 1
#%% MAIN
# rec_lst = ['AC964-20250131-02', ] # for testing

# load pooled dataframe
p_pooled_df = OUT_DIR_DF / rf"df_population_profile_pooled_dilation=0_pre{dlight_pre}_post{dlight_post}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.parquet"
df_pool_all = pd.read_parquet(p_pooled_df)

df_pool_sorted = df_pool_all.loc[(df_pool_all['dlight_valid'])&(df_pool_all['red_valid'])&(~df_pool_all['edge'])]
#%% plot heatmap

# df_pool_sorted = df_pool_sorted.sort_values(by=['roi_type', 'response_amplitude'], ascending=[False, False])
df_pool_sorted = df_pool_sorted.sort_values(by=['roi_type', 'effect_size'], ascending=[False, False])

# get the 'dlight_mean_trace' column in that sorted order
traces = np.stack(df_pool_sorted['mean_profile'])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
          aspect='auto', interpolation='none',
          extent=[-2, 4, 0, traces.shape[0]],
          # cmap='YlGnBu_r',
          cmap='Greys')
ax.set(xlim=(-1, 4))
roi_types = df_pool_sorted['roi_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
# for idx in change_idx:
#     ax.axhline(idx, color='red', lw=0.8, ls='--')
    
n = traces.shape[0]
change_idx_plot = n - change_idx
for y in change_idx_plot:
    ax.axhline(y, color='red', lw=0.8, ls='--')


save_fig(fig, OUT_DIR_FIG, r'dlight_pupulation_heatmap_greys_pre={}_post={}_ES={}_amp={}.pdf'
            .format(dlight_pre, dlight_post, effect_size_thresh, amp_shuff_thresh_up), save=save_plot)

#%% plot population mean trace
dlightUp_traces_dlight = 100*np.stack(df_pool_sorted.loc[df_pool_sorted['Up'], 'mean_profile'])
# dlightUp_traces_red = 100*np.stack(df_pool_sorted.loc[df_pool_sorted['Up'], 'mean_profile_red'])
dlightUp_traces_red = None
non_dlightUp_traces = 100*np.stack(df_pool_sorted.loc[~df_pool_sorted['Up'], 'mean_profile'])

bef, aft = 2, 4
xaxis = np.arange(30*(bef+aft))/30-bef    
fig, ax = plt.subplots(dpi=300, figsize=(2,2))
pf.plot_two_traces_with_scalebars(dlightUp_traces_dlight, non_dlightUp_traces, xaxis, ax,
                                  colors = ("tab:green", "grey"),
                                  timebar=0.5, dffbar=1, 
                                  show_xaxis=1, xlabel='time from run (s)')
ax.set(xlim=(-1, 4))

save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_pre={}_post={}_ES={}_amp={}.pdf'
            .format(dlight_pre, dlight_post, effect_size_thresh, amp_shuff_thresh_up), save=save_plot)


#%% plot by session / anm
# bef, aft = 2, 4
# xaxis = np.arange(30*(bef+aft))/30-bef   
# save_plot=0 
# for rec_id, df_session in df_pool_sorted.groupby('anm'):
    
#     # heatmap
#     df_session = df_session.sort_values(by=['roi_type', 'effect_size'], ascending=[False, False])
#     traces = np.stack(df_session['mean_profile'])
#     traces = gaussian_filter1d(traces, sigma=1)
#     traces = normalize(traces)
#     fig, ax = plt.subplots(figsize=(3,3), dpi=300)
#     ax.imshow(traces,
#               aspect='auto', interpolation='none',
#               extent=[-2, 4, 0, traces.shape[0]],
#               # cmap='YlGnBu_r',
#               cmap='Greys')
#     ax.set(xlim=(-1, 4))
#     roi_types = df_session['roi_type'].values
#     change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
#     # for idx in change_idx:
#     #     ax.axhline(idx, color='red', lw=0.8, ls='--')
#     n = traces.shape[0]
#     change_idx_plot = n - change_idx
#     for y in change_idx_plot:
#         ax.axhline(y, color='red', lw=0.8, ls='--')
#     ax.set(title=rec_id)
#     save_fig(fig, OUT_DIR_FIG, r'dlight_pupulation_heatmap_greys_pre={}_post={}_ES={}_amp={}.pdf'
#                 .format(dlight_pre, dlight_post, effect_size_thresh, amp_shuff_thresh_up), save=save_plot)
    
#     # mean trace
#     dlightUp_traces_dlight = 100*np.stack(df_session.loc[df_session['Up'], 'mean_profile'])
#     # dlightUp_traces_red = 100*np.stack(df_session.loc[df_session['Up'], 'mean_profile_red'])
#     dlightUp_traces_red = None
#     fig, ax = plt.subplots(dpi=300, figsize=(2,2))
#     pf.plot_two_traces_with_scalebars(dlightUp_traces_dlight, dlightUp_traces_red, xaxis, ax,
#                                       colors = ("tab:green", "tab:red"),
#                                       timebar=0.5, dffbar=1, 
#                                       show_xaxis=1, xlabel='time from run (s)')
#     ax.set(title=rec_id)
#     save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_pre={}_post={}_ES={}_amp={}.pdf'
#                 .format(dlight_pre, dlight_post, effect_size_thresh, amp_shuff_thresh_up), save=save_plot)
    
    


       
