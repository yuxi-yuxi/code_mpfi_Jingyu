# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 17:06:40 2025

@author: Jingyu Cao
"""
#%% imports and funcs
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from common import plotting_functions_Jingyu as pf
pf.mpl_formatting()
save_fig = pf.save_fig
from common.utils_basic import normalize

# HELP FUNCS
def division_helper(a, b):
    if b!=0:
        res = a/b
    else:
        res = np.nan
    return res

def classify_pyrs(dlight_stats, 
                  amp_shuff_thresh_up,
                  amp_shuff_thresh_down,
                  effect_size_thresh,
                  pyrUp_thresh,
                  pyrDown_thresh,
                  mean_thresh_dlight=1.5,
                  mean_thresh_geco=1,
                  geco_ratio = 'geco_ratio',
                  ):
    df_pool_sorted = dlight_stats.copy() # withou modifying the original pooled data
    
    df_pool_sorted['shuffle_amps_thresh_up']   = df_pool_sorted['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_up))
    df_pool_sorted['shuffle_amps_thresh_down'] = df_pool_sorted['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_down))
    
    # df_pool_sorted['dlight_valid'] = df_pool_sorted['mean_profile'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
    # df_pool_sorted['geco_valid'] = df_pool_sorted['mean_profile_geco'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
    if mean_thresh_dlight is not None:
        df_pool_sorted['dlight_valid'] = df_pool_sorted['mean_dlight'].apply(lambda x: 0<x<mean_thresh_dlight)
    else:
        # df_pool_sorted['dlight_valid'] = df_pool_sorted['baseline_dlight_min'].apply(lambda x: 3<x)
        df_pool_sorted['dlight_valid'] = True
    if mean_thresh_geco is not None:  
        df_pool_sorted['geco_valid'] = df_pool_sorted['mean_geco'].apply(lambda x: 0<x<mean_thresh_geco)     
    else:
        df_pool_sorted['geco_valid'] = True
        # df_pool_sorted['geco_valid'] = df_pool_sorted['baseline_geco_min'].apply(lambda x: 3<x)
        
    df_pool_sorted['valid'] = (df_pool_sorted['dlight_valid'])&(df_pool_sorted['geco_valid'])
    
    # df_pool_sorted = df_pool_sorted.loc[(df_pool_sorted['dlight_valid'])&(df_pool_sorted['geco_valid'])]
    # df_pool_sorted = df_pool_sorted.loc[(df_pool_sorted['dlight_valid'])]
    
    df_pool_sorted['dlightUp'] = np.where(
                                (df_pool_sorted['response_amplitude']>df_pool_sorted['shuffle_amps_thresh_up'])&
                                (df_pool_sorted['effect_size']>effect_size_thresh)&
                                (df_pool_sorted['valid']),
                                True, False)
    df_pool_sorted['dlightDown'] = np.where(
                                (df_pool_sorted['response_amplitude']<df_pool_sorted['shuffle_amps_thresh_down'])&
                                (df_pool_sorted['effect_size']< -effect_size_thresh)&
                                (df_pool_sorted['valid']),
                                True, False)
    df_pool_sorted['dlightStable'] = (~df_pool_sorted['dlightUp'])&(~df_pool_sorted['dlightDown'])&(df_pool_sorted['valid'])
    
    df_pool_sorted['pyrUp'] = np.where(
                                (df_pool_sorted[geco_ratio]> pyrUp_thresh)
                                &(df_pool_sorted['valid']),
                                True, False)
    df_pool_sorted['pyrDown'] = np.where(
                                (df_pool_sorted[geco_ratio]<pyrDown_thresh)
                                &(df_pool_sorted['valid']),
                                True, False)
    df_pool_sorted['pyrStable'] = (~df_pool_sorted['pyrUp'])&(~df_pool_sorted['pyrDown'])&(df_pool_sorted['valid'])
    
    
    try:
        df_pool_sorted.loc[df_pool_sorted['dlightUp'],     'dlight_type'] = 'Up'
        df_pool_sorted.loc[df_pool_sorted['dlightDown'],   'dlight_type'] = 'Down'
        df_pool_sorted.loc[df_pool_sorted['dlightStable'], 'dlight_type'] = 'Stable'
        
        df_pool_sorted.loc[df_pool_sorted['pyrUp'],     'geco_type'] = 'Up'
        df_pool_sorted.loc[df_pool_sorted['pyrDown'],   'geco_type'] = 'Down'
        df_pool_sorted.loc[df_pool_sorted['pyrStable'], 'geco_type'] = 'Stable'

    except:
        print('error')
    
    
    return df_pool_sorted



def calculate_percs(roi_stats):
    
    # roi_stats = roi_stats.set_index('roi_id', drop=False)
    
    dic_stats_session = {
    'perc_pyrUp_dlightUp':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightUp'] & roi_stats['pyrUp']]),
            len(roi_stats.loc[roi_stats['dlightUp']])
        ),

    'perc_pyrUp_dlightStable':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightStable'] & roi_stats['pyrUp']]),
            len(roi_stats.loc[roi_stats['dlightStable']])
        ),

    'perc_pyrUp_dlightDown':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightDown'] & roi_stats['pyrUp']]),
            len(roi_stats.loc[roi_stats['dlightDown']])
        ),

    'perc_pyrUp_no_dlightUp':
        division_helper(
            len(roi_stats.loc[(~roi_stats['dlightUp']) & roi_stats['pyrUp']]),
            len(roi_stats.loc[(~roi_stats['dlightUp'])])
        ),

    'perc_pyrUp_all':
        division_helper(
            len(roi_stats.loc[roi_stats['pyrUp']]),
            len(roi_stats)
        ),

    'perc_pyrDown_dlightUp':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightUp'] & roi_stats['pyrDown']]),
            len(roi_stats.loc[roi_stats['dlightUp']])
        ),

    'perc_pyrDown_dlightStable':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightStable'] & roi_stats['pyrDown']]),
            len(roi_stats.loc[roi_stats['dlightStable']])
        ),

    'perc_pyrDown_dlightDown':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightDown'] & roi_stats['pyrDown']]),
            len(roi_stats.loc[roi_stats['dlightDown']])
        ),

    'perc_pyrDown_no_dlightUp':
        division_helper(
            len(roi_stats.loc[(~roi_stats['dlightUp']) & roi_stats['pyrDown']]),
            len(roi_stats.loc[(~roi_stats['dlightUp'])])
        ),

    'perc_pyrDown_all':
        division_helper(
            len(roi_stats.loc[roi_stats['pyrDown']]),
            len(roi_stats)
        ),

    'perc_dlightUp_all':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightUp']]),
            len(roi_stats)
        ),

    'perc_dlightDown_all':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightDown']]),
            len(roi_stats)
        ),
        
    'n_valid_rois': len(roi_stats)  
        }
      
    
    return dic_stats_session

#%% PATHS AND PARAMS
# run-onset response window
dlight_pre  = (-1, 0)
dlight_post = (0, 1)
geco_pre  = (-1, 0)
geco_post = (0.5, 1.5)
bef, afte = 2, 4

time_windows=[(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4)] # time windows used to quantify dFF differecne 

effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
pyrUp_thresh = 1.12
pyrDown_thresh = 1/pyrUp_thresh
regression_name = 'single_trial_regression_anat_roi'

OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
# OUT_DIR_FIG = (OUT_DIR_RAW_DATA/'TEST_PLOTS'/'window_test'/'session_pooled'/
#                 f'dlight_pre{dlight_pre}_post{dlight_post}_geco_pre{geco_pre}_post{geco_post}_manual_selec')
# OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight\TEST_PLOTS\session_selection_test\non_up_amp_2.1")
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_GECO_dlight")
if not OUT_DIR_FIG.exists():
    OUT_DIR_FIG.mkdir(parents=True)
save_plot=0


# Load recording list
# from dlight_imaging.geco_dlight.recording_list import rec_lst_dlight_geco as rec_lst
# from recording_list import rec_lst_dlight_geco as rec_lst
#%% data pooling
p_pooled_df = OUT_DIR_RAW_DATA / 'processed_dataframe'/ rf"df_population_profile_pooled_pre{dlight_pre}_post{dlight_post}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.parquet"
df_pooled_profile = pd.read_parquet(p_pooled_df)

# df_pool_sorted = df_pooled_profile.loc[df_pooled_profile['corr_non_vs_DA-Up']<0.85]
df_pool_sorted = df_pooled_profile.loc[df_pooled_profile['non_up_amp_bef']<2.1]
df_pool_sorted = df_pool_sorted.loc[df_pool_sorted['baseline_valid']]

#%% selecr sesssions for plot
# add = ['AC991-20250718-02', 'AC991-20250721-04', 'AC991-20250728-04', 'AC991-20250729-04',]
# rec_lst_a = ['AC953-20240919-02', 'AC953-20240920-02', 'AC953-20240924-02', 'AC953-20240925-02', 'AC953-20240927-02', 'AC953-20241008-04', 'AC991-20250714-04', 'AC991-20250718-04',     'AC991-20250725-02', 'AC991-20250729-02', 'AC991-20250730-02', 'AC991-20250730-04', 'AC991-20250801-02', 'AC991-20250801-04', 'AC992-20250720-02', 'AC992-20250722-02', 'AC992-20250725-04', 'AC992-20250729-02', 'AC992-20250729-04', 'AC992-20250730-02', 'AC992-20250730-04', 'AC992-20250801-04', 'AC304-20250828-02', 'AC304-20250902-04', 'AC304-20250903-02', 'AC304-20250904-02', 'AC304-20250930-02', 'AC304-20250930-04', 'AC305-20250902-02', 'AC305-20250930-02', 'AC305-20250930-04', 'AC305-20251001-02', 'AC305-20251001-04']
# rec_lst_tmp = rec_lst_a + ['AC991-20250728-04', ] 

# df_pool_sorted = df_pooled_profile.loc[df_pooled_profile['rec_id'].isin(rec_lst_tmp)]
# df_pool_sorted = df_pool_sorted.loc[df_pool_sorted['baseline_valid']]

#%% reassign Up and Down using chosen thresholding
df_pool_dlight_up = df_pool_sorted.loc[df_pool_sorted['dlightUp']]
df_pool_non_dlight_up = df_pool_sorted.loc[~df_pool_sorted['dlightUp']]

#%%
#% plot heatmap
key_geco_ratio = 'geco_zscore_amp'
key_geco_profile = 'mean_profile_geco_zscore'
# all soma rois dlight
df_pool_sorted = df_pool_sorted.sort_values(by=['dlight_type', 'effect_size'], ascending=[False, False])
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
ax.set(xlim=(-1, 4), title='dLight_mean_profile sort by dLight')
roi_types = df_pool_sorted['dlight_type'].values
# change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
# for idx in change_idx:
#     ax.axhline(idx, color='red', lw=0.8, ls='--')  # adjust style as you like
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
n = traces.shape[0]
change_idx_plot = n - change_idx
for y in change_idx_plot:
    ax.axhline(y, color='red', lw=0.8, ls='--')  
save_fig(fig, OUT_DIR_FIG, r'all_dlight_population_heatmap_greys_ES={}_amp={}'
        .format(effect_size_thresh, amp_shuff_thresh_up), 
        save=save_plot)

# all soma rois GECO
df_pool_sorted = df_pool_sorted.sort_values(by=['geco_type', key_geco_ratio], ascending=[False, False])
# get the 'geco_mean_trace' column in that sorted order
traces = np.stack(df_pool_sorted[key_geco_profile])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
      aspect='auto', interpolation='none',
      extent=[-2, 4, 0, traces.shape[0]],
      # cmap='YlGnBu_r',
      cmap='Greys')
ax.set(xlim=(-1, 4), title=f'{key_geco_profile} sort by geco')
roi_types = df_pool_sorted['geco_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
n = traces.shape[0]
change_idx_plot = n - change_idx
for y in change_idx_plot:
    ax.axhline(y, color='red', lw=0.8, ls='--')    
    
save_fig(fig, OUT_DIR_FIG, r'all_GECO_population_heatmap_greys_pyrUp_thresh={}'
        .format(pyrUp_thresh), 
        save=save_plot)

# DA-up rois GECO
df_sorted = df_pool_dlight_up.sort_values(by=['geco_type', key_geco_ratio], ascending=[False, False])
# get the 'geco_mean_trace' column in that sorted order
traces = np.stack(df_sorted[key_geco_profile])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
      aspect='auto', interpolation='none',
      extent=[-2, 4, 0, traces.shape[0]],
      # cmap='YlGnBu_r',
      cmap='Greys')
ax.set(xlim=(-1, 4), title=f'{key_geco_profile} sort by geco')
roi_types = df_sorted['geco_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
n = traces.shape[0]
change_idx_plot = n - change_idx
for y in change_idx_plot:
    ax.axhline(y, color='red', lw=0.8, ls='--')  
save_fig(fig, OUT_DIR_FIG, r'DA_up_GECO_pupulation_heatmap_greys_pyrUp_thresh={}'
        .format(pyrUp_thresh), 
        save=save_plot)

# non-DA-up rois GECO
df_sorted = df_pool_non_dlight_up.sort_values(by=['geco_type', key_geco_ratio], ascending=[False, False])
# get the 'geco_mean_trace' column in that sorted order
traces = np.stack(df_sorted[key_geco_profile])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
      aspect='auto', interpolation='none',
      extent=[-2, 4, 0, traces.shape[0]],
      # cmap='YlGnBu_r',
      cmap='Greys')
ax.set(xlim=(-1, 4), title=f'{key_geco_profile} sort by geco')
roi_types = df_sorted['geco_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
n = traces.shape[0]
change_idx_plot = n - change_idx
for y in change_idx_plot:
    ax.axhline(y, color='red', lw=0.8, ls='--')  
save_fig(fig, OUT_DIR_FIG, r'non_DA_up_GECO_pupulation_heatmap_greys_pyrUp_thresh={}'
        .format(pyrUp_thresh), 
        save=save_plot)
    
# DA-up rois dlight
df_sorted = df_pool_dlight_up.sort_values(by=['geco_type', key_geco_ratio], ascending=[False, False])
# get the 'geco_mean_trace' column in that sorted order
traces = np.stack(df_sorted['mean_profile'])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
      aspect='auto', interpolation='none',
      extent=[-2, 4, 0, traces.shape[0]],
      # cmap='YlGnBu_r',
      cmap='Greys')
ax.set(xlim=(-1, 4), title='dlight_mean_profile sort by geco')
roi_types = df_sorted['geco_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
n = traces.shape[0]
change_idx_plot = n - change_idx
for y in change_idx_plot:
    ax.axhline(y, color='red', lw=0.8, ls='--')  
save_fig(fig, OUT_DIR_FIG, r'DA_up_dlight_pupulation_heatmap_greys_pyrUp_thresh={}'
        .format(pyrUp_thresh), 
        save=save_plot)

# non-DA-up rois dlight
df_sorted = df_pool_non_dlight_up.sort_values(by=['geco_type', key_geco_ratio], ascending=[False, False])
# get the 'geco_mean_trace' column in that sorted order
traces = np.stack(df_sorted['mean_profile'])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
      aspect='auto', interpolation='none',
      extent=[-2, 4, 0, traces.shape[0]],
      # cmap='YlGnBu_r',
      cmap='Greys')
ax.set(xlim=(-1, 4), title='dlight_mean_profile sort by geco')
roi_types = df_sorted['geco_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
n = traces.shape[0]
change_idx_plot = n - change_idx
for y in change_idx_plot:
    ax.axhline(y, color='red', lw=0.8, ls='--')  
save_fig(fig, OUT_DIR_FIG, r'non_DA_up_dlight_pupulation_heatmap_greys_pyrUp_thresh={}'
        .format(pyrUp_thresh), 
        save=save_plot)
#% plot population mean trace

# OUT_DIR_FIG = (OUT_DIR_RAW_DATA/'TEST_PLOTS'/f'geco_pre{geco_pre}_geco_post{geco_post}_zscore')

# if not OUT_DIR_FIG.exists():
#     OUT_DIR_FIG.mkdir(parents=True)
# save_plot=0

key_geco_profile = 'mean_profile_geco_zscore'

if key_geco_profile == 'mean_profile_geco_zscore':
    factor = 1
    dffbar = 0.05
    dff_label = 'zscore F'
elif key_geco_profile == 'mean_profile_geco':
    factor = 100
    dffbar = 2
    dff_label = '%dF/F'

dlightUp_pyrUp_geco_traces         = factor*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                                  (df_pool_sorted['geco_type']=='Up'), key_geco_profile])
dlightUp_pyrDown_geco_traces       = factor*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                                  (df_pool_sorted['geco_type']=='Down'), key_geco_profile])


non_dlightUp_pyrUp_geco_traces     = factor*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']!='Up')&
                                                      (df_pool_sorted['geco_type']=='Up'), key_geco_profile])

non_dlightUp_pyrDown_geco_traces   = factor*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']!='Up')&
                                                      (df_pool_sorted['geco_type']=='Down'), key_geco_profile])

factor=100
dlightUp_pyrUp_dlight_traces       = factor*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                                     (df_pool_sorted['geco_type']=='Up'), 'mean_profile'])

dlightUp_pyrDown_dlight_traces     = factor*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                                     (df_pool_sorted['geco_type']=='Down'), 'mean_profile'])
non_dlightUp_pyrUp_dlight_traces   = factor*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']!='Up')&
                                                         (df_pool_sorted['geco_type']=='Up'), 'mean_profile'])
non_dlightUp_pyrDown_dlight_traces = factor*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']!='Up')&
                                                         (df_pool_sorted['geco_type']=='Down'), 'mean_profile'])
    
dlightUp_dlight_traces      = factor*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up'), 'mean_profile'])
non_dlightUp_dlight_traces  = factor*np.stack(df_pool_sorted.loc[~(df_pool_sorted['dlight_type']=='Up'), 'mean_profile'])



bef, aft = 2, 4
xaxis = np.arange(30*(bef+aft))/30-bef    

# plot dlight DA-Up vs non DA-Up
fig, ax = plt.subplots(dpi=300, figsize=(2,2))
pf.plot_two_traces_with_scalebars_fixbar(dlightUp_dlight_traces , non_dlightUp_dlight_traces,
                                     xaxis, ax,
                                     colors = ("tab:green", "grey"),
                                     timebar=0.5, dffbar=1, 
                                     show_xaxis=1, xlabel='time from run (s)',
                                     scale_by_dffbar=True, 
                                     dffbar_plot_height=0.5,
                                     bar_y_bump_frac=0.05,
                                     dff_label = '%dF/F')

ax.set(xlim=(-1, 4),)
save_fig(fig, OUT_DIR_FIG, r'dlight_mean_trace_ES={}_amp={}'
        .format(effect_size_thresh, amp_shuff_thresh_up), 
        save=save_plot)

# plot pyrUp GECO DA-Up vs non DA-Up
fig, ax = plt.subplots(dpi=300, figsize=(2,2))
pf.plot_two_traces_with_scalebars_fixbar(dlightUp_pyrUp_geco_traces, non_dlightUp_pyrUp_geco_traces ,
                                     xaxis, ax,
                                     colors = ('brown', 'indianred'),
                                     timebar=0.5, dffbar=dffbar, 
                                     show_xaxis=1, xlabel='time from run (s)',
                                     scale_by_dffbar=True,
                                     dffbar_plot_height=0.5,
                                     dff_label = dff_label,
                                     bar_y_bump_frac=0.05)

ax.set(xlim=(-1, 4))
save_fig(fig, OUT_DIR_FIG, r'geco_pyrUp_mean_trace_non-dlightUp_ES={}_amp={}'
        .format(effect_size_thresh, amp_shuff_thresh_up), 
        save=save_plot)


# plot pyrDown GECO DA-Up vs non DA-Up
fig, ax = plt.subplots(dpi=300, figsize=(2,2))
pf.plot_two_traces_with_scalebars_fixbar(dlightUp_pyrDown_geco_traces, non_dlightUp_pyrDown_geco_traces ,
                                     xaxis, ax,
                                     colors = ('indigo', 'blueviolet'),
                                     timebar=0.5, dffbar=dffbar, 
                                     show_xaxis=1, xlabel='time from run (s)',
                                     scale_by_dffbar=True,
                                     dff_label = dff_label,
                                     dffbar_plot_height=0.5,
                                     bar_y_bump_frac=0.05)

ax.set(xlim=(-1, 4))
save_fig(fig, OUT_DIR_FIG, r'geco_pyrDown_mean_trace_non-dlightUp_ES={}_amp={}'
        .format(effect_size_thresh, amp_shuff_thresh_up), 
        save=save_plot)


# plot DA-up vs non-DA-up GECO pyrUp
fig, ax = plt.subplots(dpi=300, figsize=(2.5,2.5))
pf.plot_two_traces_with_binned_stats(dlightUp_pyrUp_geco_traces, non_dlightUp_pyrUp_geco_traces,
                                 bef=2, aft=4,
                                 ax=ax, 
                                 # baseline_window=geco_pre,
                                 baseline_window=None,
                                 time_windows=time_windows,
                                 colors = ['brown', 'indianred'],
                                 labels = ['dLight_Up_pyrUp', 'non_dLight_Up_pyrUp'],
                                 show_scalebar=0
                                 )
ax.set(xlim=(-1, 4))
save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_dlightUp-pyrUp_ES={}_amp={}'
        .format(effect_size_thresh, amp_shuff_thresh_up), 
        save=save_plot)

# plot DA-up vs non-DA-up GECO pyrDown 
fig, ax = plt.subplots(dpi=300, figsize=(2.5,2.5))
pf.plot_two_traces_with_binned_stats(dlightUp_pyrDown_geco_traces, non_dlightUp_pyrDown_geco_traces,
                                 bef=2, aft=4,
                                 ax=ax, 
                                 # baseline_window=geco_pre,
                                 baseline_window=None,
                                 time_windows=time_windows,
                                 colors = ['indigo', 'blueviolet'],
                                 labels = ['dLight_Up_pyrDown', 'non_dLight_Up_pyrDown'],
                                 show_scalebar=0
                                 )
ax.set(xlim=(-1, 4))
save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_dlightUp-pyrDown_ES={}_amp={}'
        .format(effect_size_thresh, amp_shuff_thresh_up), 
        save=save_plot)


# plot all DA-Up and non-DA-Up dLight, GECO (same y_axis)
fig, ax = plt.subplots(dpi=300, figsize=(3, 3))
pf.plot_mean_trace(dlightUp_dlight_traces, ax, xaxis, color='green')
pf.plot_mean_trace(non_dlightUp_dlight_traces, ax, xaxis, color='grey')
ax.set(xlim=(-1, 4), ylabel='%dF/F')
save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_sharedY_dlight_ES={}_amp={}'
        .format(effect_size_thresh, amp_shuff_thresh_up), 
        save=save_plot)

fig, ax = plt.subplots(dpi=300, figsize=(2, 1.5))
pf.plot_mean_trace(dlightUp_pyrUp_geco_traces, ax, xaxis, color='darkred', label='DA-Up')
pf.plot_mean_trace(non_dlightUp_pyrUp_geco_traces, ax, xaxis, color='tab:red', label='non-DA-Up')
ax.set(xlim=(-1, 4), ylabel=dff_label)
ax.legend(frameon=False, prop={'size': 6})
save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_sharedY_pyrUp_ES={}_amp={}'
        .format(effect_size_thresh, amp_shuff_thresh_up), 
        save=save_plot)

fig, ax = plt.subplots(dpi=300, figsize=(2, 1.5))
pf.plot_mean_trace(dlightUp_pyrDown_geco_traces, ax, xaxis, color='indigo', label='DA-Up')
pf.plot_mean_trace(non_dlightUp_pyrDown_geco_traces, ax, xaxis, color='blueviolet', label='non-DA-Up')
ax.set(xlim=(-1, 4), ylabel=dff_label)
ax.legend(frameon=False, prop={'size': 6})
save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_sharedY_pyrDown_ES={}_amp={}'
        .format(effect_size_thresh, amp_shuff_thresh_up), 
        save=save_plot)
#% plot %pyrUp and %pyrDown

df_perc = calculate_percs(df_pool_sorted)

# Calculate percentages per recording
df_perc_pool = df_pool_sorted.groupby('rec_id').apply(calculate_percs).apply(pd.Series)
df_perc_pool.index.name = 'rec_id'

# plot non-DA_up vs DA_up
# pyrUp
a = df_perc_pool['perc_pyrUp_no_dlightUp']
b = df_perc_pool['perc_pyrUp_dlightUp']
fig, ax = plt.subplots(dpi=300, figsize=(2,3))
pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
                          # ylim=(0, 85), 
                         ylabel='% PyrUp',
                         colors=('lightcoral', 'firebrick'),
                         xticklabels=('dlightStable+Down', 'dlightUp'))
save_fig(fig, OUT_DIR_FIG, r'%pyrUp_barplot', save=save_plot)
# pyrDown
a = df_perc_pool['perc_pyrDown_no_dlightUp']
b = df_perc_pool['perc_pyrDown_dlightUp']
fig, ax = plt.subplots(dpi=300, figsize=(2,3))
pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
                         # ylim=(0, 45), 
                         ylabel='% PyrDown',
                         colors=('violet', 'Purple'),
                         xticklabels=('dlightStable+Down', 'dlightUp'))
save_fig(fig, OUT_DIR_FIG, r'%pyrDown_barplot', save=save_plot)

# # plot DA-stable vs DA_up
# # pyrUp
# a = df_perc_pool['perc_pyrUp_dlightStable']
# b = df_perc_pool['perc_pyrUp_dlightUp']
# fig, ax = plt.subplots(dpi=300, figsize=(2,3))
# pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
#                          # ylim=(0, 45), 
#                          ylabel='% PyrUp',
#                          colors=('lightcoral', 'firebrick'),
#                          xticklabels=('dlightStable', 'dlightUp'))
# plt.tight_layout()
# plt.show()
# # pyrDown
# a = df_perc_pool['perc_pyrDown_dlightStable']
# b = df_perc_pool['perc_pyrDown_dlightUp']
# fig, ax = plt.subplots(dpi=300, figsize=(2,3))
# pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
#                          # ylim=(0, 45), 
#                          ylabel='% PyrDown',
#                          colors=('violet', 'Purple'),
#                          xticklabels=('dlightStable', 'dlightUp'))
# plt.tight_layout()
# plt.show()


# # plot DA_down vs DA_up
# # pyrUp
# a = df_perc_pool['perc_pyrDown_dlightDown']
# b = df_perc_pool['perc_pyrDown_dlightUp']
# fig, ax = plt.subplots(dpi=300, figsize=(2,3))
# pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
#                          # ylim=(0, 45), 
#                          ylabel='% PyrDown',
#                          colors=('violet', 'Purple'),
#                          xticklabels=('dlightDown', 'dlightUp'))
# fig.tight_layout()
# plt.show()
# # pyrDown
# a = df_perc_pool['perc_pyrUp_dlightDown']
# b = df_perc_pool['perc_pyrUp_dlightUp']
# fig, ax = plt.subplots(dpi=300, figsize=(2,3))
# pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
#                          # ylim=(0, 45), 
#                          ylabel='% PyrUp',
#                          colors=('lightcoral', 'firebrick'),
#                          xticklabels=('dlightDown', 'dlightUp'))
# fig.tight_layout()
# plt.show()

#%% plot GECO response ratio difference (DA-Up vs non-DA-Up)

# # key_geco_amp = 'geco_amp'
# # key_geco_ratio = 'geco_amp'

# key_geco_amp = 'geco_zscore_amp'
# key_geco_ratio = 'geco_zscore_ratio'

    
# # helper: stack profiles then mean across rows
# def mean_profile_per_rec(g):
#     arr = np.stack(g.values)          # (n_rows_in_rec, T) or (n_rows_in_rec, ...)
#     return arr.mean(axis=0)           # (T) or (...)

# mask_a = (df_pool_sorted['dlight_type']=='Up') & (df_pool_sorted['geco_type']=='Up')
# mask_b = (df_pool_sorted['dlight_type']!='Up') & (df_pool_sorted['geco_type']=='Up')

# # mask_a = (df_pool_sorted['dlight_type']=='Up') & (df_pool_sorted['geco_ratio']>1.2)
# # mask_b = (df_pool_sorted['dlight_type']!='Up') & (df_pool_sorted['geco_ratio']>1.2)

# # series indexed by rec_id, each value is a mean profile (array)
# s_a = df_pool_sorted.loc[mask_a].groupby('rec_id')[key_geco_ratio].apply(mean_profile_per_rec)
# s_b = df_pool_sorted.loc[mask_b].groupby('rec_id')[key_geco_ratio].apply(mean_profile_per_rec)

# # keep only rec_ids that have both types (paired)
# common_rec_ids = s_a.index.intersection(s_b.index)
# s_a = s_a.loc[common_rec_ids]
# s_b = s_b.loc[common_rec_ids]

# # now stack into (n_rec, T)
# profile_a = np.stack(s_a.to_numpy())
# profile_b = np.stack(s_b.to_numpy())
# fig, ax = pf.plot_paired_violin(profile_a, profile_b,
#                        colors=('firebrick', 'lightcoral'),
#                        colname=['DA-UP\nPyrUp', 'non-DA-Up\nPyrUp'], 
#                        ylabel='ratio', ylim=None,
#                        title_prefix=None,
#                        )
# ax.set(ylim=(1, 3))
# # save_fig(fig, OUT_DIR_FIG, r'pyrUp_ratio_comp', save=save_plot)

# # response  amplitude
# s_a = df_pool_sorted.loc[mask_a].groupby('rec_id')[key_geco_amp].apply(mean_profile_per_rec)
# s_b = df_pool_sorted.loc[mask_b].groupby('rec_id')[key_geco_amp].apply(mean_profile_per_rec)

# # keep only rec_ids that have both types (paired)
# common_rec_ids = s_a.index.intersection(s_b.index)
# s_a = s_a.loc[common_rec_ids]
# s_b = s_b.loc[common_rec_ids]

# # now stack into (n_rec, T)
# profile_a = np.stack(s_a.to_numpy())
# profile_b = np.stack(s_b.to_numpy())
# fig, ax = pf.plot_paired_violin(profile_a, profile_b,
#                        colors=('firebrick', 'lightcoral'),
#                        colname=['DA-UP\nPyrUp', 'non-DA-Up\nPyrUp'], 
#                        ylabel='response_amp', ylim=None,
#                        title_prefix=None,
#                        )
# # ax.set(ylim=(1, 3))
# # save_fig(fig, OUT_DIR_FIG, r'pyrUp_response_amp_comp', save=save_plot)       
