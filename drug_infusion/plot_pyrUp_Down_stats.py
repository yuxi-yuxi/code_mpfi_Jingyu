# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 19:16:29 2025

@author: Jingyu Cao
"""

#%% imports 
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
# from scipy.ndimage import gaussian_filter1d

from drug_infusion.plot_functions import plot_population_heatmap
from drug_infusion.utils_infusion import sort_response

from common import plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()
from common.utils_basic import normalize

def profile_is_valid(x):
    if x is None:
        return False
    a = np.asarray(x)
    if a.size == 0:
        return False
    return np.isfinite(a).all()   # True only if no NaN/inf inside
#%% PATHS AND PARAMS

# session list 
drug = 'SCH'
# drug = 'prazosin'
# drug = 'propranolol'

import rec_lst_infusion as recs
if drug=='SCH':
    rec_drug = recs.rec_SCH
    rec_ctrl = recs.rec_SCH_ctrl
elif drug=='prazosin':
    rec_drug = recs.rec_praz
    rec_ctrl = recs.rec_praz_ctrl
elif drug=='propranolol':
    rec_drug = recs.rec_prop
    rec_ctrl = recs.rec_prop_ctrl

# PARAMS
pre_window=(-1, 0)
post_window=(0.5, 1.5)

bef, aft = 2, 4
time_windows=[(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4)] # time windows used to quantify dFF differecne

# pyrUp_by = 'zscore_amp_valid'
# thresh_up = 0.08
# thresh_down = -thresh_up
pyrUp_by = 'response_ratio_valid'
thresh_up = 1.12
thresh_down = 1/thresh_up

prof_col_heatmap = 'mean_profile_valid'
# prof_col_trace = 'mean_profile_zscore_valid'
prof_col_trace = 'mean_profile_valid'
ratio_col = pyrUp_by

thresh_baseline = 8.5
h=np.inf # no upper bound

DRD1_ONLY = 0

# PATHS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
# OUTPUT_RES = OUT_DIR_RAW_DATA /'processed_dataframe_new_good'
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_infusion_0625")/drug
if not OUT_DIR_FIG.exists():
    OUT_DIR_FIG.mkdir(parents=True)
save_plot=0

#%% data pooling
df_drug_pool = pd.DataFrame()
df_ctrl_pool = pd.DataFrame()
# data pooling for drug sessions
for rec_idx, rec in tqdm(rec_drug.iterrows(), total=len(rec_drug), desc="loading sessions"):
    anm    = rec['anm']
    date   = rec['date']
    rec_id = anm +'-'+date
    p_profile  = OUTPUT_RES/f'{anm}-{date}_raw_dff_profile_pre{pre_window}_post{post_window}.parquet'
    p_zscore_profile = OUTPUT_RES/f'{anm}-{date}_zscored_profile_pre{pre_window}_post{post_window}.parquet'
    df_profile =  pd.read_parquet( p_profile)
    df_zscored_profile =  pd.read_parquet(p_zscore_profile)
    df_profile['anm'] = anm
    df_profile['date'] = date
    df_profile['SCH_days'] = rec['SCH_days']
    df_profile['propranolol_days'] = rec['propranolol_days']
    df_profile['prazosin_days'] = rec['prazosin_days']
    df_profile['ctrl_days'] = rec['ctrl_days']
    df_profile['mean_profile_zscore_valid_ss1'] = df_zscored_profile['mean_profile_valid_ss1']
    df_profile['mean_profile_zscore_valid_ss2'] = df_zscored_profile['mean_profile_valid_ss2']
    df_profile['zscore_amp_valid_ss1'] = df_zscored_profile['response_amplitude_valid_ss1']
    df_profile['zscore_amp_valid_ss2'] = df_zscored_profile['response_amplitude_valid_ss2']
    df_profile['mean_profile_zscore_good_ss1'] = df_zscored_profile['mean_profile_good_ss1']
    df_profile['mean_profile_zscore_good_ss2'] = df_zscored_profile['mean_profile_good_ss2']
    df_profile['zscore_amp_good_ss1'] = df_zscored_profile['response_amplitude_good_ss1']
    df_profile['zscore_amp_good_ss2'] = df_zscored_profile['response_amplitude_good_ss2']
    df_drug_pool = pd.concat((df_drug_pool, df_profile))
    
# data pooling for ctrl sessions
for rec_idx, rec in tqdm(rec_ctrl.iterrows(), total=len(rec_ctrl), desc="loading sessions"):
    anm    = rec['anm']
    date   = rec['date']
    rec_id = anm +'-'+date
    p_profile  = OUTPUT_RES/f'{anm}-{date}_raw_dff_profile_pre{pre_window}_post{post_window}.parquet'
    p_zscore_profile = OUTPUT_RES/f'{anm}-{date}_zscored_profile_pre{pre_window}_post{post_window}.parquet'
    df_profile =  pd.read_parquet( p_profile)
    df_zscored_profile =  pd.read_parquet(p_zscore_profile)
    df_profile['anm'] = anm
    df_profile['date'] = date
    df_profile['SCH_days'] = rec['SCH_days']
    df_profile['propranolol_days'] = rec['propranolol_days']
    df_profile['prazosin_days'] = rec['prazosin_days']
    df_profile['ctrl_days'] = rec['ctrl_days']
    df_profile['mean_profile_zscore_valid_ss1'] = df_zscored_profile['mean_profile_valid_ss1']
    df_profile['mean_profile_zscore_valid_ss2'] = df_zscored_profile['mean_profile_valid_ss2']
    df_profile['zscore_amp_valid_ss1'] = df_zscored_profile['response_amplitude_valid_ss1']
    df_profile['zscore_amp_valid_ss2'] = df_zscored_profile['response_amplitude_valid_ss2']
    df_profile['mean_profile_zscore_good_ss1'] = df_zscored_profile['mean_profile_good_ss1']
    df_profile['mean_profile_zscore_good_ss2'] = df_zscored_profile['mean_profile_good_ss2']
    df_profile['zscore_amp_good_ss1'] = df_zscored_profile['response_amplitude_good_ss1']
    df_profile['zscore_amp_good_ss2'] = df_zscored_profile['response_amplitude_good_ss2']
    df_ctrl_pool = pd.concat((df_ctrl_pool, df_profile))    
#%%
df_drug_pool_pyr = sort_response(df_drug_pool, thresh_up, thresh_down,
                                 ratio_type=ratio_col,)
df_ctrl_pool_pyr = sort_response(df_ctrl_pool, thresh_up, thresh_down,
                                 ratio_type=ratio_col ,
                                 )
# select valid ROIs
df_drug_pool_pyr['gcamp_valid_ss1'] = (df_drug_pool_pyr['dff_baseline_min_ss1'].between(thresh_baseline, h))&(df_drug_pool_pyr[f'{prof_col_heatmap}_ss1'].apply(profile_is_valid))
df_drug_pool_pyr['gcamp_valid_ss2'] = (df_drug_pool_pyr['dff_baseline_min_ss2'].between(thresh_baseline, h))&(df_drug_pool_pyr[f'{prof_col_heatmap}_ss2'].apply(profile_is_valid))
df_drug_pool_pyr = df_drug_pool_pyr.loc[(df_drug_pool_pyr['gcamp_valid_ss1']) & (df_drug_pool_pyr['gcamp_valid_ss2'])]
df_ctrl_pool_pyr['gcamp_valid_ss1'] = (df_ctrl_pool_pyr['dff_baseline_min_ss1'].between(thresh_baseline, h))&(df_ctrl_pool_pyr[f'{prof_col_heatmap}_ss1'].apply(profile_is_valid))
df_ctrl_pool_pyr['gcamp_valid_ss2'] = (df_ctrl_pool_pyr['dff_baseline_min_ss2'].between(thresh_baseline, h))&(df_ctrl_pool_pyr[f'{prof_col_heatmap}_ss2'].apply(profile_is_valid))
df_ctrl_pool_pyr = df_ctrl_pool_pyr.loc[(df_ctrl_pool_pyr['gcamp_valid_ss1']) & (df_ctrl_pool_pyr['gcamp_valid_ss2'])]
                        
# selection recordings for statistics                                                          
# select recordings (rec_id) with n_keep_trials > n for both ss1 and ss2
valid_recs = df_drug_pool_pyr.groupby(['anm', 'date']).apply(
    lambda g: (g['n_keep_trial_valid_ss1'].iloc[0] > 15) & (g['n_keep_trial_valid_ss2'].iloc[0] > 15),
    include_groups=False)
valid_recs = valid_recs[valid_recs].index
df_drug_pool_pyr = df_drug_pool_pyr.set_index(['anm', 'date']).loc[valid_recs].reset_index()

valid_recs = df_ctrl_pool_pyr.groupby(['anm', 'date']).apply(
    lambda g: (g['n_keep_trial_valid_ss1'].iloc[0] > 15) & (g['n_keep_trial_valid_ss2'].iloc[0] > 15),
    include_groups=False)
valid_recs = valid_recs[valid_recs].index
df_ctrl_pool_pyr = df_ctrl_pool_pyr.set_index(['anm', 'date']).loc[valid_recs].reset_index()

# Select only the first 3 recording dates per animal for ctrl
df_drug_pool_pyr_first3 = df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{drug}_days']<4]
df_drug_pool_pyr = df_drug_pool_pyr_first3
df_drug_pool_pyr.to_parquet(OUTPUT_RES/ rf"df_drug_pool_pyr_first3_{drug}_0625.parquet")

animals_in_drug = df_drug_pool_pyr_first3['anm'].unique()
df_ctrl_pool_pyr_first3 = df_ctrl_pool_pyr.loc[(df_ctrl_pool_pyr['ctrl_days']<4)&
                                               (df_ctrl_pool_pyr['anm'].isin(animals_in_drug))]
df_ctrl_pool_pyr = df_ctrl_pool_pyr_first3
df_ctrl_pool_pyr.to_parquet(OUTPUT_RES/ rf"df_ctrl_pool_pyr_first3_{drug}_0625.parquet")

if DRD1_ONLY:
    df_drug_pool_pyr = df_drug_pool_pyr.loc[df_drug_pool_pyr['drd1+']]
    df_ctrl_pool_pyr = df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr['drd1+']]
#% plot heatmap
# drug sessions
rec_id =  f'{drug}-drug'

prefix = 'ss1_baseline'
fig=plot_population_heatmap(df_drug_pool_pyr, rec_id, bef, aft, 'ss1', prefix=prefix,
                            session_for_sorting='ss1', activity_profile=prof_col_heatmap, ratio=ratio_col,
                            plot_mean=0)
save_fig(fig, OUT_DIR_FIG, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)

prefix = f'ss2_{drug}'
fig=plot_population_heatmap(df_drug_pool_pyr, rec_id, bef, aft, 'ss2', prefix=prefix,
                            session_for_sorting='ss2', activity_profile=prof_col_heatmap, ratio=ratio_col,
                            plot_mean=0)
save_fig(fig, OUT_DIR_FIG, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)


# saline ctrl
rec_id = f'{drug}-ctrl'

prefix = 'ss1_baseline'
fig=plot_population_heatmap(df_ctrl_pool_pyr, rec_id, bef, aft, 'ss1', prefix=prefix,
                            session_for_sorting='ss1', activity_profile=prof_col_heatmap, ratio=ratio_col,
                            plot_mean=0)
save_fig(fig, OUT_DIR_FIG, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)

prefix = 'ss2_saline'
fig=plot_population_heatmap(df_ctrl_pool_pyr, rec_id, bef, aft, 'ss2', prefix=prefix,
                            session_for_sorting='ss2', activity_profile=prof_col_heatmap, ratio=ratio_col,
                            plot_mean=0)
save_fig(fig, OUT_DIR_FIG, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)


#% plot mean traces
xaxis = np.arange(30*(bef+aft))/30-bef    
trace_col = prof_col_trace
# ratio_col = 'response_ratio_valid'

if trace_col == 'mean_profile_zscore_valid':
    factor = 1
    dffbar = 0.05
    dff_label = 'zscore F'
elif trace_col == 'mean_profile_valid':
    factor = 100
    dffbar = 2
    dff_label = '%dF/F'
for cell_type in ['pyrUp', 'pyrDown']:
    # drug ss1 vs ss2
    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
    profile_a = factor*np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss1'], f'{trace_col}_ss1'])
    profile_b = factor*np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])      
    pf.plot_two_traces_with_binned_stats(profile_a, profile_b, 
                                         bef=bef, aft=aft,
                                         ax=ax,
                                         test='ranksum',
                                         time_windows=time_windows,
                                         baseline_window = pre_window,
                                         # baseline_window = None,
                                         labels = ['baseline', f'{drug}'],
                                         colors = ['steelblue', 'orange'],
                                         scalebar_dff=dffbar,
                                         sample_freq=30,
                                         show_scalebar=1,)
    ax.set(xlim=(-1, 4))
    ax.legend(frameon=False)
    save_fig(fig, OUT_DIR_FIG, fig_name=f'{drug}_{cell_type}_mean_trace_ss1_ss2_stat', save=save_plot)
    
    fig, ax = plt.subplots(dpi=300, figsize=(2,2))
    pf.plot_two_traces_with_scalebars_fixbar(profile_a, profile_b ,
                                         xaxis, ax,
                                         colors = ('steelblue', 'orange'),
                                         timebar=0.5, 
                                         dffbar=dffbar, 
                                         dff_label=dff_label,
                                         show_xaxis=1, xlabel='time from run (s)',
                                         scale_by_dffbar=True,
                                         dffbar_plot_height=0.6,
                                         bar_y_bump_frac=0.05)
    ax.set(xlim=(-1, 4))
    save_fig(fig, OUT_DIR_FIG, fig_name=f'{drug}_{cell_type}_mean_trace_ss1_ss2', save=save_plot)
    
    fig, ax = plt.subplots(dpi=300, figsize=(2, 1.5))
    pf.plot_mean_trace(profile_a, ax, xaxis, color='steelblue')
    pf.plot_mean_trace(profile_b, ax, xaxis, color='orange')
    ax.set(xlim=(-1, 4), ylabel= dff_label)
    save_fig(fig, OUT_DIR_FIG, fig_name=f'{drug}_{cell_type}_mean_trace_ss1_ss2_sharedY', save=save_plot)
    
    # plot ROI mean trace normalized trace
    fig, ax = plt.subplots(figsize=(2, 1.5), dpi=300)
    a = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss1'], f'{trace_col}_ss1'])
    a = normalize(a)
    pf.plot_mean_trace(a, 
                       ax, color='steelblue', label='ss1')
    b = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])
    b = normalize(b)
    pf.plot_mean_trace(b,
                       ax, color='orange', label=f'ss2_{drug}')
    ax.set(ylabel=f'norm. {dffbar}')
    ax.legend(frameon=False)
    save_fig(fig, OUT_DIR_FIG, fig_name='', save=0)
    
    
    # plot session mean trace
    # fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
    # cell_type = 'pyrUp_ss1'
    # grouped = df_drug_pool_pyr.loc[df_drug_pool_pyr[cell_type]].groupby(['anm', 'date'])
    # profile_a = factor*np.stack(grouped[f'{trace_col}_ss1'].mean())
    # cell_type = 'pyrUp_ss2'
    # grouped = df_drug_pool_pyr.loc[df_drug_pool_pyr[cell_type]].groupby(['anm', 'date'])
    # profile_b = factor*np.stack(grouped[f'{trace_col}_ss2'].mean())
    # pf.plot_two_traces_with_binned_stats(profile_a, profile_b, ax,
    #                                       test='ranksum',
    #                                       time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
    #                                       baseline_window=(-0.5, 0.5),
    #                                       labels = ['baseline', f'{drug}'],
    #                                       colors = ['steelblue', 'orange'],
    #                                       bef=2, aft=4, sample_freq=30,
    #                                       scalebar_dff=1,
    #                                       show_scalebar=True,
    #                                       )
    # ax.set(xlim=(-1, 4))
    # ax.legend(frameon=False)
    # save_fig(fig, OUT_DIR_FIG, fig_name='', save=save_plot)
    
    
    # plot saline ss1 vs ss2
    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
    profile_a = factor*np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr[f'{cell_type}_ss1'], f'{trace_col}_ss1'])
    profile_b = factor*np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])       
    pf.plot_two_traces_with_binned_stats(profile_a, profile_b,
                                         bef=bef, aft=aft,
                                         ax=ax,
                                         test='ranksum',
                                         time_windows=time_windows,
                                         # baseline_window = pre_window,
                                         baseline_window = None,
                                         labels = ['baseline', f'saline ({drug})'],
                                         colors = ['steelblue', 'grey'],
                                         sample_freq=30,
                                         scalebar_dff=dffbar,
                                         show_scalebar=1,
                                          )
    ax.set(xlim=(-1, 4))
    ax.legend(frameon=False)
    save_fig(fig, OUT_DIR_FIG, fig_name=f'saline_{cell_type}_mean_trace_ss1_ss2_stat', save=save_plot)
    
    fig, ax = plt.subplots(dpi=300, figsize=(2, 1.5))
    pf.plot_mean_trace(profile_a, ax, xaxis, color='steelblue')
    pf.plot_mean_trace(profile_b, ax, xaxis, color='grey')
    ax.set(xlim=(-1, 4), ylabel=dff_label)
    save_fig(fig, OUT_DIR_FIG, fig_name=f'saline_{cell_type}_mean_trace_ss1_ss2_sharedY', save=save_plot)


    # plot mean trace compare saline and SCH
    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
    profile_a = factor*np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])
    profile_b = factor*np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])     
    pf.plot_two_traces_with_binned_stats(profile_a, profile_b, 
                                         bef=bef, aft=aft,
                                         ax=ax,
                                          test='ranksum',
                                          time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                                          # baseline_window = pre_window,
                                          baseline_window = None,
                                          labels = [f'saline ({drug})', f'{drug}'],
                                          colors = ['grey', 'orange'],
                                          sample_freq=30,
                                          scalebar_dff=dffbar,
                                          show_scalebar=True,
                                          )
    ax.set(xlim=(-1, 4))
    ax.legend(frameon=False)
    save_fig(fig, OUT_DIR_FIG, fig_name=f'{cell_type}_mean_trace_saline_{drug}_stat', save=save_plot)
    
    # plot sharedY mean traces
    fig, ax = plt.subplots(dpi=300, figsize=(2, 1.5))
    pf.plot_mean_trace(profile_a, ax, xaxis, color='grey')
    pf.plot_mean_trace(profile_b, ax, xaxis, color='orange')
    ax.set(xlim=(-1, 4), ylabel=dff_label)
    save_fig(fig, OUT_DIR_FIG, fig_name=f'{cell_type}_mean_trace_saline_{drug}_sharedY', save=save_plot)
    


#% quantifying mean amplitude difference
# amp_col = 'response_amplitude_valid'
# for cell_type in ['pyrUp', 'pyrDown']:
#     # drug ss1 vs ss2
#     profile_a = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss1'], f'{amp_col}_ss1'])
#     profile_b = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2'], f'{amp_col}_ss2'])

#     fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
#     pf.plot_unpaired_violin(profile_a, profile_b,
#                                     ylabel= f'{cell_type}_response_amp',
#                                     colors=['steelblue', 'orange'],
#                                     colname=['baseline', f'{drug}'],
#                                     ax = ax,
#                                     markersize = 1
#                                     # ylim=ylim,
#                                     )
#     save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_{drug}', save=0)
    
#     # Get session-mean amplitude for cells classified in ss1
#     grouped_a = df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss1']].groupby(['anm', 'date'])
#     profile_a = np.array([g[f'{amp_col}_ss1'].mean() for _, g in grouped_a])
    
#     # Get session-mean amplitude for cells classified in ss2
#     grouped_b = df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2']].groupby(['anm', 'date'])
#     profile_b = np.array([g[f'{amp_col}_ss2'].mean() for _, g in grouped_b])


#     fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
#     pf.plot_bar_with_paired_scatter(ax, profile_a, profile_b,
#                                     ylabel=f'{cell_type}_response_amp',
#                                     colors=['steelblue', 'orange'],
#                                     xticklabels=['baseline', f'{drug}'],
#                                     # ylim=ylim,
#                                     )
#     save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_{drug}', save=0)   
        
#% Bar plot
grouped = df_drug_pool_pyr.groupby(['anm', 'date'],)
recs = [name for name, g in grouped]
df_perc_drug = grouped.apply(
    lambda g: pd.Series({
        'perc_pyrUp_ss1':   g['pyrUp_ss1'].sum()   / len(g),
        'perc_pyrUp_ss2':   g['pyrUp_ss2'].sum()   / len(g),
        'perc_pyrDown_ss1': g['pyrDown_ss1'].sum()  / len(g),
        'perc_pyrDown_ss2': g['pyrDown_ss2'].sum()  / len(g),
        'delta_perc_pyrUp':   g['pyrUp_ss2'].sum()   / len(g) - g['pyrUp_ss1'].sum()   / len(g),
        'delta_perc_pyrDown': g['pyrDown_ss2'].sum()  / len(g) - g['pyrDown_ss1'].sum()  / len(g),
        'n_rois': len(g)
    }), include_groups=False
).reset_index()

grouped = df_ctrl_pool_pyr.groupby(['anm', 'date'],)
df_perc_ctrl = grouped.apply(
    lambda g: pd.Series({
        'perc_pyrUp_ss1':   g['pyrUp_ss1'].sum()   / len(g),
        'perc_pyrUp_ss2':   g['pyrUp_ss2'].sum()   / len(g),
        'perc_pyrDown_ss1': g['pyrDown_ss1'].sum()  / len(g),
        'perc_pyrDown_ss2': g['pyrDown_ss2'].sum()  / len(g),
        'delta_perc_pyrUp':   g['pyrUp_ss2'].sum()   / len(g) - g['pyrUp_ss1'].sum()   / len(g),
        'delta_perc_pyrDown': g['pyrDown_ss2'].sum()  / len(g) - g['pyrDown_ss1'].sum()  / len(g),
        'n_rois': len(g)
    }), include_groups=False
).reset_index()


for cell_type, ylim in zip(['pyrUp', 'pyrDown'], [(0, 65), (0, 65)]):
    
    # drug sessions
    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    res_drug = pf.plot_bar_with_paired_scatter(ax, 100*df_perc_drug[f'perc_{cell_type}_ss1'], 100*df_perc_drug[f'perc_{cell_type}_ss2'],
                                    ylabel=f'% {cell_type}',
                                    colors=['steelblue', 'orange'],
                                    xticklabels=['baseline', f'{drug}'],
                                    ylim=ylim,
                                    )
    save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_{drug}', save=save_plot)

    # ctrl sessions
    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    res_ctrl = pf.plot_bar_with_paired_scatter(ax, 100*df_perc_ctrl[f'perc_{cell_type}_ss1'], 100*df_perc_ctrl[f'perc_{cell_type}_ss2'],
                                    ylabel=f'% {cell_type}',
                                    colors=['steelblue', 'grey'],
                                    xticklabels=['baseline', f'saline({drug})'],
                                    ylim=ylim,
                                    )
    save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_ctrl', save=save_plot)

    # saline vs drug sessions
    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    res_ctrl_drug = pf.plot_bar_with_unpaired_scatter(ax, 100*df_perc_ctrl[f'perc_{cell_type}_ss2'], 100*df_perc_drug[f'perc_{cell_type}_ss2'],
                                    ylabel=f'% {cell_type}',
                                    colors=['grey', 'orange'],
                                    xticklabels=['saline', f'{drug}'],
                                    ylim = (-15, 15))
    save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_saline_{drug}', save=save_plot)


    # Δ% to baseline
    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    res_delta = pf.plot_bar_with_unpaired_scatter(ax, 100*df_perc_ctrl[f'delta_perc_{cell_type}'], 100*df_perc_drug[f'delta_perc_{cell_type}'],
                                    ylabel=f'Δ% {cell_type} (vs baseline)',
                                    colors=['grey', 'orange'],
                                    xticklabels=[f'saline ({drug})', f'{drug}'],
                                    ylim = (-15, 15))
    print(f'{cell_type}: {res_delta}')
    save_fig(fig, OUT_DIR_FIG, fig_name=f'delt_perc_{cell_type}_saline_{drug}', save=save_plot)



    



