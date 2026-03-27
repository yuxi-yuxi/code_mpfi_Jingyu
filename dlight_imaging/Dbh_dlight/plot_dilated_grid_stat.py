# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 12:24:42 2026

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
from common import plotting_functions_Jingyu as pf
pf.mpl_formatting()
save_fig = pf.save_fig
from common.utils_basic import normalize
# Load recording list
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst
from dlight_imaging.Dbh_dlight.decay_time_fitting import compute_tau_with_qc, plot_tau_fit_new

from scipy.optimize import curve_fit
def _exp_decay(d, A, tau):
    return A * np.exp(-d / tau)

def profile_is_valid(x):
    if x is None:
        return False
    a = np.asarray(x)
    if a.size == 0:
        return False
    return np.isfinite(a).all()   # True only if no NaN/inf inside
#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
# OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_grid_free_dilation'
OUT_DIR_DF = OUT_DIR_RAW_DATA/'processed_dataframe_grid_free_dilation'

OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_Dbh_dlight")
save_plot = 0

dlight_pre  = (-1, 0)
dlight_post = (0, 1)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
regression_name ='single_trial_regression'
# DILATION_STEPS = (0, 2, 4, 6, 8, 10)
DILATION_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


#%%
df_stats_all = pd.DataFrame()

for rec in rec_lst:
    print(f'loading: {rec}-------------------------------------------')
    anm, date, ss = rec.split('-')
    p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
    for k in DILATION_STEPS:
        p_stats = (OUT_DIR_DF / 
                   rf"{rec}_profile_combined_dilation={k}_pre{dlight_pre}_post{dlight_post}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.parquet")
        roi_stats = pd.read_parquet(p_stats)
        roi_stats['roi_id_tuple'] = roi_stats['roi_id'].apply(tuple)
        df_stats_all = pd.concat((df_stats_all, roi_stats))
        
# for thresh_baseline_dlight in np.arange(0.5, 2.25, 0.25):
#     for thresh_baseline_red in np.arange(0.5, 1.5, 0.5):
#         for thresh_npix in [0, 20, 30]:
            # try:
thresh_baseline_dlight = 2    
thresh_baseline_red    = 1
thresh_npix = 0
# OUT_DIR_FIG = (OUT_DIR_RAW_DATA/'TEST_PLOTS'/'spatial_dialtion_fitting_check_roi'/f'dlight_pre{dlight_pre}_post{dlight_post}'
#                # /f'shuff_thresh_up={amp_shuff_thresh_up}_ES={effect_size_thresh}'/f'min_npix={thresh_npix}'
#               )
# if not OUT_DIR_FIG.exists():
#     OUT_DIR_FIG.mkdir(parents=True)
# save_plot=0

df_stats_all['dlight_valid'] = (
    (df_stats_all['baseline_dlight_min'] > thresh_baseline_dlight) &
    (df_stats_all['mean_profile'].apply(profile_is_valid))&
    (df_stats_all['n_pixels_axon_and_dlight']>thresh_npix)
    )
df_stats_all['red_valid'] = (
    (df_stats_all['baseline_red_min'] > thresh_baseline_red)&
    (df_stats_all['mean_profile_red'].apply(profile_is_valid))& 
    (df_stats_all['n_pixels_axon_and_dlight']>thresh_npix)
    )

#%%
# df_stats_all_selected = df_stats_all.loc[(df_stats_all['dlight_valid'])&(df_stats_all['red_valid'])&(~df_stats_all['edge'])]
df_stats_all_selected = df_stats_all.loc[(df_stats_all['dlight_valid'])
                                         # &(df_stats_all['red_valid'])
                                         &(~df_stats_all['edge'])]
#% plot session mean +/- sem for different dialtion steps
df_dlightUp_all  = pd.DataFrame()
df_dilation_stat = pd.DataFrame()
for g_idx, df_rec in df_stats_all_selected.groupby('rec_id'):
    # start with rois that are dLightUp without dilation
    dlightUp_mask = (df_rec['dilation_k'] == 0) & (df_rec['Up']) & (df_rec['red_valid'])
    dlightUp_rois = df_rec.loc[dlightUp_mask, 'roi_id'].apply(tuple).values
    df_dlightUp_dilations = df_rec[
        df_rec['roi_id'].apply(tuple).isin(dlightUp_rois)
    ]
    df_dlightUp_all = pd.concat((df_dlightUp_all, df_dlightUp_dilations))
    
#%% organise data per ROI and fit decay per ROI
# Each ROI has one response_amplitude at each dilation step
df_roi_dilation = df_dlightUp_all.sort_values(['rec_id', 'roi_id_tuple', 'dilation_k'])

amp_axis = list(range(11))
amps = {str(k): [] for k in amp_axis}

fit_res_all = []
# tau_results_all = []

for (rec_id, roi_id), df_roi in df_roi_dilation.groupby(['rec_id', 'roi_id_tuple']):
    d_fit = df_roi['dilation_k'].values
    R = df_roi['response_amplitude'].values
    # rec_roi_id = (rec_id, roi_id)
    rec, (y, x) = (rec_id, roi_id)
    rec_roi_id = f"{rec}_{int(y)}_{int(x)}"
    # if len(d_fit) <= 3:
    #     continue

    # collect per-dilation amplitudes for the mean curve
    # for dk, amp_val in zip(d_fit, R):
    #     k_str = str(int(dk))
    #     if k_str in amps:
    #         amps[k_str].append(amp_val)

    # find 'peak' only in the 0 or 1 pix dilation
    # peak = np.argmax(R[:min(2, len(R))])
    peak = 0
    
    fit_res = compute_tau_with_qc(d_fit, R, peak_index=peak, min_points=6)
    fit_res['d_fit'] = d_fit
    fit_res['R'] = R
    fit_res['rec_id'] = rec_id
    fit_res['rec_roi_id'] = rec_roi_id
    fit_res['roi_id_tuple'] = roi_id
    fit_res_all.append(fit_res)
    
    # fig, ax = plot_tau_fit_new(d_fit, R, fit_res,
    #                        peak_index=peak,
    #                        title_prefix=f'{rec_roi_id}\n',
    #                        xlabel='dilated pix',
    #                        ylabel='res. amp.',
    #                        title_size=8,
    #                        fitting_curve=1)
    # # out_dir = OUT_DIR_FIG/f'thr_base_dlight{thresh_baseline_dlight}_thr_base_geco={thresh_baseline_red}_roi_fitting_check_grid_free'
    # out_dir = OUT_DIR_FIG
    # if not out_dir.exists():
    #     out_dir.mkdir(parents=True)
    # save_fig(fig, out_dir, f'{rec_roi_id}_fitting_check', forms=['png', ], save=0)
        
    # tau_results_all.append(fit_res['tau'])

df_fit_res_all = pd.DataFrame(fit_res_all)
#%% histogram of spatial tau per time window
# hist_stem = save_stem / 'tau_histograms'
# hist_stem.mkdir(parents=True, exist_ok=True)

TAU_MAX = 10
BIN_WIDTH = 1
bins = np.arange(0, TAU_MAX + BIN_WIDTH, BIN_WIDTH)

# thresh_r2 = 0.5

# OUT_DIR_FIG = (OUT_DIR_RAW_DATA/'TEST_PLOTS'/'spatial_dialtion_fitting_check_roi'/f'dlight_pre{dlight_pre}_post{dlight_post}/stats'
#                # /f'shuff_thresh_up={amp_shuff_thresh_up}_ES={effect_size_thresh}'/f'min_npix={thresh_npix}'
#               )
# if not OUT_DIR_FIG.exists():
#     OUT_DIR_FIG.mkdir(parents=True)
# save_plot=0

# for thresh_r2 in np.arange(0.3, 0.55, 0.05):

thresh_r2 = 0.3

# only include ROIs with high r2 fitting
# selected_rec = df_fit_res_all.loc[
#     df_fit_res_all['r2'] > thresh_r2,
#     ['rec_id', 'roi_id_tuple', 'tau', 'r2']
# ].drop_duplicates()

selected_rec = df_fit_res_all.loc[
    df_fit_res_all['r2'] > thresh_r2,
    # ['rec_id', 'roi_id_tuple', 'tau']
].drop_duplicates(subset=['rec_id', 'roi_id_tuple'])

df_stats_to_plot = df_dlightUp_all.merge(
    selected_rec,
    on=['rec_id', 'roi_id_tuple',],
    how='inner'
)

# session mean
df_plot_rec = (
    df_stats_to_plot
    .groupby(['dilation_k', 'rec_id'], as_index=False)
    .agg(response_amplitude_mean=('response_amplitude', 'mean'))
)

# fitting for each session
fit_res_all_session = []

for rec_id, df_session_all_rois in df_stats_to_plot.groupby('rec_id'):
    
    df_session = (
        df_session_all_rois
        .groupby(['dilation_k',], as_index=False)
        .agg(response_amplitude_mean=('response_amplitude', 'mean'))
    )

    d_fit = df_session['dilation_k'].values
    R = df_session['response_amplitude_mean'].values
    # rec_id = df_session['rec_id'].iloc[0]
    #     continue

    # find 'peak' only in the 0 or 1 pix dilation
    # peak = np.argmax(R[:min(2, len(R))])
    peak = 0
    
    fit_res = compute_tau_with_qc(d_fit, R, peak_index=peak, min_points=3)
    fit_res['rec_id'] = rec_id
    fit_res_all_session.append(fit_res)
    
    # --- overlay each ROI trace for this session ---
    df_roi_session = df_stats_to_plot[df_stats_to_plot['rec_id'] == rec_id].copy()
    
    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=300)
    
    for roi_id_tuple, df_roi in df_roi_session.groupby('roi_id_tuple'):
        df_roi = df_roi.sort_values('dilation_k')
        # plot data points for single ROIs
        # ax.plot(
        #     df_roi['dilation_k'],
        #     df_roi['response_amplitude'],
        #     # '-o',
        #     '.',
        #     color='lightgray',
        #     markerfacecolor='orange',
        #     markeredgecolor='none',
        #     # linewidth=2,
        #     markersize=4,
        #     alpha=0.7,
        #     zorder=1
        # )
        
        # plot fitting curve for single ROIs
        A, t_fit, tau, C, r2 = df_roi[['A', 'd_fit', 'tau', 'C', 'r2']].iloc[0]
        if r2>thresh_r2:
            y_pred = A * np.exp(-(t_fit - t_fit[0]) / tau) + C
            ax.plot(
                t_fit, y_pred,
                '--',
                linewidth=0.5,
                color='lightblue',
                # label='Exp fit'
            )
        
    fig, ax = plot_tau_fit_new(
        d_fit, R, fit_res,
        ax=ax,
        peak_index=peak,
        c_dp = 'darkgray',
        marker_dp=None, 
        title_prefix=f'{rec_id}\n',
        xlabel='dilated pix',
        ylabel='res. amp.',
        title_size=8,
        # fitting_curve=1
    )
    
    # plot data points, mean+/-SEM for single ROIs
    df_plot = (
        df_session_all_rois
        .groupby('dilation_k', as_index=False)
        .agg(
            response_amplitude_mean=('response_amplitude', 'mean'),
            response_amplitude_median=('response_amplitude', 'median'),
            sem_amp=('response_amplitude', lambda x: sem(x, nan_policy='omit'))
        )
        .sort_values('dilation_k')
    )
    x = df_plot['dilation_k'].to_numpy()
    y = df_plot['response_amplitude_mean'].to_numpy()
    yerr = df_plot['sem_amp'].to_numpy()
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='_', 
        color='dimgrey',
        ecolor='darkgray',
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        markersize=8
    )
    
    # --- ylim based on ROI data ---
    y_all = df_roi_session['response_amplitude'].to_numpy()
    y_all = y_all[np.isfinite(y_all)]
    
    if len(y_all) > 0:
        y_min = y_all.min()
        y_max = y_all.max()
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.01
        # ax.set_ylim(max(y_min - pad, -0.05), y_max + pad)
        ax.set_ylim(max(y_min - pad, -0.025), 0.075)
    # out_dir = OUT_DIR_FIG/f'thr_base_dlight{thresh_baseline_dlight}_thr_base_geco={thresh_baseline_red}_roi_fitting_check_grid_free'
    out_dir = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\TEST_PLOTS\spatial_tau_fitting_example_sessions_roi_fitting")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    save_fig(fig, out_dir, f'{rec_id}_fitting_check', 
             # forms=['png', ], 
             save=1)
df_fit_res_all_session = pd.DataFrame(fit_res_all_session)

# per roi histogram
vals = df_stats_to_plot.drop_duplicates(
    subset=['roi_id_tuple', 'rec_id']
)['tau']

# session average histogram
# vals = df_fit_res_all_session['tau']

# vals = (
#     df_stats_to_plot
#     .drop_duplicates(subset=['roi_id_tuple', 'rec_id'])
#     .groupby('rec_id')['tau']
#     .mean()
# )

fig, ax = plt.subplots(figsize=(2.4, 2.1))

ax.hist(
    vals,
    bins=bins,
    color='lightblue',
    edgecolor='none',
    linewidth=0.4
)

med = np.nanmedian(vals)
ax.axvline(med, color='teal', linestyle='--')
ax.text(
    0.9, 0.98,
    f'Median = {med:.2f}',
    transform=ax.transAxes,
    ha='right',
    va='top',
    fontsize=7,
    color='teal'
)

ax.spines[['top', 'right']].set_visible(False)
ax.set(
    title=f'r2>{thresh_r2:.2f}',
    xlabel=r'Spatial $\tau$ (px)',
    xlim=(0, TAU_MAX),
    ylabel='roi count',
    )
ax.set_xticks(np.arange(0, TAU_MAX + BIN_WIDTH, 2))

save_fig(fig, OUT_DIR_FIG,  f'thr_base_dlight{thresh_baseline_dlight}_thr_base_geco={thresh_baseline_red}_tau_hist_rois_min_r2={thresh_r2:.2f}', 
         save=save_plot)

#% median radial decay curves with IQR

# roi mean
# df_plot = (
#     df_stats_to_plot
#     .groupby('dilation_k', as_index=False)
#     .agg(
#         response_amplitude_mean=('response_amplitude', 'mean'),
#         sem_amp=('response_amplitude', lambda x: sem(x, nan_policy='omit'))
#     )
#     .sort_values('dilation_k')
# )

# session mean
df_plot = (
    df_plot_rec
    .groupby('dilation_k', as_index=False)
    .agg(
        response_amplitude_mean=('response_amplitude_mean', 'mean'),
        sem_amp=('response_amplitude_mean', lambda x: sem(x, nan_policy='omit'))
    )
    .sort_values('dilation_k')
)

# plot
x = df_plot['dilation_k'].to_numpy()
y = df_plot['response_amplitude_mean'].to_numpy()
yerr = df_plot['sem_amp'].to_numpy()

fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=300)

# --- plot each session trace ---
# for rec_id, df_rec in df_plot_rec.groupby(['rec_id']):
#     df_rec = df_rec.sort_values('dilation_k')
#     ax.plot(
#         df_rec['dilation_k'],
#         df_rec['response_amplitude_mean'],
#         '-o',
#         color='lightgray',
#         markerfacecolor='orange',  # marker fill
#         markeredgecolor=None,  # marker edge
#         linewidth=0.5,
#         markersize=1,
#         alpha=0.7,
#         zorder=1
#     )
    
# # --- plot each ROI trace ---
# for rec_id, df_roi in df_stats_to_plot.groupby(['rec_id', 'roi_id_tuple']):
#     df_roi = df_roi.sort_values('dilation_k')
#     ax.plot(
#         df_roi['dilation_k'],
#         df_roi['response_amplitude'],
#         '-o',
#         color='lightgray',
#         markerfacecolor='orange',  # marker fill
#         markeredgecolor=None,  # marker edge
#         linewidth=0.5,
#         markersize=1,
#         alpha=0.7,
#         zorder=1
#     )    

# --- overlay population summary ---
ax.errorbar(
    x,
    y,
    yerr=yerr,
    fmt='_',
    color='teal',
    ecolor='lightblue',
    elinewidth=1.2,
    capsize=3,
    capthick=1.2,
    markersize=8
)

ax.spines[['top', 'right']].set_visible(False)
ax.set(
    title=f'r2>{thresh_r2:.2f}',
    xlabel='Dilation (px)',
    ylabel='Mean dLight response amp.',
    xlim=(x.min()-1, x.max()+1),
    ylim=(-0, 0.03)
)
save_fig(fig, OUT_DIR_FIG,  f'thr_base_dlight{thresh_baseline_dlight}_thr_base_geco={thresh_baseline_red}_dilation_stat_roi_mean_min_r2={thresh_r2:.2f}', 
         save=save_plot)

#%% plot single ROI fitting for example ROIs
out_dir = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\TEST_PLOTS\spatial_tau_fitting_example_rois")
df_rois_fitting = df_stats_to_plot.drop_duplicates(
    subset=['roi_id_tuple', 'rec_id'])
df_example_rois = df_rois_fitting.loc[
                                      # (df_rois_fitting['tau']>4.5)&
                                      # (df_rois_fitting['r2']>0.5)&
                                      (df_rois_fitting['n_points']<7)]
for _, df_roi in df_example_rois.iterrows():
    roi_id = df_roi['rec_roi_id']
    fig, ax = plot_tau_fit_new(df_roi['d_fit'], df_roi['R'], df_roi,
                           peak_index=peak,
                           title_prefix=f'{roi_id}\n',
                           xlabel='dilated pix',
                           ylabel='res. amp.',
                           title_size=8,
                           fitting_curve=1)
    save_fig(fig, out_dir,  f'{roi_id}', 
             save=0)
#%% check correlation between r2 and tau
df_rois_fitting = df_stats_to_plot.drop_duplicates(
    subset=['roi_id_tuple', 'rec_id']
)
fig, ax = plt.subplots()
ax.scatter(df_rois_fitting['r2'], df_rois_fitting['tau'], s=3)
ax.set(xlabel='r2', ylabel='tau')                
plt.show()

fig, ax = plt.subplots()
ax.hist(df_rois_fitting['r2'], bins=50)
ax.set(xlabel='r2')

fig, ax = plt.subplots()
ax.hist(df_rois_fitting['tau'], bins=50)
ax.set(xlabel='tau')
