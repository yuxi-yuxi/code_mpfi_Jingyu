# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:31:35 2026

@author: Jingyu Cao
"""

#%% imports and funcs
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from common.event_response_quantification import align_trials
from common.trial_selection import seperate_valid_trial
def profile_is_valid(x):
    if x is None:
        return False
    a = np.asarray(x)
    if a.size == 0:
        return False
    return np.isfinite(a).all()   # True only if no NaN/inf inside

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

#%% PATHS AND PARAMS
# run-onset response window
dlight_pre  = (-1, 0)
dlight_post = (0, 1)
geco_pre  = (-1, 0)
geco_post = (0.5, 1.5)
bef, afte = 2, 4

effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
# pyrUp_by = 'geco_zscore_amp'
# pyrUp_thresh = 0.08
# pyrDown_thresh = -pyrUp_thresh
pyrUp_by = 'geco_ratio'
pyrUp_thresh = 1.12
pyrDown_thresh = 1/pyrUp_thresh

thresh_baseline_dlight = 0.5
thresh_baseline_geco    = 1.0


regression_name = 'single_trial_regression_anat_roi'

OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
# OUT_DIR_FIG = (OUT_DIR_RAW_DATA/'TEST_PLOTS'/'window_test'/'session_pooled'/
#                 f'dlight_pre{dlight_pre}_post{dlight_post}_geco_pre{geco_pre}_post{geco_post}_manual_selec')
OUT_DIR_FIG = (OUT_DIR_RAW_DATA/'TEST_PLOTS'/f'geco_pre{geco_pre}_geco_post{geco_post}_wAC991-20250728-04_no_filter')

if not OUT_DIR_FIG.exists():
    OUT_DIR_FIG.mkdir(parents=True)
save_plot=0


# Load recording list
from dlight_imaging.geco_dlight.recording_list import rec_lst_dlight_geco as rec_lst
#%% collect roi profile for all sessions
df_pooled_profile = pd.DataFrame()
# rec_lst = ['AC991-20250728-04',]
for rec in rec_lst:
    print(f'loading: {rec}--------------------------------------------------')
    anm, date, ss = rec.split('-')
    
    # data paths
    p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
    p_regression = (OUR_DIR_REGRESS / rec / regression_name )
    p_masks      = OUR_DIR_REGRESS / rec / 'masks'
    p_suite2p_geco      = Path(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\nonrigid_reg_geco\suite2p_anat_detec\plane0")
    p_stats             = OUT_DIR_RAW_DATA / 'processed_dataframe' / f'{rec}_profile_stat_dlight_pre{dlight_pre}_post{dlight_post}.parquet'
    p_stats_geco        = OUT_DIR_RAW_DATA / 'processed_dataframe' / f'{rec}_profile_stat_geco_pre{geco_pre}_post{geco_post}.parquet'
    p_stats_geco_zscore = OUT_DIR_RAW_DATA / 'processed_dataframe_zscore' / f'{rec}_zscore_profile_stat_geco_pre{geco_pre}_post{geco_post}.parquet'
    
    # load dataframes
    dlight_stats = pd.read_parquet(p_stats)
    geco_zscore_stats = pd.read_parquet(p_stats_geco_zscore)
    geco_stats = pd.read_parquet(p_stats_geco)
    dlight_stats['rec_id'] = rec
    dlight_stats['mean_profile_geco'] = geco_stats['mean_profile']
    dlight_stats['mean_profile_geco_zscore'] = geco_zscore_stats['mean_profile']
    dlight_stats['geco_ratio'] = geco_stats['response_ratio']
    dlight_stats['geco_amp'] = geco_stats['response_amplitude']
    dlight_stats['geco_zscore_amp'] = geco_zscore_stats['response_amplitude']
    dlight_stats['mean_dlight'] = (dlight_stats['mean_profile'].apply(np.nanmean))
    dlight_stats['mean_geco']   = (dlight_stats['mean_profile_geco'].apply(np.nanmean))
    dlight_stats['mean_geco_zscore']   = (dlight_stats['mean_profile_geco_zscore'].apply(np.nanmean))
    
    # load is_soma index
    is_soma_idx        = np.load(OUR_DIR_REGRESS / rec / 'masks' / 'soma_class.npz')['is_soma']
    is_active_soma_idx = np.load(OUR_DIR_REGRESS / rec / 'masks' / 'soma_class.npz')['is_active_soma']
    dlight_stats['is_soma'] = is_soma_idx
    dlight_stats['is_active_soma'] = is_active_soma_idx
    
    # extract baseline info
    baseline_geco   = np.load(p_regression / 'baseline_geco_.npy')
    baseline_dlight = np.load(p_regression / 'baseline_corrected_dlight.npy')
    baseline_mean_geco = np.load(p_regression / f'baseline{geco_pre}_mean_geco.npy')
    baseline_std_geco  = np.load(p_regression / f'baseline{geco_pre}_std_geco.npy')
    baseline_mean_dlight = np.load(p_regression / f'baseline{dlight_pre}_mean_corrected_dlight.npy')
    baseline_std_dlight  = np.load(p_regression / f'baseline{dlight_pre}_std_corrected_dlight.npy')
    dlight_stats['baseline_geco_min'] = np.nanmin(baseline_geco, axis=-1)
    dlight_stats['baseline_dlight_min'] = np.nanmin(baseline_dlight, axis=-1)
    dlight_stats['dff_baseline_geco_mean'] = np.nanmean(baseline_geco, axis=-1)
    dlight_stats['dff_baseline_dlight_mean'] = np.nanmean(baseline_dlight, axis=-1)
    # dlight_stats['baseline_geco_max'] = np.nanmax(baseline_geco, axis=-1)
    # dlight_stats['baseline_dlight_max'] = np.nanmax(baseline_dlight, axis=-1)
    dlight_stats['run_baseline_geco_mean'] = baseline_mean_geco
    dlight_stats['run_baseline_geco_std'] = baseline_std_geco
    dlight_stats['run_baseline_dlight_mean'] = baseline_mean_dlight
    dlight_stats['run_baseline_dlight_std'] = baseline_std_dlight

    

    # count rois pixels
    geco_suite2p_stat = np.load(p_suite2p_geco / 'stat.npy', allow_pickle=True)
    dlight_mask = np.load(p_masks/'global_dlight_mask_enhanced.npy')
    roi_id = geco_stats['roi_id'].tolist()
    for roi in roi_id:
        ypix, xpix = geco_suite2p_stat[roi]['ypix'], geco_suite2p_stat[roi]['xpix']
        npix =  np.sum(dlight_mask[ypix, xpix])   
        dlight_stats.loc[dlight_stats['roi_id']==roi, 'n_pix'] = npix
        
    # identify valid ROIs based on baseline min
    dlight_stats['dlight_valid'] = (
        (dlight_stats['baseline_dlight_min'] > thresh_baseline_dlight) &
        dlight_stats['mean_profile'].apply(profile_is_valid)
        )
    dlight_stats['red_valid'] = (
        (dlight_stats['baseline_geco_min'] > thresh_baseline_geco) &
        dlight_stats['mean_profile_geco'].apply(profile_is_valid)
        )    
    dlight_stats['baseline_valid'] = ((thresh_baseline_geco<dlight_stats['baseline_geco_min'])
                                      &(thresh_baseline_dlight<dlight_stats['baseline_dlight_min'])
                                      )
    
    # assign Up and Down ROIs for dLight and GECO
    dlight_stats = classify_pyrs(dlight_stats, 
                      amp_shuff_thresh_up,
                      amp_shuff_thresh_down,
                      effect_size_thresh,
                      pyrUp_thresh,
                      pyrDown_thresh,
                      mean_thresh_geco=None,
                      mean_thresh_dlight=None,
                      geco_ratio=pyrUp_by
                    )
    
    # quantify change for non-DA-Up ROIs to evaluate session movement
    non_up_profile = np.nanmean(np.stack(dlight_stats.loc[(~dlight_stats['dlightUp'])&(dlight_stats['baseline_valid'])
                                                          &(dlight_stats['is_soma']), 'mean_profile']), axis=0)
    dlight_up_profile = np.nanmean(np.stack(dlight_stats.loc[(dlight_stats['dlightUp'])&(dlight_stats['baseline_valid'])
                                                          &(dlight_stats['is_soma']), 'mean_profile']), axis=0)
    dlight_stats['corr_non_vs_DA-Up'] = pearsonr(dlight_up_profile, non_up_profile)[0]
    non_up_amp_bef = np.nanmax(non_up_profile[int(30*(bef-0.5)):int(30*(bef+0.5))]) - np.nanmean(non_up_profile[int(30*(bef-1)):int(30*(bef-0.5))])
    non_up_amp_aft = np.nanmax(non_up_profile[int(30*(bef-0.5)):int(30*(bef+0.5))]) - np.nanmean(non_up_profile[int(30*(bef+0.5)):int(30*(bef+1))])
    non_up_amp_bef = non_up_amp_bef*100
    non_up_amp_aft = non_up_amp_aft*100
    dlight_stats['non_up_amp_bef'] = non_up_amp_bef
    dlight_stats['non_up_amp_aft'] = non_up_amp_aft
    dlight_stats['non_up_amp_max'] = np.max([non_up_amp_bef, non_up_amp_aft])
    
    
    # only save dataframe for active soma
    dlight_stats = dlight_stats.loc[dlight_stats['is_active_soma']]
    dlight_stats = dlight_stats.loc[dlight_stats['n_keep_trial']>80]
            
    # save per session dataframe
    p_df_out = OUT_DIR_RAW_DATA/'processed_dataframe' / rf"{rec}_profile_combined_geco_pre{geco_pre}_geco_post{geco_post}.parquet"
    # dlight_stats.to_parquet(p_df_out)
    
    geco_profile = np.vstack(dlight_stats['mean_profile_geco'])

    # nDA_pyrup_profile = np.nanmean(np.stack(dlight_stats.loc[(~dlight_stats['dlightUp'])&(dlight_stats['baseline_valid'])
    #                                                       &(dlight_stats['is_soma'])&(dlight_stats['geco_type']=='Up'), 'mean_profile']), axis=0)
    # DA_pyrup_profile = np.nanmean(np.stack(dlight_stats.loc[(dlight_stats['dlightUp'])&(dlight_stats['baseline_valid'])
    #                                                       &(dlight_stats['is_soma'])&(dlight_stats['geco_type']=='Up'), 'mean_profile']), axis=0)
    # nDA_pyrup_bef_run = np.nanmean(nDA_pyrup_profile[int(30*(bef-1)):int(30*(bef+0))]) 
    # DA_pyrup_bef_run = np.nanmean(DA_pyrup_profile[int(30*(bef-1)):int(30*(bef+0))]) 
    dlight_stats['dff_geco_run_base'] = np.nanmean(geco_profile[:, int(30*(bef-1)):int(30*(bef+0))], axis=1) 
    dlight_stats['dff_geco_full_trials'] = np.nanmean(geco_profile, axis=1) 

    
    baseline_window=dlight_pre 
    response_window=dlight_post,
    pre_event_window=2
    post_event_window=4
    imaging_rate = 30
    
    p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
    beh = pd.read_pickle(p_beh_file)
    run_onset_frames = np.array(beh['run_onset_frames'])
    valid_trials = (seperate_valid_trial(beh))&(run_onset_frames!=-1)
    run_onset_frames_valid = run_onset_frames[valid_trials]
    # ---------- build baseline segments ----------
    baseline_frames = (int(baseline_window[0] * imaging_rate),
                       int(baseline_window[1] * imaging_rate))

    trial_aligned_traces = align_trials(
        baseline_geco,
        event_frames=run_onset_frames_valid,
        bef=pre_event_window,
        aft=post_event_window,
        gpu=0
    )  # expected: cupy array [n_rois, n_trials, trial_len]
    pre_event_frames = int(pre_event_window * imaging_rate)
    baseline_start = pre_event_frames + baseline_frames[0]
    baseline_end   = pre_event_frames + baseline_frames[1]
    baseline_segment_all = trial_aligned_traces[:, :, baseline_start:baseline_end]
    baseline_segment_mean = np.nanmean(baseline_segment_all, axis=(1,2))
    dlight_stats['dff_baseline_geco_bef_run'] = baseline_segment_mean[np.stack(dlight_stats['roi_id'])]
    
    p_suite2p_geco = Path(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\nonrigid_reg_geco\suite2p_anat_detec\plane0")
    geco_trace = np.load(p_suite2p_geco / 'F.npy')
    geco_trace_neu = np.load(p_suite2p_geco / 'Fneu.npy')
    # correct calcium trace with neuropil signal
    geco_trace_corr = geco_trace - 0.7*geco_trace_neu
    dlight_stats['geco_Fc_mean'] = np.nanmean(geco_trace_corr, axis=-1)[np.stack(dlight_stats['roi_id'])]
    # add to dataframe pool
    df_pooled_profile = pd.concat((df_pooled_profile, dlight_stats))

# save pooled dataframes
p_pooled_df = OUT_DIR_RAW_DATA / 'processed_dataframe'/ rf"df_population_profile_pooled_pre{dlight_pre}_post{dlight_post}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.parquet"
# df_pooled_profile.to_parquet(p_pooled_df)

#%%
# df_pool_sorted = df_pooled_profile.loc[df_pooled_profile['corr_non_vs_DA-Up']<0.85]
df_pool_sorted = df_pooled_profile.loc[df_pooled_profile['non_up_amp_bef']<2.1]
df_pool_sorted = df_pool_sorted.loc[df_pool_sorted['baseline_valid']]    

#%% reassign Up and Down using chosen thresholding
df_pool_dlight_up = df_pool_sorted.loc[df_pool_sorted['dlightUp']]
df_pool_non_dlight_up = df_pool_sorted.loc[~df_pool_sorted['dlightUp']]

#%% test plotting
import matplotlib.pyplot as plt
from common import plotting_functions_Jingyu as pf
DA_pyrup_mean = np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                         (df_pool_sorted['geco_type']=='Up'),'dff_baseline_geco_mean'])
DA_pyrup_std = np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                         (df_pool_sorted['geco_type']=='Up'),'run_baseline_geco_std'])
fig, ax = plt.subplots()
ax.hist(DA_pyrup_mean, range=(0,60), bins=100, color = 'green', alpha=.7)    
ax.set(title='DA-Up PyrUp dF/F baseline mean', ylabel='roi count')
fig, ax = plt.subplots()
ax.hist(DA_pyrup_std, range=(0,20), bins=100, color = 'orange', alpha=.7)    
ax.set(title='DA-Up PyrUp baseline std', ylabel='roi count')

nDA_pyrup_mean = np.stack(df_pool_sorted.loc[~(df_pool_sorted['dlight_type']=='Up')&
                                         (df_pool_sorted['geco_type']=='Up'),'dff_baseline_geco_mean'])
nDA_pyrup_std = np.stack(df_pool_sorted.loc[~(df_pool_sorted['dlight_type']=='Up')&
                                         (df_pool_sorted['geco_type']=='Up'),'run_baseline_geco_std'])
fig, ax = plt.subplots()
ax.hist(nDA_pyrup_mean, range=(0,60), bins=100, color = 'green', alpha=.4)    
ax.set(title='non_DA-Up PyrUp dF/F baseline mean', ylabel='roi count')
fig, ax = plt.subplots()
ax.hist(nDA_pyrup_std, range=(0,20), bins=100, color = 'orange', alpha=.4)   
ax.set(title='non_DA-Up PyrUp baseline std', ylabel='roi count') 

DA_pyrdown_mean = np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                         (df_pool_sorted['geco_type']=='Down'),'dff_baseline_geco_mean'])
DA_pyrdown_std = np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                         (df_pool_sorted['geco_type']=='Down'),'run_baseline_geco_std'])
fig, ax = plt.subplots()
ax.hist(DA_pyrdown_mean, range=(0,60), bins=100, color = 'green', alpha=.7)    
ax.set(title='DA-Up PyrDown dF/F baseline mean', ylabel='roi count')
fig, ax = plt.subplots()
ax.hist(DA_pyrdown_std, range=(0,20), bins=100, color = 'orange', alpha=.7)    
ax.set(title='DA-Up PyrDown baseline std', ylabel='roi count')

nDA_pyrdown_mean = np.stack(df_pool_sorted.loc[~(df_pool_sorted['dlight_type']=='Up')&
                                         (df_pool_sorted['geco_type']=='Down'),'dff_baseline_geco_mean'])
nDA_pyrdown_std = np.stack(df_pool_sorted.loc[~(df_pool_sorted['dlight_type']=='Up')&
                                         (df_pool_sorted['geco_type']=='Down'),'run_baseline_geco_std'])
fig, ax = plt.subplots()
ax.hist(nDA_pyrdown_mean, range=(0,60), bins=100, color = 'green', alpha=.4)    
ax.set(title='non_DA-Up PyrDown dF/F baseline mean', ylabel='roi count')
fig, ax = plt.subplots()
ax.hist(nDA_pyrdown_std, range=(0,20), bins=100, color = 'orange', alpha=.4)   
ax.set(title='non_DA-Up PyrDown baseline std', ylabel='roi count') 

#%% quantify baseline difference between DA-Up and non-DA-Up group
def _per_session_baseline_mean(df, dlight_is_up, geco_type,
                               metric='dff_baseline_geco_mean'):
    sub = df.loc[((df['dlight_type'] == 'Up') == dlight_is_up) &
                 (df['geco_type'] == geco_type)]
    return sub.groupby('rec_id')[metric].mean()


def compare_DA_vs_nonDA(df, metric, ylabel, title_suffix=''):
    """For a given metric column, plot DA-Up vs non-DA-Up comparisons
    (single-cell unpaired violin + per-session paired bar/scatter) for both
    PyrUp and PyrDown groups."""
    suffix = f' ({title_suffix})' if title_suffix else ''
    groups = [
        ('Up',   'PyrUp',   ('lightgreen', 'green')),
        ('Down', 'PyrDown', ('violet',     'purple')),
    ]
    for geco_type, label, colors in groups:
        DA_vals  = df.loc[(df['dlight_type']=='Up') &
                          (df['geco_type']==geco_type), metric].values
        nDA_vals = df.loc[~(df['dlight_type']=='Up') &
                          (df['geco_type']==geco_type), metric].values

        # single cell (unpaired)
        fig, ax = plt.subplots(dpi=300, figsize=(2, 3))
        pf.plot_unpaired_violin(
            nDA_vals, DA_vals,
            colors=list(colors),
            colname=['non-DA-Up', 'DA-Up'],
            ylabel=ylabel,
            title_prefix=f'{label} single-cell{suffix}',
            ax=ax)

        # per session (paired by rec_id)
        s_nDA = _per_session_baseline_mean(df, False, geco_type, metric=metric)
        s_DA  = _per_session_baseline_mean(df, True,  geco_type, metric=metric)
        recs  = s_nDA.index.intersection(s_DA.index)
        fig, ax = plt.subplots(dpi=300, figsize=(2, 3))
        pf.plot_bar_with_paired_scatter(
            ax, s_nDA.loc[recs].values, s_DA.loc[recs].values,
            colors=colors,
            xticklabels=('non-DA-Up', 'DA-Up'),
            ylabel=ylabel,
            title=f'{label} per-session{suffix}')


metrics_to_compare = [
    ('dff_baseline_geco_mean',    'dFF baseline',               ''),
    ('geco_Fc_mean', 'Fc_mean', ''),
    # ('run_baseline_geco_mean',    'run-onset baseline (geco)',   'run baseline'),
    # ('dff_geco_run_base',         'dff geco run-onset baseline', 'dff run baseline'),
    # ('dff_baseline_geco_bef_run', 'dff baseline bef run',        'bef-run baseline'),
    ('dff_geco_full_trials',      'dff full trials',             'full trial mean dff'),
]
for metric, ylabel, title_suffix in metrics_to_compare:
    compare_DA_vs_nonDA(df_pool_sorted, metric, ylabel, title_suffix)
