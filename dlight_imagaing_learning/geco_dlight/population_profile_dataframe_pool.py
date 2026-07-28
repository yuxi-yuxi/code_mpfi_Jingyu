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

def profile_is_valid(x):
    if x is None:
        return False
    a = np.asarray(x)
    if a.size == 0:
        return False
    if np.max(a)>10:
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


OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\geco_dlight")
OUT_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
# OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'TEST_PLOTS' / 'regression_res'
# if not OUT_DIR_FIG.exists():
#     OUT_DIR_FIG.mkdir(parents=True)
# save_plot=0
regression_name ='single_trial_regression_anat_roi'
OUT_DIR = OUT_DIR_RAW_DATA / 'processed_dataframe'

# p_session_info = OUT_DIR_RAW_DATA / 'all_animals_learning_classification.parquet'
# df_all = pd.read_parquet(p_session_info)
# rec_lst = df_all.loc[(df_all['days_from_learned']<=2)&
#                      (df_all['animal']=='AC953'), 'rec'].to_list()



#%% collect roi profile for all sessions
df_pooled_profile = pd.DataFrame()
rec_lst = [
'AC330-20260602-02',
'AC330-20260603-02', 
'AC330-20260604-02', 
'AC330-20260605-02', 
'AC330-20260606-02', 
'AC330-20260607-02', 
'AC330-20260608-02',
'AC330-20260609-02', 
'AC330-20260610-02',          
'AC330-20260611-02',          
'AC330-20260612-02',      
    
'AC327-20260602-02',     
'AC327-20260603-02',     
'AC327-20260604-02',     
'AC327-20260605-02',     
'AC327-20260606-02',     
'AC327-20260607-02',     
'AC327-20260608-02',     
'AC327-20260609-02',     
'AC327-20260610-02',     
'AC327-20260611-02',
'AC327-20260612-02',      
    ]
for rec in rec_lst:
    print(f'loading: {rec}--------------------------------------------------')
    anm, date, ss = rec.split('-')
    
    # data paths
    p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
    p_regression = (OUT_DIR_REGRESS / rec / regression_name )
    p_masks      = OUT_DIR_REGRESS / rec / 'masks'
    p_suite2p_geco      = Path(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\anat_detect\suite2p\plane0")
    # if not p_suite2p_geco.exists():
    #     p_suite2p_geco  = Path(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\dLight+GECO\GECO")
    p_stats             = OUT_DIR_RAW_DATA / 'processed_dataframe' / f'{rec}_profile_stat_dlight_pre{dlight_pre}_post{dlight_post}.parquet'
    p_stats_geco        = OUT_DIR_RAW_DATA / 'processed_dataframe' / f'{rec}_profile_stat_geco_pre{geco_pre}_post{geco_post}.parquet'
    p_stats_geco_zscore = OUT_DIR_RAW_DATA / 'processed_dataframe_zscore' / f'{rec}_zscore_profile_stat_geco_pre{geco_pre}_post{geco_post}.parquet'
    
    # load dataframes
    dlight_stats = pd.read_parquet(p_stats)
    # geco_zscore_stats = pd.read_parquet(p_stats_geco_zscore)
    geco_stats = pd.read_parquet(p_stats_geco)
    dlight_stats['rec_id'] = rec
    dlight_stats['mean_profile_geco'] = geco_stats['mean_profile']
    # dlight_stats['mean_profile_geco_zscore'] = geco_zscore_stats['mean_profile']
    dlight_stats['geco_ratio'] = geco_stats['response_ratio']
    dlight_stats['geco_amp'] = geco_stats['response_amplitude']
    # dlight_stats['geco_zscore_amp'] = geco_zscore_stats['response_amplitude']
    dlight_stats['mean_dlight'] = (dlight_stats['mean_profile'].apply(np.nanmean))
    dlight_stats['mean_geco']   = (dlight_stats['mean_profile_geco'].apply(np.nanmean))
    # dlight_stats['mean_geco_zscore']   = (dlight_stats['mean_profile_geco_zscore'].apply(np.nanmean))
    
    # load is_soma index
    is_soma_idx        = np.load(OUT_DIR_REGRESS / rec / 'masks' / 'soma_class.npz')['is_soma']
    is_active_soma_idx = np.load(OUT_DIR_REGRESS / rec / 'masks' / 'soma_class.npz')['is_active_soma']
    dlight_stats['is_soma'] = is_soma_idx
    dlight_stats['is_active_soma'] = is_active_soma_idx
    
    # extract baseline info
    baseline_geco   = np.load(p_regression / 'baseline_geco.npy')
    baseline_dlight = np.load(p_regression / 'baseline_corrected_dlight.npy')
    dlight_stats['baseline_geco_min'] = np.nanmin(baseline_geco, axis=-1)
    dlight_stats['baseline_dlight_min'] = np.nanmin(baseline_dlight, axis=-1)
    # dlight_stats['baseline_geco_median'] = np.nanmedian(baseline_geco, axis=-1)
    # dlight_stats['baseline_dlight_median'] = np.nanmedian(baseline_dlight, axis=-1)
    # dlight_stats['baseline_geco_max'] = np.nanmax(baseline_geco, axis=-1)
    # dlight_stats['baseline_dlight_max'] = np.nanmax(baseline_dlight, axis=-1)

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
    # dlight_stats = dlight_stats.loc[dlight_stats['is_active_soma']]
    # dlight_stats = dlight_stats.loc[dlight_stats['n_keep_trial']>80]
            
    # save per session dataframe
    p_df_out = OUT_DIR_RAW_DATA/'processed_dataframe' / rf"{rec}_profile_combined_geco_pre{geco_pre}_geco_post{geco_post}.parquet"
    dlight_stats.to_parquet(p_df_out)
    
    # add to dataframe pool
    df_pooled_profile = pd.concat((df_pooled_profile, dlight_stats))

# save pooled dataframes
p_pooled_df = OUT_DIR_RAW_DATA / 'processed_dataframe'/ rf"df_population_profile_pooled_pre{dlight_pre}_post{dlight_post}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.parquet"
df_pooled_profile.to_parquet(p_pooled_df)
    

    

