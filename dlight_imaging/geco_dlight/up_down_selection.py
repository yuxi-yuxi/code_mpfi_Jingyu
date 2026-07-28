# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:32:56 2026

@author: Jingyu Cao
"""

import numpy as np
# import pandas as pd
# from pathlib import Path
# from scipy.stats import pearsonr

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