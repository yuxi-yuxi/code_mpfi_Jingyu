# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 16:41:33 2026

@author: Jingyu Cao

Commonly used fucntion for drug infusion analysis
"""
#%%
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import center_of_mass, gaussian_filter1d

sys.path.extend(['Z:/Jingyu/code_mpfi_Jingyu/drug_infusion',
                 'Z:/Jingyu/code_mpfi_Jingyu/common',
                 'Z:/Jingyu/code_mpfi_Jingyu/drd1_detection'])
from utils_basic import nd_to_list, trace_filter, normalize
from utils_imaging import align_trials
from trial_selection import select_good_trials

OUT_DIR = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\processed_dataframe"
PATH_GCAMP_PROFILE = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\gcamp_profile"
BEHAVIOR_DATA_DIR = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\behaviour_profile"
#%%
def load_profile(rec):
    anm_id = rec['anm']
    date = rec['date']
    df_gcamp_profile = pd.read_parquet(
        PATH_GCAMP_PROFILE+ f"\{anm_id}_{date}_df_gcamp_profile_20perc_dff.parquet")
    return df_gcamp_profile

def load_behaviour(rec, s):
    if s == 'ss1':
        ss='02'
    elif s == 'ss2':
        ss='04'
    p_beh = BEHAVIOR_DATA_DIR+r'\{}.pkl'.format(rec['anm']+'-'+rec['date']+'-'+ ss)
    beh=pd.read_pickle(p_beh)
    return beh

def _filter(df_cal_profile_all, trace_type, s):
    df_cal_profile_all[f"full_dff_trace_{trace_type}_{s}"] = \
    df_cal_profile_all[f"full_dff_trace_{trace_type}_{s}"].apply(lambda x:
                                                                 trace_filter(x, 99)
                                                                )
    return df_cal_profile_all
    
def calculate_ratio(df_cal_profile_all, s, session_info, trace_type, beh,
                    selected_trials=None,
                    trial_bef=2, trial_aft=4, pre_window=(-1, 0), post_window=(0, 1.5)):
    
    epsilon = 1e-9 # to prevent divide by zero
    
    pre_start = int(trial_bef*30+pre_window[0]*30)
    pre_end = int(trial_bef*30+pre_window[1]*30)
    post_start = int(trial_bef*30+post_window[0]*30)
    post_end = int(trial_bef*30+post_window[1]*30)
    
    valid_trials = session_info[f'valid_trials_{s}']
    
    df_cal_profile_all = _filter(df_cal_profile_all, trace_type, s)
    
    df_cal_profile_all[f"run_cal_profile_all_{trace_type}_{s}"]=''
    df_cal_profile_all[f"run_cal_profile_all_{trace_type}_{s}"] = df_cal_profile_all[f"full_dff_trace_{trace_type}_{s}"].apply(lambda x: 
                                                                                                                  align_trials(
                                                                                                                  x, 
                                                                                                                  alignment='run', 
                                                                                                                  beh=beh, bef=trial_bef, aft=trial_aft))
    # df_cal_profile_all[f"run_cal_profile_all_{trace_type}_{s}"] = df_cal_profile_all[f"run_cal_profile_all_{trace_type}_{s}"].apply(lambda x:
    #                                                                                                           nd_to_list(x)                      
    #                                                                                                           )
        
    df_cal_profile_all[f"run_cal_profile_{trace_type}_{s}"] = \
    df_cal_profile_all[f"run_cal_profile_all_{trace_type}_{s}"].apply(lambda x: np.nanmean(np.stack(x)[valid_trials], axis=0))
    # df_cal_profile_all[f'active_trials_{s}'] = df_cal_profile_all[f"run_cal_profile_all_{trace_type}_{s}"].apply(lambda x: detect_active_trial(np.stack(x)))
    
    if trace_type == 'raw':
        # normalize traces for ratio calculation
        # df_cal_profile_all[f"run_cal_profile_nr_{trace_type}_{s}"] = \
        # df_cal_profile_all[f"run_cal_profile_{trace_type}_{s}"].apply(lambda x: normalize(x))
        # use smoothed mean trace for ratio calculation
        df_cal_profile_all[f"run_ratio_{trace_type}_{s}"] = \
        df_cal_profile_all[f"run_cal_profile_{trace_type}_{s}"].apply(lambda x: np.nanmean(gaussian_filter1d(x, 1)[post_start:post_end], axis=-1)/\
                                                                     (np.nanmean(gaussian_filter1d(x, 1)[pre_start:pre_end], axis=-1)))
    elif trace_type == 'rsd':
        # use smoothed mean trace for ratio calculation
        df_cal_profile_all[f"run_ratio_{trace_type}_{s}"] = \
        df_cal_profile_all[f"run_cal_profile_{trace_type}_{s}"].apply(lambda x: np.nanmean(gaussian_filter1d(x, 1)[post_start:post_end], axis=-1)/\
                                                                     (np.nanmean(gaussian_filter1d(x, 1)[pre_start:pre_end], axis=-1)+epsilon)) 
    if selected_trials is not None:
         good_trials = selected_trials
         df_cal_profile_all[f"run_cal_profile_{trace_type}_good_{s}"] = \
         df_cal_profile_all[f"run_cal_profile_all_{trace_type}_{s}"].apply(lambda x: np.nanmean(np.stack(x)[good_trials], axis=0))
         
         if trace_type == 'raw':
             # normalize traces for ratio calculation
             # df_cal_profile_all[f"run_cal_profile_nr_{trace_type}_good_{s}"] = \
             # df_cal_profile_all[f"run_cal_profile_{trace_type}_good_{s}"].apply(lambda x: normalize(x))
             # use smoothed mean trace for ratio calculation
             df_cal_profile_all[f"run_ratio_{trace_type}_good_{s}"] = \
             df_cal_profile_all[f"run_cal_profile_{trace_type}_good_{s}"].apply(lambda x: np.nanmean(gaussian_filter1d(x, 1)[post_start:post_end], axis=-1)/\
                                                                               (np.nanmean(gaussian_filter1d(x, 1)[pre_start:pre_end], axis=-1)))   
         elif trace_type == 'rsd':
             # use smoothed mean trace for ratio calculation
             df_cal_profile_all[f"run_ratio_{trace_type}_good_{s}"] = \
             df_cal_profile_all[f"run_cal_profile_{trace_type}_good_{s}"].apply(lambda x: np.nanmean(gaussian_filter1d(x, 1)[post_start:post_end], axis=-1)/\
                                                                           (np.nanmean(gaussian_filter1d(x, 1)[pre_start:pre_end], axis=-1)+epsilon))   

                                                                                        
    return df_cal_profile_all 


def extract_session_info(rec, trial_type,
                         df_session_info, ##added
                         trial_bef=2, trial_aft=4, pre_window=(-1, 0), post_window=(0, 1.5),):
    anm_id = rec['anm']
    ss = rec['session']
    n_sessions = len(ss)
    sessions = ['ss1', 'ss2']
    date = rec['date']
    rec_id = anm_id+'-'+date
    session_label = rec['label']
    session_info = df_session_info.loc[rec_id]

    INT_PATH = r"Z:\Jingyu\2P_Recording\{}\{}\concat".format(anm_id, anm_id+'-'+date)
    
    print('{}_{}_loading...--------------------------'.format(anm_id, date))
    
    p_df_out = OUT_DIR+r"\{}_{}_df_cal_profile_all_to_pool_valid_trials_pre{}_post{}.parquet".format(anm_id, date, pre_window, post_window)
    if not os.path.exists(p_df_out):
        extract_info = 1
    else:
        df_out = pd.read_parquet(p_df_out)
        
    if extract_info:
        df_cal_profile_all = pd.read_parquet(
            PATH_GCAMP_PROFILE+r"\{}_{}_df_cal_profile_all_valid_trials.parquet".format(anm_id, date))
    
    if extract_info:
        # calcultate run_response widnow by a defined window
        for i, s in enumerate(sessions):
            p_beh = r"Z:\Jingyu\Code\Python\2p_SCH23390_infusion\behaviour_profile\{}.pkl".format(rec['anm']+
                                                                                                  '-'+rec['date']+
                                                                                                  '-'+rec['session'][i])
            beh=pd.read_pickle(p_beh)
            selected_trials = select_good_trials(beh)
            selected_trials[:10] = 0 # exclude first 10 trials
            df_cal_profile_all[f'n_keep_trials_{s}'] = np.sum(selected_trials)
            
            if trial_type == r'_good_trials':
                selected_trials = selected_trials
            else:
                selected_trials = None
                
            df_cal_profile_all[f'baseline_mean_{s}'] = df_cal_profile_all[f"dff_baseline_{s}"].apply(lambda x: np.nanmean(x))
            df_cal_profile_all = calculate_ratio(df_cal_profile_all, s, session_info, trace_type='rsd', beh=beh,
                                                 selected_trials=selected_trials,
                                                 trial_bef=trial_bef, trial_aft=trial_aft, pre_window=pre_window, post_window=post_window) 
        
        A_master = xr.open_dataarray(INT_PATH + r"\A_master.nc").rename(
            {"master_uid": "unit_id"}
        )
        gcamp_rois = np.array(A_master).squeeze()
    
        p_suite2p = r"Z:\Jingyu\2P_Recording\{}\{}\02\suite2p\plane0".format(rec['anm'], rec_id)
        suite2p_ops = np.load(p_suite2p+r'\ops.npy', allow_pickle=True).item()
        ref_mean = suite2p_ops['meanImg_chan2_corrected']
        df_cal_profile_all['roi_com'] = df_cal_profile_all['map'].apply(lambda x: center_of_mass(np.stack(x)))
        df_cal_profile_all['roi_pix_ch2'] = df_cal_profile_all['map'].apply(lambda x: ref_mean*(np.where(np.stack(x)>0, 1, 0)))
        df_cal_profile_all['roi_pix_ch2_mean'] = df_cal_profile_all['roi_pix_ch2'].apply(lambda x: np.sum(x)/np.sum(x>0))
        df_cal_profile_all['roi_pix_ch2_com'] = df_cal_profile_all['roi_com'].apply(lambda x: ref_mean[int(x[0]), int(x[1])])
        
        # all cells are baseon Drd1 detection
        # thresh_ch2 = np.nanpercentile(df_cal_profile_all['roi_pix_ch2_mean'], 85)
        # thresh_ch2 = np.nanpercentile(ref_mean, 90)
        # df_cal_profile_all.loc[:, 'drd1+'] = np.where((df_cal_profile_all['npix']>100),
        #                                               # (df_cal_profile_all['roi_pix_ch2_com']>thresh_ch2),
        #                                               True, False)
        
        # df_cal_profile_all.loc[:, 'drd1_bright'] = np.where((df_cal_profile_all['npix']>100)&
        #                                               (df_cal_profile_all['roi_pix_ch2_com']>thresh_ch2),
        #                                               True, False)
        
        matches, unmatched_A, unmatched_B, scores = drd1_cell_match(ref_mean, gcamp_rois)
        drd1_gcamp = [match[1] for match in matches]

        # thresh_ch2 = np.nanpercentile(df_cal_profile_all['roi_pix_ch2_mean'], 95)
        thresh_ch2 = np.nanpercentile(ref_mean, 90)
        for roi_idx, df_roi in df_cal_profile_all.iterrows():
            if df_roi['unit_id'] in drd1_gcamp and (df_roi['roi_pix_ch2_com']>thresh_ch2):
                df_cal_profile_all.loc[roi_idx, 'drd1+'] = True
            else:
                df_cal_profile_all.loc[roi_idx, 'drd1+'] = False
        
        # df_cal_profile_all['ch2+'] = np.where((df_cal_profile_all['roi_pix_ch2_mean']>thresh_ch2)&
        #                                       (df_cal_profile_all.loc[roi_idx, 'drd1+'] == False),
        #                                       True,
        #                                       False)
        # # validate Drd1+ selection
        # drd1_cells = df_cal_profile_all.loc[(df_cal_profile_all['drd1+'])&(df_cal_profile_all['active_soma']), 'unit_id']
        # roi_map = np.zeros((512, 512))
        # for roi in drd1_cells:
        #     roi_map[gcamp_rois[roi]>0] = 1

        # roi_map = np.where(roi_map>0, 1, np.nan)
        # fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
        # ax.imshow(ref_mean,
        #           vmin = np.nanpercentile(ref_mean, 80),
        #           vmax = np.nanpercentile(ref_mean, 95),
        #           cmap = 'gray'
        #           )
        # ax.imshow(roi_map, cmap='Set1', alpha=.3)
        # ax.set_axis_off()
        # fig.tight_layout()
        # plt.savefig(INT_PATH+r'\false_drd1_active_soma_detection.png')
        # # plt.show()
        # plt.close()
        
        # 1. Extract the metadata row as a DataFrame, and repeat it to match df_cal_pyr
        # meta = rec_lst.loc[anm_id + '-' + date, ['anm', 'date', 'session', 'label', 'conc', 'latency']]
        meta = rec.loc[['anm', 'date', 'session', 'label', 'conc', 'latency']] 
        meta_df = pd.DataFrame([meta] * len(df_cal_profile_all))  # repeat for each row
        
        # 2. Reset index of df_cal_pyr to align
        df_cal_profile_all = df_cal_profile_all.reset_index(drop=True)
        meta_df = meta_df.reset_index(drop=True)
        
        # 3. Concatenate column-wise
        df_to_pool = pd.concat([meta_df, df_cal_profile_all[[
            'unit_id',
            'active_soma',
            # 'ch2+',
            'baseline_mean_ss1',
            'baseline_mean_ss2',
            'drd1+',
            'n_keep_trials_ss1',
            'n_keep_trials_ss2', 
            # 'drd1_bright',
            'active_trials_ss1',
            'active_trials_ss2',
            'run_ratio_rsd_ss1',
            'run_ratio_rsd_ss2',
            'run_ratio_rsd_good_ss1',
            'run_ratio_rsd_good_ss2',
            'run_tunned_higher_ss1',
            'run_tunned_higher_ss2',
            'run_tunned_lower_ss1',
            'run_tunned_lower_ss2',
            
            'full_dff_trace_rsd_ss1',
            'full_dff_trace_rsd_ss2',
            
            'run_cal_profile_rsd_ss1',
            'run_cal_profile_rsd_ss2',
            'run_cal_profile_rsd_good_ss1',
            'run_cal_profile_rsd_good_ss2',
            # 'active_ss1',
            # 'active_ss2'
        ]]], axis=1)
        
        df_out.to_parquet(p_df_out)

    return df_out

def sort_response(df_profile, thresh_up, thresh_down,
                  ratio_type='',
                  # trace_type=''
                  ):
    
    df_profile = df_profile.copy()
    # df_profile['gcamp_valid_ss1'] = df_profile[f'{trace_type}_ss1'].apply(lambda x: False if x is None else (np.all(np.abs(x) < 1, axis=-1)))
    # df_profile['gcamp_valid_ss2'] = df_profile[f'{trace_type}_ss2'].apply(lambda x: False if x is None else (np.all(np.abs(x) < 1, axis=-1)))
    # df_profile['gcamp_valid_ss1'] = df_profile[f'{trace_type}_ss1'].apply(lambda x: False if x is None else (not np.any(np.isnan(x))) and np.all(np.abs(x) < 1))
    # df_profile['gcamp_valid_ss2'] = df_profile[f'{trace_type}_ss2'].apply(lambda x: False if x is None else (not np.any(np.isnan(x))) and np.all(np.abs(x) < 1))
    # df_profile['mean_dff_ss1'] =  df_profile[f'{trace_type}_ss1'].apply(lambda x: False if x is None else (np.nanmean(x)))
    # df_profile['mean_dff_ss2'] =  df_profile[f'{trace_type}_ss2'].apply(lambda x: False if x is None else (np.nanmean(x)))
    
    # df_profile['gcamp_valid_ss1'] = df_profile['mean_dff_ss1'].apply(lambda x: 0<x<1)
    # df_profile['gcamp_valid_ss2'] = df_profile['mean_dff_ss2'].apply(lambda x: 0<x<1)
    # df_profile['gcamp_valid_ss1'] = df_profile['dff_baseline_min_ss1']>1
    # df_profile['gcamp_valid_ss2'] = df_profile['dff_baseline_min_ss1']>1
    
    # df_profile = df_profile.loc[(df_profile['gcamp_valid_ss1']) & (df_profile['gcamp_valid_ss2'])
    #                             # &(df_profile['gcamp_valid_f_all_median'])
    #                             ]
    
    for i in ['ss1', 'ss2']:
        
        run_ratio = f'{ratio_type}_{i}'
    
        df_profile.loc[:, f'pyrUp_{i}'] = np.where(
                                            (df_profile[run_ratio]>thresh_up),
                                            # (df_profile[f'gcamp_valid']),
                                            # &(df_profile[f'run_tunned_higher_{i}']),
                                            True,
                                            False
                                            )
        
        df_profile.loc[:, f'pyrDown_{i}'] = np.where(
                                            (df_profile[run_ratio]<thresh_down),
                                            # (df_profile[f'gcamp_valid']),
                                            # &(df_profile[f'run_tunned_lower_{i}']),
                                            True,
                                            False
                                            )
        
        df_profile.loc[:, f'pyrStab_{i}'] = np.where(
                                            (df_profile[run_ratio]<thresh_up)&
                                            (df_profile[run_ratio]>thresh_down),
                                            # (df_profile[f'gcamp_valid']),
                                            # (df_profile[f'active_{i}']),
                                            True,
                                            False
                                            )        
    return df_profile

#%% 
# if __name__ == '__main__':
    
