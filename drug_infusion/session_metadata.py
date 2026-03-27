# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 12:36:15 2026

@author: Jingyu Cao

Session Metadata Management for Drug Infusion Experiments

This module builds and maintains a metadata table (parquet file) containing
session-level information for all drug infusion imaging experiments. For each
recording session, it extracts trial validity information from behavioral data
and stores experimental parameters (drug concentration, latency, etc.).

Output:
    infusion_session_info.parquet - DataFrame with columns:
        - anm: Animal ID (e.g., 'AC986')
        - date: Recording date (YYYYMMDD format)
        - session: List of session numbers (e.g., ['02', '04'])
        - label: Condition labels (e.g., ['baseline', 'SCH'])
        - conc: Drug concentrations in μM
        - latency: Wait time between sessions (minutes)
        - valid_trials_ss1/ss2/ss3: Boolean arrays of valid trials per session
        - n_valid_trials_ss1/ss2/ss3: Count of valid trials
        - perc_valid_trials_ss1/ss2/ss3: Percentage of valid trials

Usage:
    Run as script to process all sessions in rec_lst_infusion:
        python session_metadata.py

    Or import and use the function:
        from session_metadata import update_session_info
        df = update_session_info(df_session_info, rec)
"""
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add script directory to path for local imports
if (r"Z:\Jingyu\code_mpfi_Jingyu\drug_infusion" in sys.path) == False:
    sys.path.append(r"Z:\Jingyu\code_mpfi_Jingyu\drug_infusion")
from rec_lst_infusion import rec_lst_infusion
from common.utils_basic import zero_padding
from common.trial_selection import select_good_trials
# =============================================================================
# Constants
# =============================================================================
BEHAVIOR_DATA_DIR = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\behaviour_profile"
OUTPUT_DIR = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion"

# =============================================================================
# Functions
# =============================================================================
def update_session_info(df_session_info, rec, exp='GCaMP8s_infusion'):
    """
    Add or update a single recording session's metadata in the session info table.

    For each session, loads the behavioral data and determines which trials are
    valid based on: (1) reward delivery, (2) no mid-trial stops, (3) no full stops.

    Parameters
    ----------
    df_session_info : pd.DataFrame or list
        Existing session info table, or empty list to create new one.
    rec : dict
        Recording metadata with keys:
        - 'anm': Animal ID (str)
        - 'date': Recording date as YYYYMMDD (str)
        - 'session': List of session numbers (list of str)
        - 'label': Condition labels (list of str)
        - 'conc': Drug concentrations in μM (list, optional)
        - 'latency': Wait time in minutes (int, optional)
    exp : str, optional
        Experiment name (default: 'GCaMP8s_infusion').

    Returns
    -------
    pd.DataFrame
        Updated session info table with new row added (if not already present).

    Notes
    -----
    Valid trials are defined as trials where:
    - Reward was delivered (~np.isnan(reward_times))
    - Animal did not stop mid-trial (non_stop_trials == 0)
    - Animal did not fully stop (non_fullstop_trials == 0)
    """
    # Initialize DataFrame if needed
    if not isinstance(df_session_info, pd.DataFrame) or df_session_info.empty:
        df_session_info = pd.DataFrame(columns=[
            'anm', 'date', 'session', 'label', 'conc', 'latency',
            'valid_trials_ss1', 'valid_trials_ss2', 'valid_trials_ss3',
            'lick_idx_ss1', 'lick_idx_ss2', 'lick_idx_ss3',
            'max_speed_median_ss1', 'max_speed_median_ss2', 'max_speed_median_ss3',
            'full_trial_len_median_ss1', 'full_trial_len_median_ss2', 'full_trial_len_median_ss3',
            'good_trials_ss1', 'good_trials_ss2', 'good_trials_ss3',
            'bad_trials_ss1', 'bad_trials_ss2', 'bad_trials_ss3',
            'n_valid_trials_ss1', 'n_valid_trials_ss2', 'n_valid_trials_ss3',
            'perc_valid_trials_ss1', 'perc_valid_trials_ss2', 'perc_valid_trials_ss3',
            'n_good_trials_ss1', 'n_good_trials_ss2', 'n_good_trials_ss3',
            'perc_good_trials_ss1', 'perc_good_trials_ss2', 'perc_good_trials_ss3',
            'n_bad_trials_ss1', 'n_bad_trials_ss2', 'n_bad_trials_ss3'
        ])

    # Ensure object dtype for columns that hold lists/arrays
    for col in ['label', 'session',
                'good_trials_ss1', 'good_trials_ss2', 'good_trials_ss3',
                'bad_trials_ss1', 'bad_trials_ss2', 'bad_trials_ss3']:
        if col in df_session_info.columns:
            df_session_info[col] = df_session_info[col].astype('object')

    # Extract recording info
    anm_id = rec['anm']
    session_nums = rec['session']  # e.g., ['02', '04', '06']
    date = rec['date']
    session_keys = ['ss1', 'ss2', 'ss3'][:len(session_nums)]
    session_labels = rec['label']  # e.g., ['baseline', 'SCH', 'SCH_2']
    row_idx = f"{anm_id}-{date}"

    # Skip if this session already exists
    if row_idx in df_session_info.index:
        # print('EXSISTS')
        return df_session_info
    

    # Build row data as a dictionary first, then append
    row_data = {
        'anm': anm_id,
        'date': date,
        'session': session_nums,
        'label': session_labels,
        'conc': rec.get('conc', [None] * len(session_nums)),
        'latency': rec.get('latency', None),
    }

    # Load behavioral data and compute trial validity for each session
    for i, sess_key in enumerate(session_keys):
        beh_file = os.path.join(
            BEHAVIOR_DATA_DIR,
            f"{anm_id}-{date}-{session_nums[i]}.pkl"
        )
        beh = pd.read_pickle(beh_file)
        n_trials = len(beh['run_onsets'])

        # Valid trial criteria:
        # 1. Reward was delivered
        # 2. No mid-trial stops
        # 3. No full stops
        valid_trials = (
            (~np.isnan(beh['reward_times'])) &
            (np.array(beh['non_stop_trials']) == 0) &
            (np.array(beh['non_fullstop_trials']) == 0)
        )
        
        # select good trials
        # good_tr_param = dict(
        #     STOP_TH=10.0,          # cm/s, minimum speed to be considered as running
        #     MAX_STOP_FRAC=0.3,     # fraction of trial (0-1)
        #     MIN_MAX_SPEED=55,    # cm/s
        #     MIN_FL_TIME=1.0,       # s
        #     MAX_DUR=8,             # s, max trial length, if none then use 2*MAD filter
        #     ITI_MAX=3,           # s
        #     NORM_T=100,            # bins for time normalization
        #     fs=1000,               # Hz
        #     template_filter=False,
        #     MIN_TEMP_CORR=.75,
        #     template_drop_pct=20,  # drop worst X% by correlation
        #     return_reason = 0
        # )
        good_trials = select_good_trials(beh,
                                         STOP_TH=10.0,          # cm/s, minimum speed to be considered as running
                                         MAX_STOP_FRAC=0.3,     # fraction of trial (0-1)
                                         MIN_MAX_SPEED=55,    # cm/s
                                         MIN_FL_TIME=1.0,       # s
                                         MAX_DUR=8,             # s, max trial length, if none then use 2*MAD filter
                                         ITI_MAX=3,           # s
                                         NORM_T=100,            # bins for time normalization
                                         fs=1000,               # Hz
                                         template_filter=False,
                                         MIN_TEMP_CORR=.75,
                                         template_drop_pct=20,  # drop worst X% by correlation
                                         return_reason = 0)
        # lick index mean
        lick_idx = np.nanmean(beh['lick_selectivities'])
        # max speed
        smp_rate = 1000
        max_len = 6 * smp_rate
        speed_time = [zero_padding(np.vstack(tr_speed)[:, 1], max_len) if len(tr_speed) > 0 else np.full(max_len, np.nan)
                      for tr_speed in beh['speed_times_aligned']
                    ]
        speed_time = np.vstack(speed_time)
        # trial length
        full_trial_len = [np.nan] # run_onset to run_onset
        for t in range(1, len(beh['speed_times_aligned'])-1):
            if len(beh['speed_times_aligned'][t])>0 and len(beh['speed_times_aligned'][t+1])>0:
                full_trial_len.append(beh['speed_times_aligned'][t+1][0][0]-beh['speed_times_aligned'][t][0][0])
            else:
                full_trial_len.append(np.nan)
        full_trial_len.append(np.nan)
        full_trial_len_median = np.nanmedian(full_trial_len)
        
        max_speed_all = np.nanpercentile(speed_time, 99, axis=-1)
        max_speed_median = np.nanmedian(max_speed_all)
        row_data[f'valid_trials_{sess_key}'] = valid_trials
        row_data[f'n_valid_trials_{sess_key}'] = np.sum(valid_trials)
        row_data[f'perc_valid_trials_{sess_key}'] = np.sum(valid_trials) / n_trials
        row_data[f'good_trials_{sess_key}'] = good_trials
        row_data[f'n_good_trials_{sess_key}'] = np.sum(good_trials)
        row_data[f'perc_good_trials_{sess_key}'] = np.sum(good_trials) / n_trials
        row_data[f'lick_idx_{sess_key}'] = lick_idx
        row_data[f'max_speed_median_{sess_key}'] = max_speed_median
        row_data[f'full_trial_len_median_{sess_key}'] = full_trial_len_median


    # Add the complete row to DataFrame
    df_session_info.loc[row_idx] = row_data

    return df_session_info


#%%    
if __name__ == "__main__":
    output_file = os.path.join(OUTPUT_DIR, 'infusion_session_info.parquet')

    # Load existing data or start fresh
    if os.path.exists(os.path.join(OUTPUT_DIR, 'infusion_session_info_new.parquet')):
        df_session_info = pd.read_parquet(output_file)
        print(f"Loaded existing session info with {len(df_session_info)} records")
    else:
        df_session_info = []
        print("Starting new session info table")

    # Process all recordings
    for rec in tqdm(rec_lst_infusion, desc="Processing sessions"):
        df_session_info = update_session_info(df_session_info, rec)

    # Save updated table
    df_session_info.to_parquet(output_file)
    print(f"Saved {len(df_session_info)} sessions to {output_file}")
    

