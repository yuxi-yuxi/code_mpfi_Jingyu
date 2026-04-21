# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 2026

@author: Jingyu Cao

Classify sessions as 'learning' or 'well-trained' based on behaviour metrics.
Criteria for task acquisition (must be met for 3 consecutive days):
  - (lick@rwd > 70% OR dist < 500ms) AND max_speed > 50

Sessions up to and including the first day of the 3-day streak are 'learning'.
Sessions after are 'well-trained'.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob

if __name__ == '__main__':
    #%% Configuration
    OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\geco_dlight")
    # OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\geco_dlight")
    BEHAVIOUR_PROFILE_DIR = OUT_DIR_RAW_DATA / 'behaviour_profile'
    
    # Criteria thresholds
    LICK_NEAR_REWARD_THRESH = 70  # %
    MEDIAN_DIST_THRESH = 500  # ms
    MAX_SPEED_THRESH = 50
    CONSECUTIVE_DAYS_REQUIRED = 2
    
    #%% Load all animal summary files
    summary_files = list(BEHAVIOUR_PROFILE_DIR.glob('*_behaviour_summary.parquet'))
    print(f"Found {len(summary_files)} animal summary files")
    
    all_results = []
    
    for summary_file in summary_files:
        anm = summary_file.stem.replace('_behaviour_summary', '')
        df = pd.read_parquet(summary_file)
    
        if len(df) == 0:
            print(f"  {anm}: No sessions found, skipping")
            continue
    
        # Check if animal meets criteria for each session
        # Criteria: (lick@rwd > 70% OR dist < 500ms) AND max_speed > 50
        df['meets_lick_criteria'] = (df['pct_licks_near_reward'] > LICK_NEAR_REWARD_THRESH) | \
                                     (df['median_lick_reward_dist'] < MEDIAN_DIST_THRESH)
        df['meets_speed_criteria'] = df['max_speed'] > MAX_SPEED_THRESH
        df['meets_all_criteria'] = df['meets_lick_criteria'] & df['meets_speed_criteria']
    
        # Find first occurrence of 3 consecutive sessions meeting criteria
        learned_idx = None
        for i in range(len(df) - CONSECUTIVE_DAYS_REQUIRED + 1):
            if df['meets_all_criteria'].iloc[i:i + CONSECUTIVE_DAYS_REQUIRED].all():
                learned_idx = i  # First day of the 3-day streak
                break
    
        # Classify sessions
        if learned_idx is not None:
            # Sessions 0 to learned_idx (inclusive) are 'learning'
            # Sessions after learned_idx are 'well-trained'
            df['stage'] = ['learning' if i <= learned_idx else 'well-trained'
                           for i in range(len(df))]
            df['learned_session'] = df['rec'].iloc[learned_idx]
            # Days relative to learned session (negative = before, 0 = learned day, positive = after)
            df['days_from_learned'] = [i - learned_idx for i in range(len(df))]
            print(f"  {anm}: Learned at session {learned_idx + 1} ({df['rec'].iloc[learned_idx]})")
        else:
            # Animal never met criteria for 3 consecutive days
            df['stage'] = 'learning'
            df['learned_session'] = None
            df['days_from_learned'] = np.nan
            print(f"  {anm}: Did not reach learning criteria (all sessions marked as 'learning')")
    
        df['animal'] = anm
        all_results.append(df)
    
    #%% Combine and save results
    if len(all_results) > 0:
        df_all = pd.concat(all_results, ignore_index=True)
    
        # Reorder columns
        cols_order = ['animal', 'rec', 'stage', 'learned_session', 'days_from_learned',
                      'lick_idx_median', 'max_speed', 'mean_speed',
                      'pct_valid', 'pct_early_lick', 'pct_late_lick',
                      'pct_licks_near_reward', 'median_lick_reward_dist',
                      'meets_lick_criteria', 'meets_speed_criteria', 'meets_all_criteria']
        df_all = df_all[[c for c in cols_order if c in df_all.columns]]
    
        # Save combined results
        output_path = OUT_DIR_RAW_DATA / 'all_animals_learning_classification.parquet'
        df_all.to_parquet(output_path)
        print(f"\nSaved combined results to: {output_path}")
    
        # Print summary
        print("\n=== Summary ===")
        for anm in df_all['animal'].unique():
            df_anm = df_all[df_all['animal'] == anm]
            n_learning = (df_anm['stage'] == 'learning').sum()
            n_trained = (df_anm['stage'] == 'well-trained').sum()
            learned_sess = df_anm['learned_session'].iloc[0]
            print(f"{anm}: {n_learning} learning, {n_trained} well-trained | "
                  f"Learned at: {learned_sess if learned_sess else 'N/A'}")
    else:
        print("No data to process!")
else:
    OUT_DIR_RAW_DATA 
    
