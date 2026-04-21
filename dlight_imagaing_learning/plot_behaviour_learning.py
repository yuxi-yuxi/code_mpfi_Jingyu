# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:01:35 2026

@author: Jingyu Cao


plot learning curve for mean speed and lick selectivity index for leanring task
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pickle 
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
# from dlight_imaging.Dbh_dlight.recording_list import all_recs
from dlight_imaging.geco_dlight.recording_list import all_recs
import common.utils_behaviour as utl
from common.utils_basic import zero_padding
from common import plotting_functions_Jingyu as pf
if (r"Z:\Dinghao\code_mpfi_dinghao\utils" in sys.path) == False:
    sys.path.append(r"Z:\Dinghao\code_mpfi_dinghao\utils")
# import pre-processing functions 
import behaviour_functions as bf
from common.trial_selection import seperate_valid_trial

#%%

# OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
# OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_grid_free_dilation' n.m

# OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\Dbh_dlight")
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\geco_dlight")
OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'TEST_PLOTS' / 'behaviour_summary_figs'
if not OUT_DIR_FIG.exists():
    OUT_DIR_FIG.mkdir(parents=True, exist_ok=False)

max_length = 8 * 1000
SESSIONS_PER_FIG = 5

# Group recordings by animal
recs_by_animal = defaultdict(list)
for rec in all_recs:
    recs_by_animal[rec[:5]].append(rec)

for anm, anm_recs in recs_by_animal.items():
    # Process all sessions first to collect data
    session_data = []

    for rec in tqdm(anm_recs):
        p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'

        if not p_beh_file.exists():
            print('processing {}...'.format(rec))
            txt_path = Path(r"Z:\Jingyu\mice-expdata\{}\A{}T.txt".format(rec[0:5], rec[2:]))
            if not txt_path.exists():
                print('no txt file!!!')
                continue
            behavioural_data = bf.process_behavioural_data_imaging(txt_path)
            with open(p_beh_file, 'wb') as f:
                pickle.dump(behavioural_data, f)

        if not p_beh_file.exists():
            continue

        beh = pd.read_pickle(p_beh_file)
        lick_idx_median = np.nanmedian(beh['lick_selectivities'])
        reward_times = beh['reward_times_aligned']
        speed_all = utl.extract_speed_trace(beh)
        speed_array = [zero_padding(speed, max_length) for speed in speed_all if len(speed) != 0]
        licks_all = beh['lick_times_aligned']
        first_licks_all = utl.extract_first_licks(beh)
        valid_trials = seperate_valid_trial(beh)

        # Calculate statistics
        if len(speed_array) > 0:
            mean_speed_trace = np.nanmean(speed_array, axis=0)
            max_speed = np.nanmax(mean_speed_trace)
            mean_speed = np.nanmean(mean_speed_trace)
        else:
            max_speed = np.nan
            mean_speed = np.nan

        # Calculate % valid trials
        n_total_trials = len(valid_trials)
        pct_valid = np.sum(valid_trials) / n_total_trials * 100 if n_total_trials > 0 else np.nan

        # Calculate % first licks < 2s and > 2.5s
        valid_first_licks = first_licks_all[~np.isnan(first_licks_all)]
        n_with_licks = len(valid_first_licks)
        pct_early_lick = np.sum(valid_first_licks < 2000) / n_with_licks * 100 if n_with_licks > 0 else np.nan
        pct_late_lick = np.sum(valid_first_licks > 2500) / n_with_licks * 100 if n_with_licks > 0 else np.nan

        # Calculate lick concentration around reward
        lick_reward_distances = []
        licks_near_reward = 0
        total_licks = 0
        for licks, reward_t in zip(licks_all, reward_times):
            if licks is None or (isinstance(licks, float) and np.isnan(licks)):
                continue
            if np.isnan(reward_t):
                continue
            licks = np.array(licks)
            if len(licks) == 0:
                continue
            distances = np.abs(licks - reward_t)
            lick_reward_distances.extend(distances)
            licks_near_reward += np.sum(distances <= 1000)
            total_licks += len(licks)

        pct_licks_near_reward = (licks_near_reward / total_licks * 100) if total_licks > 0 else np.nan
        median_lick_reward_dist = np.median(lick_reward_distances) if len(lick_reward_distances) > 0 else np.nan

        session_data.append({
            'rec': rec,
            'lick_idx_median': lick_idx_median,
            'max_speed': max_speed,
            'mean_speed': mean_speed,
            'pct_valid': pct_valid,
            'pct_early_lick': pct_early_lick,
            'pct_late_lick': pct_late_lick,
            'pct_licks_near_reward': pct_licks_near_reward,
            'median_lick_reward_dist': median_lick_reward_dist,
            'speed_array': speed_array,
            'first_licks_all': first_licks_all,
            'licks_all': licks_all
        })

    if len(session_data) == 0:
        continue

    # Save summary dataframe for this animal (only scalar statistics, not arrays)
    df_summary = pd.DataFrame([{
        'rec': s['rec'],
        'lick_idx_median': s['lick_idx_median'],
        'max_speed': s['max_speed'],
        'mean_speed': s['mean_speed'],
        'pct_valid': s['pct_valid'],
        'pct_early_lick': s['pct_early_lick'],
        'pct_late_lick': s['pct_late_lick'],
        'pct_licks_near_reward': s['pct_licks_near_reward'],
        'median_lick_reward_dist': s['median_lick_reward_dist']
    } for s in session_data])
    summary_path = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{anm}_behaviour_summary.parquet'
    df_summary.to_parquet(summary_path)
    print(f'Saved summary: {summary_path}')

    #%%
    # Group sessions into figures of 5
    n_sessions = len(session_data)
    n_figs = int(np.ceil(n_sessions / SESSIONS_PER_FIG))

    for fig_idx in range(n_figs):
        start_idx = fig_idx * SESSIONS_PER_FIG
        end_idx = min(start_idx + SESSIONS_PER_FIG, n_sessions)
        sessions_in_fig = session_data[start_idx:end_idx]
        n_sess_in_fig = len(sessions_in_fig)

        # Create figure: 3 rows (speed, first lick hist, all licks hist) x n_sessions columns
        # Add extra width for text annotations
        fig, axs = plt.subplots(3, n_sess_in_fig, figsize=(4 * n_sess_in_fig, 8),
                                squeeze=False, dpi=150)
        fig.suptitle(f'{anm} - Sessions {start_idx + 1} to {end_idx}', fontsize=14, fontweight='bold')

        for col_idx, sess in enumerate(sessions_in_fig):                                                                                                                                                                      
            rec = sess['rec']
            speed_array = sess['speed_array']
            first_licks_all = sess['first_licks_all']
            licks_all = sess['licks_all']

            # Row 0: Speed trace
            ax = axs[0, col_idx]
            if len(speed_array) > 0:
                pf.plot_mean_trace(speed_array, ax)
            # Include stats in the title (speed-related)
            title_text = (f"{rec}\n"
                          f"lick_idx: {sess['lick_idx_median']:.2f} | "
                          f"max spd: {sess['max_speed']:.1f} | "
                          f"mean spd: {sess['mean_speed']:.1f}\n"
                          f"valid: {sess['pct_valid']:.0f}%")
            ax.set_title(title_text, fontsize=9)
            ax.set_xlabel('Time (ms)')
            if col_idx == 0:
                ax.set_ylabel('Speed')

            # Row 1: First lick histogram
            ax = axs[1, col_idx]
            valid_first_licks = first_licks_all[~np.isnan(first_licks_all)]
            if len(valid_first_licks) > 0:
                ax.hist(valid_first_licks, bins=30, range=(0, max_length), color='steelblue', alpha=.6)
            ax.set_title(f"lick<2s: {sess['pct_early_lick']:.0f}% | lick>2.5s: {sess['pct_late_lick']:.0f}%", fontsize=8)
            ax.set_xlabel('Time (ms)')
            if col_idx == 0:
                ax.set_ylabel('First lick count')

            # Row 2: All licks histogram
            ax = axs[2, col_idx]
            all_licks_flat = np.hstack(licks_all)
            if len(all_licks_flat) > 0:
                ax.hist(all_licks_flat, bins=100, range=(0, max_length), color='orange', alpha=.6)
            ax.set_title(f"lick@rwd: {sess['pct_licks_near_reward']:.0f}% | dist: {sess['median_lick_reward_dist']:.0f}ms", fontsize=8)
            ax.set_xlabel('Time (ms)')
            if col_idx == 0:
                ax.set_ylabel('All licks count')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle

        # Save figure
        fig_name = f'{anm}_sessions_{start_idx + 1}_to_{end_idx}.png'
        pf.save_fig(fig, OUT_DIR_FIG, fig_name, dpi=150, forms=['png',], save=1)
        print(f'Saved: {fig_name}')
        plt.close(fig)

    print(f'Completed {anm}: {n_sessions} sessions in {n_figs} figure(s)')

print('All animals processed!')

        