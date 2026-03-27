# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 17:04:54 2025

@author: Jingyu Cao
"""
#%%
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, iqr
from scipy.ndimage import gaussian_filter1d

if r"Z:\Jingyu\code_mpfi_Jingyu" not in sys.path:
    sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")
from common.trial_selection import select_good_trials
from drug_infusion.utils_infusion import load_behaviour
from common.utils_basic import zero_padding
from common import plotting_functions_Jingyu as pf
pf.mpl_formatting()
#%% functions
def plot_paired_violin_comparison(df, colname, ylabel=None, title_prefix=None, ylim=None):
    """
    Plot 4-column half-violin plot for paired comparison:
    baseline vs ctrl and baseline vs SCH, for a given variable.

    Parameters:
        df (pd.DataFrame): Must have 'anm_id', 'date', 'label', and the `colname`.
        colname (str): Name of the column to compare.
        ylabel (str): Label for y-axis (optional).
        title_prefix (str): Text to prepend to title (optional).
        ylim (tuple): y-axis limits (optional).
    """
    # Pairing extraction
    ctrl_pairs, sch_pairs = [], []
    for (anm, date), g in df.groupby(['anm_id', 'date']):
        if 'baseline' in g['label'].values and 'ctrl' in g['label'].values:
            base = g.loc[g['label'] == 'baseline', colname].values[0]
            ctrl = g.loc[g['label'] == 'ctrl', colname].values[0]
            if not (np.isnan(base) or np.isnan(ctrl)):
                ctrl_pairs.append((base, ctrl))
        if 'baseline' in g['label'].values and 'SCH' in g['label'].values:
            base = g.loc[g['label'] == 'baseline', colname].values[0]
            sch = g.loc[g['label'] == 'SCH', colname].values[0]
            if not (np.isnan(base) or np.isnan(sch)):
                sch_pairs.append((base, sch))

    ctrl_pairs = np.array(ctrl_pairs)
    sch_pairs = np.array(sch_pairs)
    if len(ctrl_pairs) == 0 or len(sch_pairs) == 0:
        print(f"!!! Not enough data for {colname}. Skipping.")
        return

    ctrl_p = wilcoxon(ctrl_pairs[:, 0], ctrl_pairs[:, 1]).pvalue
    sch_p = wilcoxon(sch_pairs[:, 0], sch_pairs[:, 1]).pvalue

    # Plot
    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
    data = [ctrl_pairs[:, 0], ctrl_pairs[:, 1], sch_pairs[:, 0], sch_pairs[:, 1]]
    positions = [1, 2, 3.5, 4.5]
    colors = ['steelblue', 'grey', 'steelblue', 'orange']

    vp = ax.violinplot(data, positions=positions, showextrema=False)

    for i in range(len(data)):
        body = vp['bodies'][i]
        body.set_color(colors[i])
        body.set_edgecolor('none')
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        if i % 2 == 0:
            body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], -np.inf, m)
        else:
            body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], m, np.inf)

    # Paired lines and scatter
    for a, b, x1, x2, col1, col2 in [
        (ctrl_pairs[:, 0], ctrl_pairs[:, 1], 1.1, 1.9, colors[0], colors[1]),
        (sch_pairs[:, 0], sch_pairs[:, 1], 3.6, 4.4, colors[2], colors[3])
    ]:
        # ax.plot([x1]*len(a), a, 'o', color=col1, alpha=0.3, markersize=3)
        # ax.plot([x2]*len(b), b, 'o', color=col2, alpha=0.3, markersize=3)
        ax.plot([[x1]*len(a), [x2]*len(b)], [a, b], color='gray', alpha=0.2, linewidth=1)
        ax.plot([x1, x2], [np.median(a), np.median(b)], color='gray', linewidth=2)
        ax.scatter(x1, np.median(a), s=10, facecolor=col1, edgecolor=None, alpha=1, zorder=3)
        ax.scatter(x2, np.median(b), s=10, facecolor=col2, edgecolor=None, alpha=1, zorder=3)

    ax.set_xticks([1.5, 4])
    ax.set_xticklabels(['Baseline vs Ctrl', 'Baseline vs SCH'])
    ax.set_xlim(0.5, 5)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title(f'{title_prefix or colname}\nCtrl p={ctrl_p:.3f}, SCH p={sch_p:.3f}')
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    fig.tight_layout()
    # plt.show()
    return fig

def extract_behaviour_info(rec_lst, PATH_beh, labels):
    df_beh_pool = pd.DataFrame(
        {'anm_id': str,
         'date': str,
         'session': int,
         'label': str,
         
         'rew_rate': float,
         'lick_distance_profile': [],
         'lick_time_profile': [],
         'lick_frequency_profile': [],
         'lick_frequency_mean': [],
         'speed_distance_profile': [],
         'speed_time_profile': [],
         'speed_time_profile_array': [],
         'speed_time_profile_mean': [],
         'speed_distance_mean': [],
         'speed_distance_max': [],
         'speed_distance_median': [],
         
         'first_licks_dist': [],
         'first_licks_dist_median': float,
         'first_licks_dist_iqr': float,
         'first_licks_dist_var': float,
         
         'first_licks_time': [],
         'first_licks_time_median': float,
         'first_licks_time_iqr': float,
         'first_licks_time_var': float,
         
         'keep_trials': []
        }
        )
    
    ss_name = ['ss1', 'ss2', 'ss3']
    for i, rec in rec_lst.iterrows():
        # rew_percen = []
        # fig, ax = plt.subplots(figsize=(3,2), dpi=200)
        print('processing {}-------------------------'.format(rec['anm']+'-'+rec['date']))
        for i, ss in enumerate(['02', '04']):
            # load behaviour
            session = rec['anm']+'-'+rec['date']+'-'+ss
            p_beh = PATH_beh / f'{session}.pkl'
            beh = pd.read_pickle(p_beh)
            selected_trials = select_good_trials(beh)
            selected_trials[:10] = 0 # exclude first 10 trials
            
            licks_dist = beh['lick_distances_aligned'][1:]
            first_licks_dist = [t[t>30][0] 
                                if type(t)!=float and len(t)>0 and len(t[t>30])>0 
                                else np.nan 
                                for t in licks_dist]
            first_licks_dist_median = np.nanmedian(first_licks_dist)
            first_licks_dist_iqr = iqr(first_licks_dist, nan_policy='omit')
            first_licks_dist_var = np.nanstd(first_licks_dist)
            
            run_onsets  = beh['run_onsets']
            licks_time = beh['lick_times_aligned']
            licks_time_filtered = []
            lick_freqs = []
            for licks in licks_time:
                if type(licks)!=float and len(licks)>0:
                    licks = np.array(licks)
                    # licks = licks[licks>500]
                    lick_freq = np.histogram(licks, bins=4000, range=(0, 4000))[0]*1000 # Hz
                    lick_freq = gaussian_filter1d(lick_freq, sigma=10)
                    licks_time_filtered.append(licks)
                    lick_freqs.append(lick_freq)
                else:
                    licks_time_filtered.append(np.nan)

            lick_freqs_mean = np.nanmean(np.vstack(lick_freqs), axis=0)
            first_licks_time = [t[t>500][0] 
                                if type(t)!=float and len(t[t>500])>0 
                                else np.nan 
                                for t in licks_time_filtered]
            first_licks_time_median = np.nanmedian(first_licks_time)
            first_licks_time_iqr = iqr(first_licks_time, nan_policy='omit')
            first_licks_time_var = np.nanstd(first_licks_time)
            
            speed_time_aligned =[np.vstack(speed)[:, 1] if len(speed)>0 else []
                                 for speed in beh['speed_times_aligned']]
            speed_time_aligned_array = []
            for speed in speed_time_aligned:
                if len(speed)>0:
                    speed_time_aligned_array.append(zero_padding(speed, 4000))
            speed_time_aligned_array = np.vstack(speed_time_aligned_array)
            speed_time_aligned_mean = np.nanmean(speed_time_aligned_array, axis=0)
            
            speed_dist_raw = beh['speed_distances_aligned'][1:]
            speed_dist = np.vstack([s for s in speed_dist_raw if len(s)>0])
            speed_dist_mean = np.nanmean(speed_dist)
            speed_dist_max = np.nanmean(np.nanmax(speed_dist, axis=1))
            speed_dist_median = np.nanmean(np.nanmedian(speed_dist, axis=1))
            
            rews = beh['reward_times']
            non_rew = np.isnan(rews)
            rew_rate = 1-(np.sum(non_rew)/len(rews))
        
            df_beh_pool.loc[len(df_beh_pool)] = np.asarray([
                                                         rec['anm'], # anm_id
                                                         rec['date'],
                                                         session,
                                                         # rec['label'][i],
                                                         labels[i],
                                                         
                                                         rew_rate,
                                                         licks_dist,
                                                         licks_time_filtered,
                                                         lick_freqs,
                                                         lick_freqs_mean,
                                                         speed_dist,
                                                         speed_time_aligned,
                                                         speed_time_aligned_array,
                                                         speed_time_aligned_mean,
                                                         speed_dist_mean,
                                                         speed_dist_max,
                                                         speed_dist_median,
                                                         
                                                         first_licks_dist,
                                                         first_licks_dist_median,
                                                         first_licks_dist_iqr,
                                                         first_licks_dist_var,
                                                         
                                                         first_licks_time,
                                                         first_licks_time_median,
                                                         first_licks_time_iqr,
                                                         first_licks_time_var,
                                                         
                                                         selected_trials
                                                         ],
                                                        dtype='object')
    return df_beh_pool

#%% load recording list (match with GCaMP data selected sessions)

# drug = 'prazosin'
# drug = 'propranolol'
drug = 'SCH'
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
OUTPUT_RES = OUT_DIR_RAW_DATA /'processed_dataframe'
df_drug = pd.read_parquet(OUTPUT_RES/rf"df_drug_pool_pyr_first3_{drug}.parquet")
df_ctrl = pd.read_parquet(OUTPUT_RES/rf"df_ctrl_pool_pyr_first3_{drug}.parquet")
PATH_beh = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\behaviour_profile")



# Extract unique recording info from neuron-level dataframes
def get_unique_rec_info(df):
    return df[['anm', 'date']].drop_duplicates().reset_index(drop=True)

rec_drug = get_unique_rec_info(df_drug)
rec_ctrl = get_unique_rec_info(df_ctrl)

out_dir = r"Z:\Jingyu\LC_HPC_manuscript\fig_infusion\behaviour"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

df_beh_drug = extract_behaviour_info(rec_drug, PATH_beh, labels=['baseline', drug])
df_beh_ctrl = extract_behaviour_info(rec_ctrl, PATH_beh, labels=['baseline', 'ctrl'])

#%% mean lick trace
xaxis = np.arange(4000)/1000
licks_profile_drug = df_beh_drug.loc[df_beh_drug['label']==drug, 'lick_frequency_mean']
licks_profile_drug = np.vstack(licks_profile_drug)
licks_profile_ctrl = df_beh_ctrl.loc[df_beh_ctrl['label']=='ctrl', 'lick_frequency_mean']
licks_profile_ctrl = np.vstack(licks_profile_ctrl)

fig, ax = pf.plot_two_traces_with_binned_stats(licks_profile_drug , licks_profile_ctrl,
                                  colors=['orange', 'grey'],
                                  labels=[f'{drug}', 'ctrl'],
                                  time_windows=[(0, 1), (1, 2), (2, 3), (3, 4)],
                                  baseline_window=None,
                                  bef=0, aft=4, sample_freq=1000
                                  )
ax.set(xlabel='time from run (s)', ylabel='lick rate (Hz)',
       xlim=(0,4), ylim=(0, 25))
plt.tight_layout()
for form in ['pdf', 'png']:
    plt.savefig(out_dir+r'\lick_trace_comparision_{}.{}'.format(drug, form),
            dpi=300)
plt.show()  
#%% mean speed trace
speeds_profile_drug = df_beh_drug.loc[df_beh_drug['label']==drug, 'speed_time_profile_mean']
speeds_profile_drug = np.vstack(speeds_profile_drug)
speeds_profile_ctrl = df_beh_ctrl.loc[df_beh_ctrl['label']=='ctrl', 'speed_time_profile_mean']
speeds_profile_ctrl = np.vstack(speeds_profile_ctrl)

fig, ax = pf.plot_two_traces_with_binned_stats(speeds_profile_drug, speeds_profile_ctrl,
                                  colors=['orange', 'grey'],
                                  labels=[f'{drug}', 'ctrl'],
                                  time_windows=[(0, 1), (1, 2), (2, 3), (3, 4)],
                                  baseline_window=None,
                                  bef=0, aft=4, sample_freq=1000
                                  )
ax.set(xlabel='time from run (s)', ylabel='speed (cm/s)',
       xlim=(0,4), ylim=(0,70))
plt.tight_layout()
for form in ['pdf', 'png']:
    plt.savefig(out_dir+r'\speed_trace_comparision_{}.{}'.format(drug, form),
            dpi=300) 
plt.show() 

#%% first lick time
fl_time_drug = df_beh_drug.loc[df_beh_drug['label']==drug, 'first_licks_time_median']
fl_time_drug = np.stack(fl_time_drug)/1000
fl_time_ctrl = df_beh_ctrl.loc[df_beh_ctrl['label']=='ctrl', 'first_licks_time_median']
fl_time_ctrl = np.stack(fl_time_ctrl)/1000
fig, ax = plt.subplots(figsize=(2, 3), dpi=300)
pf.plot_bar_with_unpaired_scatter(ax, fl_time_drug, fl_time_ctrl,
                                  xticklabels=[f'{drug}', 'saline'],
                                  colors=['orange', 'grey'],
                                  jitter=0.01, ylabel='first lick time (s)')
ax.set(ylim=(0,4))
plt.tight_layout()
for form in ['pdf', 'png']:
    plt.savefig(out_dir+r'\first_lick_time_comparision_{}.{}'.format(drug, form),
            dpi=300) 
plt.show()
#%% speed mean 
# fl_time_drug = df_beh_drug.loc[df_beh_drug['label']==drug, 'first_licks_time_median']
# fl_time_drug = np.stack(fl_time_drug)/1000
# fl_time_ctrl = df_beh_ctrl.loc[df_beh_ctrl['label']=='ctrl', 'first_licks_time_median']
# fl_time_ctrl = np.stack(fl_time_ctrl)/1000
# fig, ax = plt.subplots(figsize=(2, 3), dpi=300)
# pf.plot_bar_with_unpaired_scatter(ax, fl_time_drug, fl_time_ctrl,
#                                   xticklabels=[f'{drug}', 'saline'],
#                                   colors=['orange', 'grey'],
#                                   jitter=0.01, ylabel='mean speed')
# plt.show()