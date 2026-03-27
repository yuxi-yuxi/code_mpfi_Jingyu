# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:32:24 2026

@author: Jingyu Cao
"""
import numpy as np
from scipy.stats import ttest_ind

def extract_first_licks(beh):
    lick_times = beh['lick_times_aligned']
    first_lick_times = []
    # tot_trials = len(lick_times)
    for licks in lick_times:
        if licks is np.nan:
            first_lick_times.append(np.nan)
        else:
            licks = np.array(licks)
            licks_filtered = licks[licks>500] # only include licks after 0.5s
            if len(licks_filtered)>0:
                first_lick_times.append(licks_filtered[0])
            else: # no licks after 0.5s
                first_lick_times.append(np.nan)
    first_lick_times = np.array(first_lick_times)         
    
    return  first_lick_times
    
def mean_speed()
def speed_match(beh, idx_a, idx_b, tolerance = 1.5):
    '''
    

    Parameters
    ----------
    beh : TYPE
        DESCRIPTION.
    idx_a : TYPE
        DESCRIPTION.
    idx_b : TYPE
        DESCRIPTION.
    tolerance : TYPE, optional
        How tight the speed match should be. The default is 1.5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    speeds_aligned = [np.stack(speed)[:, 1] if len(speed)>0 else [] 
                      for speed in beh['speed_times_aligned']]
    
    # binning
    speeds_binned = []
    speed_trial_idx = []
    for idx, speed in enumerate(speeds_aligned):
        if len(speed)<4000:
            speeds_binned.append(np.nan)
        else: # 0.5s bin, 7 bins
            speed_binned = speed[:4000].reshape(8, 500).mean(axis=1)
            speeds_binned.append(speed_binned)
            speed_trial_idx.append(idx)
    idx_a_speed = [i for i in idx_a if i in speed_trial_idx]
    idx_b_speed = [i for i in idx_b if i in speed_trial_idx]
    if len(idx_a_speed)==0 or len(idx_b_speed)==0:
        return  #skip sessions without out any early or late trials                        
    speed_a_binned = [speeds_binned[i] for i in idx_a_speed]
    speed_b_binned = [speeds_binned[i] for i in idx_b_speed]        
    speed_a_miu = np.mean(speed_a_binned, axis=0)
    speed_a_std = np.std(speed_a_binned, axis=0)
    speed_b_miu = np.mean(speed_b_binned, axis=0)
    speed_b_std = np.std(speed_b_binned, axis=0)
    
    a_bound_low = speed_a_miu-tolerance*speed_a_std 
    a_bound_high = speed_a_miu+tolerance*speed_a_std
    b_bound_low = speed_b_miu-tolerance*speed_b_std 
    b_bound_high = speed_b_miu+tolerance*speed_b_std
    
    a_in_bound = np.all((speed_a_binned>=a_bound_low)&
                        (speed_a_binned<=a_bound_high),
                        axis=1)
    b_in_bound = np.all((speed_b_binned>=b_bound_low)&
                        (speed_b_binned<=b_bound_high),
                        axis=1)
    a_in_bound_idx = [idx_a_speed[i] for i in np.where(a_in_bound)[0]]
    b_in_bound_idx = [idx_b_speed[i] for i in np.where(b_in_bound)[0]]
    
    # if len(early_in_bound_idx)<10 or len(late_in_bound_idx)<10:
    #     continue # skip sessions without enough trials
    
    # check speed match using t-test
    a_speed_binned_inbound = np.vstack([speeds_binned[i] for i in a_in_bound_idx])
    b_speed_binned_inbound = np.vstack([speeds_binned[i] for i in b_in_bound_idx])
    p_values = []
    for i in range(a_speed_binned_inbound.shape[1]): #n_bins
        res = ttest_ind(a_speed_binned_inbound[:, i], b_speed_binned_inbound[:, i])
        p_values.append(res[1])
    p_values = np.array(p_values)
    
    # # plot speed match validation
    # fig, axs = plt.subplots(1, 2, figsize=(4,2), dpi=300); fig.tight_layout()
    # plt.suptitle=(f'{rec}')
    # early_speeds = [utl.zero_padding(speeds_aligned[i], 4000) for i in early_trials_indx]
    # late_speeds = [utl.zero_padding(speeds_aligned[i], 4000) for i in late_trials_indx]
    # early_speeds_matched = [utl.zero_padding(speeds_aligned[i], 4000) for i in early_in_bound_idx]
    # late_speeds_matched = [utl.zero_padding(speeds_aligned[i], 4000) for i in late_in_bound_idx]
    # ax=axs[0]
    # pf.plot_mean_trace(early_speeds, ax, color='grey')
    # pf.plot_mean_trace(late_speeds, ax, color='steelblue')
    # ax.set(title='before_match', ylabel='speed', xlabel='time')
    # ax=axs[1]
    # pf.plot_mean_trace(early_speeds_matched, ax, color='grey')
    # pf.plot_mean_trace(late_speeds_matched, ax, color='steelblue')
    # ax.set(title='after_match', ylabel='speed', xlabel='time')
    # plt.savefig(figure_out+r'\speed_match_{}.png'.format(rec))
    # # plt.show()
    # plt.close()
    
    return a_in_bound_idx, b_in_bound_idx, p_values

