# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:32:18 2026

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd

#%%
def seperate_valid_trial(beh, time_thresh=None):    
    if time_thresh is None:
        time_thresh = 0
    valid_trials = np.where((~np.isnan(beh['reward_times']))&
                            (np.array(beh['non_stop_trials'])==0)&
                            (np.array(beh['non_fullstop_trials'])==0)&
                            (np.array(beh['reward_times_aligned'])<time_thresh),
                            True, False)
    return valid_trials.astype('bool')

def extract_first_lick(beh):
    # calculate first lick time alingend to run-onset
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
    
    return first_lick_times

def extract_speed_trace(beh):
    speeds_aligned = [np.stack(speed)[:, 1] if len(speed)>0 else [] 
                      for speed in beh['speed_times_aligned']]
    return speeds_aligned

def extract_reward(beh):
    rews = beh['reward_times']
    ro = beh['run_onsets']
    
    reward_aligned = [rews[t] - ro[t]
                      for t in range(len(ro))]
    return reward_aligned

def valid_speed(sp, min_len=2):
    return isinstance(sp, (list, np.ndarray)) and len(sp) >= min_len

def mad_bounds(x, k=2.0):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return med - k*mad, med + k*mad

def time_normalize(sp, n=100):
    # sp must be 1D array length >= 2
    x_old = np.linspace(0, 1, len(sp))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, sp)

def z(x):
    return (x - np.nanmean(x)) / np.nanstd(x)

DEFAULTS = dict(
    STOP_TH=10.0,          # cm/s, minimum speed to be considered as running
    MAX_STOP_FRAC=0.3,     # fraction of trial (0-1)
    MIN_MAX_SPEED=65.0,    # cm/s
    MIN_FL_TIME=1.5,       # s
    MAX_DUR=8,             # s, max trial length, if none then use 2*MAD filter
    ITI_MAX=2.0,           # s
    NORM_T=100,            # bins for time normalization
    fs=1000,               # Hz
    template_filter=False,
    MIN_TEMP_CORR=.75,
    template_drop_pct=20,  # drop worst X% by correlation
    return_reason = 0
)

def select_good_trials(beh, **overrides):
    """
    Selects "good" trials and returns a boolean keep mask aligned to original trials.

    Parameters can be overridden via keyword args, e.g.:
        keep = select_good_trials(beh, ITI_MAX=3, template_filter=True)

    Returns
    -------
    keep : (n_trials,) bool
    """
    P = DEFAULTS.copy()
    P.update(overrides)

    STOP_TH = P["STOP_TH"]
    MAX_STOP_FRAC = P["MAX_STOP_FRAC"]
    MIN_MAX_SPEED = P["MIN_MAX_SPEED"]
    MIN_FL_TIME = P["MIN_FL_TIME"]
    MAX_DUR = P['MAX_DUR']
    ITI_MAX = P["ITI_MAX"]
    NORM_T = P["NORM_T"]
    fs = P["fs"]
    template_filter = P["template_filter"]
    MIN_TEMP_CORR = P['MIN_TEMP_CORR']
    template_drop_pct = P["template_drop_pct"]
    return_reason = P['return_reason']
    
    speeds_aligned = extract_speed_trace(beh)
    first_lick_times = extract_first_lick(beh)
    ro = np.asarray(beh["run_onsets"], dtype=float)
    rew = np.asarray(beh["reward_times"], dtype=float)
    n_trials = len(speeds_aligned)

    # --- ITI in seconds (do NOT multiply by fs) ---
    ITI = np.full(n_trials, np.nan)
    for t in range(1, n_trials):
        if np.isfinite(ro[t]) and np.isfinite(rew[t-1]):
            ITI[t] = ro[t] - rew[t-1]

    # --- Per-trial metrics in original trial indexing ---
    dur = np.full(n_trials, np.nan)
    mean_sp = np.full(n_trials, np.nan)
    max_sp = np.full(n_trials, np.nan)
    cv_sp = np.full(n_trials, np.nan)
    stop_frac = np.full(n_trials, np.nan)

    for t in range(n_trials):
        sp = speeds_aligned[t]
        if not valid_speed(sp):
            continue

        sp = np.asarray(sp, dtype=float)

        dur[t] = len(sp)
        mean_sp[t] = np.mean(sp)
        max_sp[t] = np.max(sp)
        m = mean_sp[t]
        cv_sp[t] = np.std(sp) / m if (np.isfinite(m) and m > 0) else np.nan

        stop_frac[t] = np.mean(sp < STOP_TH)


    # trial filtering

    reject = np.zeros(n_trials, dtype=bool)
    reject_reason = {}
    
    # first round filter: trials without reward or clear run onset
    valid_trials = seperate_valid_trial(beh)
    reject_reason["invalid_trials"] = ~valid_trials
    reject |= reject_reason["invalid_trials"]
    
    # invalid speed
    reject_reason["invalid_speed"] = ~np.isfinite(dur)
    reject |= reject_reason["invalid_speed"]
    
    # ITI
    reject_reason["bad_ITI"] = (~np.isfinite(ITI)) | (ITI >= ITI_MAX * fs)
    reject |= reject_reason["bad_ITI"]
    
    # first lick time
    reject_reason["early_or_missing_first_lick"] = (
        (~np.isfinite(first_lick_times)) |
        (first_lick_times <= MIN_FL_TIME * 1000)
    )
    reject |= reject_reason["early_or_missing_first_lick"]
    
    # duration
    if MAX_DUR:
        reject_reason["long_duration"] = dur >= MAX_DUR * fs
    else:
        lo, hi = mad_bounds(dur)
        # reject_reason["long_duration"] = (dur <= lo) | (dur >= hi)
        reject_reason["long_duration"] = (dur >= hi)
    reject |= reject_reason["long_duration"]
    
    # max speed
    reject_reason["low_max_speed"] = (~np.isfinite(max_sp)) | (max_sp < MIN_MAX_SPEED)
    reject |= reject_reason["low_max_speed"]
    
    # mean speed
    lo, hi = mad_bounds(mean_sp)
    # reject_reason["bad_mean_speed"] = (mean_sp <= lo) | (mean_sp >= hi)
    reject_reason["bad_mean_speed"] = (mean_sp <= lo)

    reject |= reject_reason["bad_mean_speed"]
    
    # CV
    # lo, hi = mad_bounds(cv_sp)
    # reject_reason["bad_speed_CV"] = (cv_sp <= lo) | (cv_sp >= hi)
    # reject |= reject_reason["bad_speed_CV"]
    
    # stop fraction
    reject_reason["excessive_stopping"] = (
        (~np.isfinite(stop_frac)) | (stop_frac >= MAX_STOP_FRAC)
    )
    reject |= reject_reason["excessive_stopping"]
    
    keep = ~reject


    # --- Template filter (optional) ---
    if template_filter and np.sum(keep) >= 5:
        
        reject_reason["template_mismatch"] = np.zeros(n_trials, dtype=bool)
        
        sp_norm = np.full((n_trials, NORM_T), np.nan)

        for t in range(n_trials):
            if keep[t]:
                sp = np.asarray(speeds_aligned[t], dtype=float)
                sp_norm[t] = time_normalize(sp, NORM_T)

        template = np.nanmedian(sp_norm[keep], axis=0)
        template_z = z(template)

        corr = np.full(n_trials, np.nan)
        for t in range(n_trials):
            if keep[t]:
                corr[t] = np.corrcoef(z(sp_norm[t]), template_z)[0, 1]
        
        if MIN_TEMP_CORR:
            thr = MIN_TEMP_CORR
        else:
            thr = np.nanpercentile(corr[keep], template_drop_pct)
        
        reject_reason["template_mismatch"] = keep & (corr < thr)
        keep &= corr >= thr
        
    if return_reason:    
        return keep, reject_reason
    else:
        return keep

def get_reject_reason_str(reject_reason, t):
    reasons = [k for k, v in reject_reason.items() if v[t]]
    if len(reasons) == 0:
        return "KEEP"
    return "REJECT: " + ", ".join(reasons)     
#%%
if __name__ == '__main__':
    import sys
    import os
    import matplotlib.pyplot as plt
    if ("Z:\Jingyu\Code\Python" in sys.path) == False:
        sys.path.append("Z:\Jingyu\Code\Python")
    import utils_Jingyu as utl
    import plotting_functions_Jingyu as pf
    
    rec = 'AC310-20250821-02'
    
    out_dir = r"\\mpfi.org\Public\Wang lab\Jingyu\Code\2p_SCH23390_infusion\test_trial_selection\{}".format(rec)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    beh = pd.read_pickle(r"\\mpfi.org\Public\Wang lab\Jingyu\Code\Python\2p_SCH23390_infusion\behaviour_profile\{}.pkl".format(rec))
    keep_trials, reject_reason = select_good_trials(beh,
                                     STOP_TH = 10.0 ,         # cm/s
                                     MAX_STOP_FRAC = 0.3,  # % of trial
                                     MIN_MAX_SPEED = 50, # cm/s
                                     MIN_FL_TIME = 2, # s
                                     MAX_DUR = 6.5, 
                                     ITI_MAX = 2,         # s
                                     NORM_T = 100,           # bins for time normalization
                                     fs = 1000,
                                     template_filter = 1,
                                     MIN_TEMP_CORR=.75,
                                     return_reason = 1)
    
    speeds_aligned = extract_speed_trace(beh)
    first_lick_times = extract_first_lick(beh)
    rew = extract_reward(beh)
    
    for t, sp in enumerate(speeds_aligned):
        if len(sp)>0:
            fig, ax = plt.subplots(figsize=(3, 2), dpi=200)
            ax.plot(sp)
            ax.axvline(first_lick_times[t], .85, .9, color='orange', lw=5)
            ax.axvline(rew[t], .85, .9, color='tab:blue', lw=5)
            
            reason_str = get_reject_reason_str(reject_reason, t)
            ax.set_title(
            f"Trial {t} | {'KEEP' if keep_trials[t] else 'REJECT'}\n{reason_str}",
            fontsize=8,
            )
            fig.tight_layout()
            plt.savefig(out_dir+f'\Trial_{t}.png', dpi=200)
            # plt.show()
            plt.close()
    
    fig, ax = plt.subplots(figsize=(3, 2), dpi=200)
    speeds_aligned_selected = [utl.zero_padding(sp, 4000) for t, sp in enumerate(speeds_aligned) if keep_trials[t]]
    pf.plot_mean_trace(speeds_aligned_selected, ax)
    plt.show()
    