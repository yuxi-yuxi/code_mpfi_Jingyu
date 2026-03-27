# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 10:39:21 2026

@author: Jingyu Cao
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

if r"Z:\Jingyu\code_mpfi_Jingyu" not in sys.path:
    sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")
from common.utils_basic import zero_padding, normalize
from common.utils_imaging import align_trials, load_bin_file, fov_trace_extraction
from common.mask.utils_mask import generate_adaptive_membrane_mask

def cross_corr_speed(raw_dlight_traces, beh,
                     fs_old = 30,
                     fs_new = 1000, # upsample dlight trace to 1000 hz
                     ):
    # correlation with speed
    n_old = raw_dlight_traces.shape[-1]
    # duration = n_old / fs_old
    # time vectors
    t_old = np.arange(n_old) / fs_old
    t_new = np.arange(round(n_old * fs_new / fs_old)) / fs_new
    # dlight trace
    raw_dlight_tarces_1000 = np.interp(t_new, t_old, raw_dlight_traces) # linear interpolation
    # upsample frame times
    frame_times_1000 = np.interp(t_new, t_old, beh['frame_times'][:n_old])
    # find new run_onset index
    run_onsets_times = beh['run_onsets']
    
    run_frames_1000 = []
    for run in run_onsets_times:
        if np.isnan(run):
            run_frames_1000.append(0)
        else:
            nearest_frame_idx = np.argmin(np.abs(frame_times_1000 - run))
            run_frames_1000.append(nearest_frame_idx)
    
    # align upsampled trace to run
    bef, aft = 0, 4
    win_frames = int((bef+aft)*fs_new)
    tot_trial = len(run_onsets_times)
    aligned_signal = np.zeros((tot_trial, win_frames))
    for t in range(tot_trial):
        curr_trace = raw_dlight_tarces_1000[run_frames_1000[t]-int(bef*fs_new):run_frames_1000[t]+int(aft*fs_new)]
        if curr_trace.shape[0]<win_frames or run_frames_1000[t]==0:
            aligned_signal[t,:]=np.nan
        else:
            aligned_signal[t,:]=curr_trace
    raw_dlight_tarces_1000_aligned = aligned_signal     
    # raw_dlight_tarces_1000_aligned = utl.align_trials(raw_dlight_tarces_1000, alignment, beh, bef, aft, fs=fs_new)
    
    aligned_trials = ~np.isnan(raw_dlight_tarces_1000_aligned).any(axis=1)
    aligned_trials_idx = np.where(aligned_trials)[0]
    raw_dlight_traces_1000_mean = np.nanmean(raw_dlight_tarces_1000_aligned[aligned_trials], axis=0)
    # calculate spead trial mean
    speeds = [np.vstack(beh['speed_times_aligned'][i])[:4000, 1] for i in aligned_trials_idx]
    speeds = np.stack([zero_padding(speed, 4000) for speed in speeds])
    speed_mean = np.nanmean(speeds, axis=0)
    
    # single trial correlation speed-dlight
    x = raw_dlight_tarces_1000_aligned[aligned_trials]
    y = speeds
    # single_trial_corr = pearsonr(x, y, axis=-1)
    # print(np.nanmedian(single_trial_corr[0]))
    # df_sess_info.at[rec, 'speed_corr_single_trial_r_median'] = np.nanmedian(single_trial_corr[0])
    
    # cross-correlation of fov_dlight and speed
    x = speed_mean
    y = raw_dlight_traces_1000_mean
    # normalize
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    # compute cross-correlation
    corr = np.correlate(x, y, mode='full')
    lags = np.arange(-len(x)+1, len(x))
    
    # normalize to correlation coefficient
    corr /= len(x)
    
    return corr, lags

if __name__ == '__main__':
    #%% define paths and params
    # exp = r'Dbh_dlight'
    exp = r'geco_dlight'
    
    if exp == r'Dbh_dlight':
        from dlight_imaging.Dbh_dlight.recording_list import rec_lst
        OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
        ch_dlight, ch_red = 0, 1
        dlight_mov_bin = r'data.bin'
        red_mov_bin = r'data_chan2.bin'
    elif exp == r'geco_dlight':
        from dlight_imaging.geco_dlight.recording_list import rec_lst
        OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight")
        ch_dlight, ch_red = 1, 0
        dlight_mov_bin = r'data_chan2.bin'
        red_mov_bin = r'data.bin'
        
    p_df_out = OUT_DIR_RAW_DATA/rf"{exp}_session_info.parquet"
    f_out_plot = None
    
    MIN_LICK_IDX = 0.85 # lick selectivity index
    MIN_MAX_SPEED = 55 # speed, cm/s
    MIN_PERC_VALID = 0.6 # % valid trials
    MAX_TRIAL_LENGTH = 6000 # trial length, run onset to next trial's run onset, ms
    
    if not os.path.exists(p_df_out):
        extract_session_info = 1
    else: 
        extract_session_info = 0
        
    plot = 0
    #%% calculate behaviour metrics
    if extract_session_info == 0:
        df_sess_info = pd.read_parquet(p_df_out)
    elif extract_session_info:
        df_sess_info=pd.DataFrame()
        df_sess_info['mean_speed'] = ''
        df_sess_info['max_speed'] = ''
        df_sess_info['full_trial_len'] = ''
        df_sess_info['speed_len'] = ''
        df_sess_info['valid_trials'] = ''
        df_sess_info['first_lick_times'] = ''
        df_sess_info['early_lick_trials'] = ''
        df_sess_info['late_lick_trials'] = ''
        
        print('extracting behaviour info...')
        for session in tqdm(rec_lst):
            print('processing {}...'.format(session))
            beh = pd.read_pickle(OUT_DIR_RAW_DATA/'behaviour_profile'/f'{session}.pkl')
            valid_trials = np.where((~np.isnan(beh['reward_times']))&
                                    (np.array(beh['non_stop_trials'])==0)&
                                    (np.array(beh['non_fullstop_trials'])==0),
                                    True, False)
            
            smp_rate = 1000
            max_len = 8 * smp_rate
            full_trial_len = [np.nan] # run_onset to run_onset
            for t in range(1, len(beh['speed_times_aligned'])-1):
                if len(beh['speed_times_aligned'][t])>0 and len(beh['speed_times_aligned'][t+1])>0:
                    full_trial_len.append(beh['speed_times_aligned'][t+1][0][0]-beh['speed_times_aligned'][t][0][0])
                else:
                    full_trial_len.append(np.nan)
            full_trial_len.append(np.nan)
            
            speed_len = [len(tr_speed) if len(tr_speed)>0 else np.nan for tr_speed in beh['speed_times_aligned']]
            speed_time = [zero_padding(np.vstack(tr_speed)[:, 1], max_len) if len(tr_speed) > 0 else np.full(max_len, np.nan)
                          for tr_speed in beh['speed_times_aligned']
                        ]
            speed_time = np.vstack(speed_time)
            mean_speed_time = np.nanmean(speed_time, axis=0)
            
            lick_idx = np.nanmean(beh['lick_selectivities'])
            lick_dis = beh['lick_distances_aligned'][1:]
            lick_time = beh['lick_times_aligned']
            
            first_lick_times = []
            for licks in lick_time:
                if type(licks)==float and np.isnan(licks):
                    first_lick_times.append(np.nan)
                else:
                    licks_filtered = [l for l in licks if l>1000]
                    first_lick_times.append(licks_filtered[0] if len(licks_filtered)>0 else np.nan)
    
            # log in behaviour info
            df_sess_info.at[session, 'lick_idx'] = lick_idx
            df_sess_info.at[session,'first_lick_times'] = first_lick_times
            df_sess_info.at[session, 'mean_speed'] = np.nanmean(speed_time, axis=-1)
            df_sess_info.at[session, 'max_speed'] = np.nanpercentile(speed_time, 99, axis=-1)
            df_sess_info['max_speed_median'] = df_sess_info['max_speed'].apply(lambda x: np.nanmedian(x))
            df_sess_info['mean_speed_median'] = df_sess_info['mean_speed'].apply(lambda x: np.nanmedian(x))   
            df_sess_info.at[session, 'full_trial_len'] = full_trial_len
            df_sess_info.at[session, 'speed_len'] = speed_len
            df_sess_info['full_trial_len_median'] = df_sess_info['full_trial_len'].apply(lambda x: np.nanmedian(x))
            df_sess_info['speed_len_median'] = df_sess_info['speed_len'].apply(lambda x: np.nanmedian(x))
            df_sess_info.at[session, 'valid_trials'] = valid_trials
            df_sess_info['valid_trials_num'] = df_sess_info['valid_trials'].apply(lambda x: np.nansum(x)) 
            df_sess_info['valid_trials_perc'] = df_sess_info['valid_trials'].apply(lambda x: np.nansum(x)/len(x)) 
            
            if plot:
                # plot review of lick and speed
                fig, (ax1, ax2) = plt.subplots(1,2,dpi=150, figsize=(8,3))
                xaxis = np.arange(max_len)/smp_rate
                ax1.plot(xaxis, speed_time.T, c='grey', alpha=.1, lw=1)
                ax1.plot(xaxis, mean_speed_time, c='green', lw=2)
                ax1.set(xlabel='time (s)', ylabel='speed (cm/s)',
                        ylim=(0,np.nanpercentile(speed_time,99.5)+5),
                        xlim=(0,8),
                        )
                # plot lick profile
                lick_dis_flat = np.hstack(lick_dis)
                ax2.hist(lick_dis_flat,bins=100, density=True, cumulative=False, alpha=.7)
                ax2.set(title='lick_idx={}'.format(round(lick_idx,4)),
                        xlabel='distance (cm)',
                        # ylabel='lick histgram'
                        xlim=(0,220)
                        )
            
                fig.tight_layout()
                fig.suptitle(r'{} max_speed_median={:.2f} mean_speed_median={:.2f}'.format(
                    session, 
                    df_sess_info.loc[session,'max_speed_median'],
                    df_sess_info.loc[session,'mean_speed_median']), 
                    y=1, size=12)
                plt.savefig(f_out_plot+r'\{}_behaviour.png'.format(session), dpi=150, bbox_inches='tight')
                # plt.show()
                plt.close()        
    
        df_sess_info.to_parquet(p_df_out)
        # df_sess_info.to_csv(p_df_out.replace('.parquet', '.csv'))
    #%% calculate fov trace correlation for sessions pass behaviour check
    df_session_info = pd.read_parquet(p_df_out)
    # only calculate fov signal & speed correlation for sessions pass the behaviour check 
    df_session_selected = df_session_info.loc[(df_session_info['lick_idx']>MIN_LICK_IDX)&
                                  (df_session_info['max_speed_median']>MIN_MAX_SPEED)&
                                  (df_session_info['valid_trials_num']<200)&
                                  (df_session_info['valid_trials_perc']>MIN_PERC_VALID)&
                                  (df_session_info['full_trial_len_median']<MAX_TRIAL_LENGTH)
                                  ] 
    rec_lst = df_session_selected.index.tolist()
    error_lst = []
    print('extracting signal & speed correlation info...')
    # correlation on full trace
    df_sess_info['rg_corr_full_trial_single_trial_r'] =  pd.Series(dtype='object')
    df_sess_info['rg_corr_full_trial_single_trial_p'] =  pd.Series(dtype='object')
    for rec in tqdm(rec_lst):
        # if not os.path.exists(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\regression_res\{rec}"):
        #     continue
        print(rec)
        beh = pd.read_pickle(OUT_DIR_RAW_DATA/'behaviour_profile'/f'{rec}.pkl')
        anm, date, ss = rec.split('-')
        p_rec = r"Z:\Jingyu\2P_Recording\{}\{}\{}".format(rec[:5], 
                                                            rec[:-3], 
                                                            rec[-2:],
                                                            )
        if exp == r'Dbh_dlight':
            p_data = p_rec+r'\RegOnly\suite2p\plane0'
        elif exp == r'geco_dlight':
            p_data = p_rec+r'\nonrigid_reg_geco\suite2p\plane0'
        
        suite2p_ops = np.load(p_data+r'\ops.npy', allow_pickle=True).item()
        nframes = suite2p_ops['nframes']
        mean_imgaes = [suite2p_ops['meanImg'], suite2p_ops['meanImg_chan2']]
        
        p_green_fov_trace = OUT_DIR_RAW_DATA/'fov_trace'/ f'{rec}_fov_trace_dlight.npy'
        p_red_fov_trace = OUT_DIR_RAW_DATA/'fov_trace'/ f'{rec}_fov_trace_red.npy'
        
        try:
            if os.path.exists(p_green_fov_trace):
                raw_dlight_traces = np.load(p_green_fov_trace)
            else:
                dlight_mask = generate_adaptive_membrane_mask(mean_imgaes[ch_dlight], visualize=0)
                mov = load_bin_file(p_data, dlight_mov_bin, nframes, height=512, width=512)
                raw_dlight_traces = fov_trace_extraction(mov, dlight_mask, p_green_fov_trace)
                del mov
            if os.path.exists(p_red_fov_trace):
                red_traces = np.load(p_red_fov_trace)
            else:
                mov = load_bin_file(p_data, red_mov_bin, nframes, height=512, width=512)
                if exp == r'Dbh_dlight':
                    red_mask = np.load(OUT_DIR_RAW_DATA/rf"regression_res\{rec}\masks\dilated_global_axon_k=0.npy")
                elif exp == r'geco_dlight':
                    red_mask = generate_adaptive_membrane_mask(mean_imgaes[ch_red], visualize=0)
                    
                red_traces = fov_trace_extraction(mov, red_mask, p_red_fov_trace)
                del mov
                
            # smoothing
            raw_dlight_traces = gaussian_filter1d(raw_dlight_traces, sigma=1, axis=-1)
            red_traces = gaussian_filter1d(red_traces, sigma=1, axis=-1)
        
            # align to run-onset
            alignment = 'run'; bef, aft = 1, 4
            raw_dlight_traces_aligned = align_trials(raw_dlight_traces, alignment, beh, bef, aft)
            red_traces_aligned = align_trials(red_traces, alignment, beh, bef, aft)
        
            # raw_dlight_mean = np.nanmean(raw_dlight_traces_aligned, axis=0)
            # red_traces_mean = np.nanmean(red_traces_aligned, axis=0)
        
            corr_window = [-bef, aft]
            raw_dlight_tarces_window = raw_dlight_traces_aligned[:, int((bef+corr_window[0])*30):int((bef+corr_window[1])*30)]
            red_dlight_tarces_window = red_traces_aligned[:, int((bef+corr_window[0])*30):int((bef+corr_window[1])*30)]
            
            # full_trial_mean_trace correlation
            raw_dlight_window_mean = np.nanmean(raw_dlight_tarces_window, axis=0)
            red_traces_window_mean = np.nanmean(red_dlight_tarces_window, axis=0)
            # normalizatiion
            raw_dlight_window_mean_nor = normalize(raw_dlight_window_mean)
            red_traces_window_mean_nor = normalize(red_traces_window_mean)
            corr = pearsonr(raw_dlight_window_mean_nor, red_traces_window_mean_nor)
            df_sess_info.loc[rec, 'rg_corr_full_trial_mean_trace_r'] = corr[0]
            df_sess_info.loc[rec, 'rg_corr_full_trial_mean_trace_p'] = corr[1]
            
            # full_trial_trace single trial correlation
            # normalization
            raw_dlight_tarces_window_nor = normalize(raw_dlight_tarces_window)
            red_tarces_window_nor = normalize(red_dlight_tarces_window)
            corr_single_trial = pearsonr(raw_dlight_tarces_window_nor, red_tarces_window_nor, axis=-1)
            df_sess_info.at[rec, 'rg_corr_full_trial_single_trial_r'] = corr_single_trial[0]
            df_sess_info.at[rec, 'rg_corr_full_trial_single_trial_p'] = corr_single_trial[1]
            single_trial_corr_r_median = np.nanmedian(corr_single_trial[0])
            single_trial_corr_p_median = np.nanmedian(corr_single_trial[1])
            df_sess_info.at[rec, 'rg_corr_full_trial_single_trial_r_median'] = single_trial_corr_r_median
            df_sess_info.at[rec, 'rg_corr_full_trial_single_trial_p_median'] = single_trial_corr_p_median
            
            # correlation with speed
            # upsample dlight trace to 1000 hz
            fs_old = 30
            fs_new = 1000
            n_old = raw_dlight_traces.shape[-1]
            duration = n_old / fs_old
            # time vectors
            t_old = np.arange(n_old) / fs_old
            t_new = np.arange(round(n_old * fs_new / fs_old)) / fs_new
            # dlight trace
            raw_dlight_tarces_1000 = np.interp(t_new, t_old, raw_dlight_traces) # linear interpolation
            # upsample frame times
            frame_times_1000 = np.interp(t_new, t_old, beh['frame_times'][:n_old])
            # find new run_onset index
            run_onsets_times = beh['run_onsets']
            
            run_frames_1000 = []
            for run in run_onsets_times:
                if np.isnan(run):
                    run_frames_1000.append(0)
                else:
                    nearest_frame_idx = np.argmin(np.abs(frame_times_1000 - run))
                    run_frames_1000.append(nearest_frame_idx)
            
            # align upsampled trace to run
            bef, aft = 0, 4
            win_frames = int((bef+aft)*fs_new)
            tot_trial = len(run_onsets_times)
            aligned_signal = np.zeros((tot_trial, win_frames))
            for t in range(tot_trial):
                curr_trace = raw_dlight_tarces_1000[run_frames_1000[t]-int(bef*fs_new):run_frames_1000[t]+int(aft*fs_new)]
                if curr_trace.shape[0]<win_frames or run_frames_1000[t]==0:
                    aligned_signal[t,:]=np.nan
                else:
                    aligned_signal[t,:]=curr_trace
            raw_dlight_tarces_1000_aligned = aligned_signal     
            # raw_dlight_tarces_1000_aligned = utl.align_trials(raw_dlight_tarces_1000, alignment, beh, bef, aft, fs=fs_new)
            
            aligned_trials = ~np.isnan(raw_dlight_tarces_1000_aligned).any(axis=1)
            aligned_trials_idx = np.where(aligned_trials)[0]
            raw_dlight_traces_1000_mean = np.nanmean(raw_dlight_tarces_1000_aligned[aligned_trials], axis=0)
            # calculate spead trial mean
            speeds = [np.vstack(beh['speed_times_aligned'][i])[:4000, 1] for i in aligned_trials_idx]
            speeds = np.stack([zero_padding(speed, 4000) for speed in speeds])
            speed_mean = np.nanmean(speeds, axis=0)
            
            # single trial correlation speed-dlight
            x = raw_dlight_tarces_1000_aligned[aligned_trials]
            y = speeds
            single_trial_corr = pearsonr(x, y, axis=-1)
            print(np.nanmedian(single_trial_corr[0]))
            df_sess_info.at[rec, 'speed_corr_single_trial_r_median'] = np.nanmedian(single_trial_corr[0])
            
            # cross-correlation of fov_dlight and speed
            x = speed_mean
            y = raw_dlight_traces_1000_mean
            # normalize
            x = (x - np.mean(x)) / np.std(x)
            y = (y - np.mean(y)) / np.std(y)
            
            # compute cross-correlation
            corr = np.correlate(x, y, mode='full')
            lags = np.arange(-len(x)+1, len(x))
            
            # normalize to correlation coefficient
            corr /= len(x)
            
            df_sess_info.at[rec, 'speed_corr_peak_lag'] = lags[np.argmax(corr)]/1000
            df_sess_info.at[rec, 'speed_corr_peak']     = np.nanmax(corr)
            df_sess_info.at[rec, 'speed_lag0_corr=']    = corr[lags == 0][0]
        
        except Exception as e:
            print(f"Error processing {rec}: {e}")
            error_lst.append({'rec': rec, 'error': str(e)})
            df_sess_info.loc[rec, 'rg_corr_full_trial_mean_trace_r'] = np.nan
            df_sess_info.loc[rec, 'rg_corr_full_trial_mean_trace_p'] = np.nan
            df_sess_info.at[rec, 'rg_corr_full_trial_single_trial_r_median'] = np.nan
            df_sess_info.at[rec, 'rg_corr_full_trial_single_trial_p_median'] = np.nan
            df_sess_info.at[rec, 'speed_corr_single_trial_r_median'] = np.nan
            df_sess_info.at[rec, 'speed_corr_peak_lag'] = np.nan
            df_sess_info.at[rec, 'speed_corr_peak']  = np.nan 
            df_sess_info.at[rec, 'speed_lag0_corr='] = np.nan
            
    #%% save meta info dataframe of selected sessions
    df_sess_info.to_parquet(p_df_out)
    df_sess_info.to_csv(p_df_out.with_suffix('.csv'), index=False)
    
    # save error list
    if error_lst:
        df_errors = pd.DataFrame(error_lst)
        p_error_out = p_df_out.replace('.parquet', '_errors.csv')
        df_errors.to_csv(p_error_out.with_suffix('.csv'), index=False)
        print(f"Errors saved to {p_error_out}")