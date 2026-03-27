# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 17:06:40 2025

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binomtest, ttest_1samp
from common import plotting_functions_Jingyu as pf
from dlight_imaging.regression.utils_regression import get_axon_grids
save_fig = pf.save_fig
pf.mpl_formatting()
#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
# OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_grid_free_dilation'
OUT_DIR_DF = OUT_DIR_RAW_DATA/'processed_dataframe_grid_free_dilation'
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_Dbh_dlight")
save_plot=1

dlight_pre  = (-1, 0)
dlight_post = (0, 1)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
pval_thresh = 0.05
null_chance = 1-(amp_shuff_thresh_up/100) # test against this probability
regression_name ='single_trial_regression'

p_pooled_df = OUT_DIR_DF / rf"df_population_profile_pooled_dilation=0_pre{dlight_pre}_post{dlight_post}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.parquet"
df_pool = pd.read_parquet(p_pooled_df)
#%%
all_shuffle_ups = []
all_real_ups = []
all_shuffle_ups_perc = []
all_real_ups_perc = []

session_binom_significant = []
n_valid_rois_all = []
n_dlight_up_all = []
pers_dlight_up = []

for rec_id, roi_stats in df_pool.groupby('rec_id', sort=False):
    axon_mask = np.load(OUR_DIR_REGRESS/rec_id/'masks'/'dilated_global_axon_k=0.npy')
    # axon_grids, pix_counts = get_axon_grids(axon_mask)
    roi_stats = roi_stats.reset_index(drop=True)
    
    if len(roi_stats) == 0: # skip session if no valid rois
        continue
    
    # select_valid_rois
    roi_stats_valid = roi_stats.loc[(roi_stats['dlight_valid'])&(roi_stats['red_valid'])&(~roi_stats['edge'])]

    
    n_valid_rois = len(roi_stats_valid)
    # n_valid_rois = len(axon_grids) # 2/12/26 Jingyu, change percentage quantification
    #                                 # use all grids containing any axon masks as denominator
    n_valid_rois_all.append(n_valid_rois)
    n_dlight_up = np.sum(roi_stats_valid['Up'])
    n_dlight_up_all.append(n_dlight_up)
    print(rec_id, n_dlight_up/n_valid_rois)
    pers_dlight_up.append(n_dlight_up/n_valid_rois) 
    pval_binom = binomtest(k=n_dlight_up, n=n_valid_rois, p=null_chance, alternative='greater').pvalue
    session_binom_significant.append(pval_binom<pval_thresh)

pval_ttest = ttest_1samp(pers_dlight_up, popmean=null_chance).pvalue
print(f'{np.sum(session_binom_significant)}/{len(session_binom_significant)} sessions are significant higher than shuffle {amp_shuff_thresh_up}%')
plt.hist(n_dlight_up_all, bins=35, range=(-5, 70), edgecolor='white')    
plt.show()
#%%

fig, ax, stats = pf.plot_overlay_1bar([], # no shuffle data 
                 100*np.array(pers_dlight_up),
                 colname=['shuffle_95', 'data'],
                 ylabel='% roi',
                 bar_width=0.15,
                 point_size=5,
                 jitter=0.03,
                 # ylim=(0, 45),
                 annotation=True)
ax.annotate(f'num_sig_sessions: {np.sum(session_binom_significant)}/{len(session_binom_significant)}\n(null_chance: {null_chance:.2f})',
            xy=(0.4, 20), size=8)
ax.annotate(f'ttest_pval: {pval_ttest:4f} vs\nnull_chance: {null_chance:.2f}',
            xy=(0.4, 15), size=8)
save_fig(fig, OUT_DIR_FIG, r'dlightUp_grid_number_signicance_ES={}_amp={}.pdf'
            .format(effect_size_thresh, amp_shuff_thresh_up), save=save_plot)

