# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:43:05 2026

@author: Jingyu Cao

Plot FOV tracking for each animal across sessions (5 sessions per figure)
"""
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from dlight_imagaing_learning.geco_dlight.recording_list import rec_lst
from common import plotting_functions_Jingyu as pf

#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\Dbh_dlight")
OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'TEST_PLOTS' / 'FOV_tracking'
if not OUT_DIR_FIG.exists():
    OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

SESSIONS_PER_FIG = 5

# Group recordings by animal
recs_by_animal = defaultdict(list)
for rec in rec_lst:
    recs_by_animal[rec[:5]].append(rec)

error_lst = []

for anm, anm_recs in recs_by_animal.items():
    # Collect FOV data for all sessions
    session_data = []

    for rec in tqdm(anm_recs[:5]):
        anm_name, date, ss = rec.split('-')
        p_suite2p = Path(rf"Z:\Jingyu\2P_Recording\{anm_name}\{anm_name}-{date}\{ss}\RegOnly\suite2p\plane0")

        if not p_suite2p.exists():
            error_lst.append(rec)
            continue

        ops_path = p_suite2p / 'ops.npy'
        if not ops_path.exists():
            error_lst.append(rec)
            continue

        suite2p_ops = np.load(ops_path, allow_pickle=True).item()

        mean_img_ch1 = suite2p_ops.get('meanImg', None)
        mean_img_ch2 = suite2p_ops.get('meanImg_chan2', None)

        if mean_img_ch1 is None:
            error_lst.append(rec)
            continue

        session_data.append({
            'rec': rec,
            'mean_img_ch1': mean_img_ch1,
            'mean_img_ch2': mean_img_ch2
        })

    if len(session_data) == 0:
        continue

    # Group sessions into figures of 5
    n_sessions = len(session_data)
    n_figs = int(np.ceil(n_sessions / SESSIONS_PER_FIG))

    for fig_idx in range(n_figs):
        start_idx = fig_idx * SESSIONS_PER_FIG
        end_idx = min(start_idx + SESSIONS_PER_FIG, n_sessions)
        sessions_in_fig = session_data[start_idx:end_idx]
        n_sess_in_fig = len(sessions_in_fig)

        # Create figure: 2 rows (ch1, ch2) x n_sessions columns
        fig, axs = plt.subplots(2, n_sess_in_fig, figsize=(4 * n_sess_in_fig, 8),
                                squeeze=False, dpi=150)
        fig.suptitle(f'{anm} - Sessions {start_idx + 1} to {end_idx}', fontsize=14, fontweight='bold')

        for col_idx, sess in enumerate(sessions_in_fig):
            rec = sess['rec']
            mean_img_ch1 = sess['mean_img_ch1']
            mean_img_ch2 = sess['mean_img_ch2']

            # Row 0: Channel 1 (green/functional)
            ax = axs[0, col_idx]
            ax.imshow(mean_img_ch1, cmap='gray',
                      vmin=np.nanpercentile(mean_img_ch1, 1),
                      vmax=np.nanpercentile(mean_img_ch1, 99))
            ax.set_title(rec, fontsize=10)
            ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel('Ch1', fontsize=12)

            # Row 1: Channel 2 (red/structural)
            ax = axs[1, col_idx]
            if mean_img_ch2 is not None:
                ax.imshow(mean_img_ch2, cmap='gray',
                          vmin=np.nanpercentile(mean_img_ch2, 1),
                          vmax=np.nanpercentile(mean_img_ch2, 98))
            else:
                ax.text(0.5, 0.5, 'No Ch2', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel('Ch2', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        fig_name = f'{anm}_FOV_sessions_{start_idx + 1}_to_{end_idx}.png'
        plt.show()
        # pf.save_fig(fig, OUT_DIR_FIG, fig_name, dpi=150, forms=['png'], save=1)
        print(f'Saved: {fig_name}')
        plt.close(fig)

    print(f'Completed {anm}: {n_sessions} sessions in {n_figs} figure(s)')

print(f'All animals processed! Errors: {len(error_lst)}')
if error_lst:
    print('Sessions with errors:', error_lst)
