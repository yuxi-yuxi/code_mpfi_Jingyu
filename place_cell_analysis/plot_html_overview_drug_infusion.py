# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 2026

Interactive response map with clickable ROIs for time cell / place cell analysis.

Two sessions per date (ss1='02', ss2='04'), same FOV.
- Left panel: shared FOV response map (clickable)
- Middle / right columns: per-session trace + per-lap + sequence + stability
- Top controls: mode switch (time / place) + info-threshold input
- Only ROIs present in BOTH sessions' dataframes are shown.

Generates standalone HTML files with embedded JavaScript.

@author: Jingyu Cao
"""

from pathlib import Path
import numpy as np
import pandas as pd
import html_plotting_function as pf
from config import CONFIG_TIME, CONFIG_PLACE
#%% PARAMETERS
OUT_DIR_RAW_DATA = CONFIG_TIME['out_dir_raw_data']
BEH_DIR = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\behaviour_profile")
OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'interactive_plots'/'2.5rsd'
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

# Time cell thresholds + axis
TI_threshold = 0.2
shuff_TI_thresh = None
time_bin_size = 0.1
max_lap_duration_s = 6.0
frame_rate = 30
n_time_bins = int(max_lap_duration_s / time_bin_size)
xaxis_time = np.arange(n_time_bins) * time_bin_size

# Place cell thresholds + axis
SI_threshold = 0.2
shuff_SI_thresh = None
track_length = 180  # cm
pos_bin_size = 4    # cm
n_pos_bins = int(track_length / pos_bin_size)
xaxis_place = np.arange(n_pos_bins) * pos_bin_size

# Sessions per date
SS1_CODE = '02'
SS2_CODE = '04'

# rec_lst contains DATE-level ids like 'AC989-20250711'.
# For each, we load sessions SS1_CODE and SS2_CODE.
rec_lst = ['AC989-20250711', ]
# Session list
drug = 'SCH'
# drug = 'prazosin'
# drug = 'propranolol'

from drug_infusion import rec_lst_infusion as recs
if drug == 'SCH':
    rec_drug = recs.rec_SCH
    rec_ctrl = recs.rec_SCH_ctrl
elif drug == 'prazosin':
    rec_drug = recs.rec_praz
    rec_ctrl = recs.rec_praz_ctrl
elif drug == 'propranolol':
    rec_drug = recs.rec_prop
    rec_ctrl = recs.rec_prop_ctrl

#%% Main loop
if __name__ == '__main__':
    error_lst = []

    for rec_date in rec_lst:
    # for _, rec in rec_drug.iterrows():
    #     anm = rec['anm']
    #     date = rec['date']
    #     rec_date = f'{anm}-{date}'
        
        print(f"\nProcessing {rec_date} (ss1={SS1_CODE}, ss2={SS2_CODE})...")
        rec_ss1 = f'{rec_date}-{SS1_CODE}'
        rec_ss2 = f'{rec_date}-{SS2_CODE}'

        try:
            mean_img = pf.load_mean_image(rec_ss1, CONFIG_TIME)
            roi_stat, active_soma_indices = None, None
            try:
                roi_stat, active_soma_indices = pf.load_roi_stat(rec_ss1, CONFIG_TIME)
                print(f"  Loaded {len(roi_stat)} ROIs from gcamp_stats.npy, {len(active_soma_indices)} active somas")
            except FileNotFoundError as e:
                print(f"  Warning: {e}")

            time_data = pf._load_mode_for_both_sessions(
                rec_ss1, rec_ss2, CONFIG_TIME, xaxis_time,
                roi_stat, active_soma_indices,
            )
            place_data = pf._load_mode_for_both_sessions(
                rec_ss1, rec_ss2, CONFIG_PLACE, xaxis_place,
                roi_stat, active_soma_indices,
            )

            if time_data is None and place_data is None:
                print(f"  Skipping {rec_date}: no usable dataframes.")
                error_lst.append(rec_date)
                continue

            # Load behaviour data for both sessions
            behaviour_data = {}
            for ss_key, ss_code in [('ss1', SS1_CODE), ('ss2', SS2_CODE)]:
                beh_path = BEH_DIR / f"{rec_date}-{ss_code}.pkl"
                if beh_path.exists():
                    try:
                        behaviour_data[ss_key] = pd.read_pickle(beh_path)
                        print(f"  Loaded behaviour: {beh_path.name}")
                    except Exception as e:
                        print(f"  Warning: Could not load behaviour {beh_path.name}: {e}")
                        behaviour_data[ss_key] = None
                else:
                    behaviour_data[ss_key] = None

            save_path = OUT_DIR_FIG / f"{rec_date}_interactive_time_place_cell.html"
            pf.generate_interactive_html(
                rec_date=rec_date,
                mean_img=mean_img,
                time_data=time_data,
                place_data=place_data,
                save_path=save_path,
                session_labels={'ss1': SS1_CODE, 'ss2': SS2_CODE},
                behaviour_data=behaviour_data if any(v is not None for v in behaviour_data.values()) else None,
            )

        except Exception as e:
            print(f"  Error processing {rec_date}: {e}")
            import traceback
            traceback.print_exc()
            error_lst.append(rec_date)

    if error_lst:
        print(f"\nErrors occurred for: {error_lst}")
    else:
        print("\nAll sessions processed successfully!")
