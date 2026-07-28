# -*- coding: utf-8 -*-
"""
Created on Tue Jun 09 2026

Interactive response map with clickable ROIs for time cell / place cell analysis.
Single-session variant of `plot_html_overview_single_session.py`.

For each recording id ('anm-date-ss'):
- Left panel: FOV response map (clickable) + behaviour
- Right column: trace + per-lap + sequence + stability + sig-stability
- Top controls: mode switch (time / place) + field criteria inputs

Generates one standalone HTML file per session.

@author: Jingyu Cao
"""

from pathlib import Path
import numpy as np
import pandas as pd
import html_plotting_function as pf
from config_place_cell_geco import CONFIG_TIME, CONFIG_PLACE

#%% PARAMETERS
OUT_DIR_RAW_DATA = CONFIG_TIME['out_dir_raw_data']
BEH_DIR = Path(r"Z:\Jingyu\dlight_learning\geco_dlight\behaviour_profile")
OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'interactive_plots' / '3rsd_single'
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

# Time cell axis
time_bin_size = 0.1
max_lap_duration_s = 6.0
frame_rate = 30
n_time_bins = int(max_lap_duration_s / time_bin_size)
xaxis_time = np.arange(n_time_bins) * time_bin_size

# Place cell axis
track_length = 180  # cm
pos_bin_size = 4    # cm
n_pos_bins = int(track_length / pos_bin_size)
xaxis_place = np.arange(n_pos_bins) * pos_bin_size

# Session ids: 'anm-date-ss'
rec_lst = [
# 'AC327-20260602-02',     
# 'AC330-20260602-02',

# 'AC327-20260603-02',     
# 'AC330-20260603-02', 

# 'AC327-20260604-02',     
# 'AC330-20260604-02', 

# 'AC327-20260605-02',     
# 'AC330-20260605-02', 

# 'AC327-20260606-02',     
# 'AC330-20260606-02', 

# 'AC327-20260607-02',     
# 'AC330-20260607-02',  

# 'AC327-20260608-02',     
# 'AC330-20260608-02',

# 'AC327-20260609-02',     
# 'AC330-20260609-02',

# 'AC327-20260610-02',     
# 'AC330-20260610-02',

'AC327-20260611-02',     
'AC330-20260611-02', 

'AC327-20260612-02',     
'AC330-20260612-02',   
]

#%% Main loop
if __name__ == '__main__':
    error_lst = []

    for rec_id in rec_lst:
        print(f"\nProcessing {rec_id}...")

        try:
            mean_img = pf.load_mean_image(rec_id, CONFIG_TIME)
            roi_stat, active_soma_indices = None, None
            try:
                roi_stat, active_soma_indices = pf.load_roi_stat(rec_id, CONFIG_TIME, rec_id)
                print(f"  Loaded {len(roi_stat)} ROIs from gcamp_stats.npy, "
                      f"{len(active_soma_indices)} active somas")
            except FileNotFoundError as e:
                print(f"  Warning: {e}")

            time_data = pf._load_mode_for_single_session(
                rec_id, CONFIG_TIME, xaxis_time,
                roi_stat, active_soma_indices,
            )
            place_data = pf._load_mode_for_single_session(
                rec_id, CONFIG_PLACE, xaxis_place,
                roi_stat, active_soma_indices,
            )

            if time_data is None and place_data is None:
                print(f"  Skipping {rec_id}: no usable dataframes.")
                error_lst.append(rec_id)
                continue

            # Load behaviour pickle for this session
            beh_path = BEH_DIR / f"{rec_id}.pkl"
            behaviour = None
            if beh_path.exists():
                try:
                    behaviour = pd.read_pickle(beh_path)
                    print(f"  Loaded behaviour: {beh_path.name}")
                except Exception as e:
                    print(f"  Warning: Could not load behaviour {beh_path.name}: {e}")

            save_path = OUT_DIR_FIG / f"{rec_id}_interactive_time_place_cell.html"
            pf.generate_interactive_html_single_session(
                rec_id=rec_id,
                mean_img=mean_img,
                time_data=time_data,
                place_data=place_data,
                save_path=save_path,
                behaviour_data=behaviour,
            )

        except Exception as e:
            print(f"  Error processing {rec_id}: {e}")
            import traceback
            traceback.print_exc()
            error_lst.append(rec_id)

    if error_lst:
        print(f"\nErrors occurred for: {error_lst}")
    else:
        print("\nAll sessions processed successfully!")
