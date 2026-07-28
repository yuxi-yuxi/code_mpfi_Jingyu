# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:26:44 2026

@author: Jingyu Cao

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from place_cell_analysis import place_cell_functions as pcf

path = r"Z:\Jingyu\GCaMP_drug_infusion\place_cell_dataframe_3rsd\AC989-20250711-02_place_cell_dataframe_test.parquet"
df = pd.read_parquet(path)
tentative = [[np.asarray(m, dtype=bool) for m in row] for row in df['tentative_field']]
pf_map = np.vstack(df['place_field_map'].apply(np.asarray).values)
tcount = np.vstack(df['transient_count_per_bin'].apply(np.asarray).values)
occ = np.vstack(df['occupancy_per_bin'].apply(np.asarray).values)
#%%
new_final = pcf.filter_tentative_fields(df,
                                        min_peak_dff=0.1,
                                        min_in_out_ratio=2,
                                        min_width_cm=12, 
                                        min_transient_fraction=0.15)

n_tentative_pcs = len([i for i in tentative if len(i)>0])
n_final_pcs = len([i for i in new_final if len(i)>0])
print(n_tentative_pcs, n_final_pcs)

#%% Plot heatmap of place cell sequence sorted by place_field_position_cm
# Identify cells with at least one final place field
pc_mask = np.array([len(fields) > 0 for fields in new_final])
pc_indices = np.where(pc_mask)[0]

# Get place_field_map and place_field_position_cm for place cells
pf_map_pc = pf_map[pc_indices]
position_cm = df['place_field_position_cm'].values[pc_indices]

# Sort by place_field_position_cm
sort_order = np.argsort(position_cm)
pf_map_sorted = pf_map_pc[sort_order]

# Normalize each row to [0, 1]
row_max = np.nanmax(pf_map_sorted, axis=1, keepdims=True)
row_max[row_max == 0] = 1  # avoid division by zero
pf_map_norm = pf_map_sorted / row_max

# Plot
bin_size = 4  # cm per bin
n_bins = pf_map_norm.shape[1]
x_extent = n_bins * bin_size  # total track length in cm

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(pf_map_norm, aspect='auto', cmap='grey',
               extent=[0, x_extent, pf_map_norm.shape[0], 0],
               interpolation='nearest')
ax.set_xlabel('Position (cm)')
ax.set_ylabel('Cell # (sorted by peak position)')
ax.set_title(f'Place Cell Sequence (n = {n_final_pcs})')
plt.colorbar(im, ax=ax, label='Normalized dF/F')
plt.tight_layout()
plt.show()

