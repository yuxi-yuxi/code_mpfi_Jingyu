# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:55:13 2026

@author: Jingyu Cao
"""
from pathlib import Path
#============================================================================
# CONFIGURATION
#============================================================================
CONFIG_TIME = {
    'out_dir_raw_data': Path(r"Z:\Jingyu\GCaMP_drug_infusion"),
    'df_subdir': 'time_cell_dataframe',
    'df_pattern': '{rec}_time_cell_dataframe.parquet',
    'suite2p_pattern': r"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\suite2p\plane0",
    'gcamp_stats_pattern': r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals\{anm}-{date}\gcamp_stats.npy",
    'soma_class_pattern': r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals\{anm}-{date}\soma_class.npz",
    'mean_img_key': 'meanImg',

    'cell_id_col': 'cell_id',
    'field_map_col': 'time_field_map_norm',
    'info_bits_col': 'temporal_information_bits',
    'shuffled_info_col': 'shuffled_TI',
    'peak_position_col': 'time_field_position_s',
    'peak_amplitude_col': 'time_field_peak_amplitude',

    'signal_label': 'Time Field',
    'x_label': 'Time (s)',
    'info_label': 'Temporal Info (bits)',
    'threshold_label': 'TI Threshold (bits):',
    'cell_type_label': 'Time Cell',
    'non_cell_type_label': 'Non-time Cell',
    'max_x_limit': 4.0,
}

CONFIG_PLACE = {
    **CONFIG_TIME,
    'df_subdir': 'place_cell_dataframe',
    'df_pattern': '{rec}_place_cell_dataframe_test.parquet',
    'field_map_col': 'place_field_map_norm',
    'info_bits_col': 'spatial_information_bits',
    'shuffled_info_col': 'shuffled_SI',
    'peak_position_col': 'place_field_position_cm',
    'peak_amplitude_col': 'place_field_peak_amplitude',
    'signal_label': 'Place Field',
    'x_label': 'Position (cm)',
    'info_label': 'Spatial Info (bits)',
    'threshold_label': 'SI Threshold (bits):',
    'cell_type_label': 'Place Cell',
    'non_cell_type_label': 'Non-place Cell',
    'max_x_limit': 180.0,
}


