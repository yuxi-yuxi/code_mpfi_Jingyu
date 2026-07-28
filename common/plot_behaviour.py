# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:47:47 2026

@author: Jingyu Cao
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from common.utils_basic import zero_padding
from common.utils_behaviour import extract_speed_trace, extract_first_licks
import common.plotting_functions_Jingyu as pf
f_beh = pd.read_pickle(r"Z:\Jingyu\GCaMP_learning\behaviour_profile\AC322-20260505-02.pkl")
block_numbers = np.array(f_beh['block_numbers'])
# unique blocks (preserves order)
blocks = np.unique(block_numbers)
# dict: block → trial indices
block_to_idx = {b: np.where(block_numbers == b)[0] for b in blocks}

#%%
align_by='distance'
speed_aligned = extract_speed_trace(f_beh, align_by)
first_licks = extract_first_licks(f_beh, align_by)

speed_by_block = {
    int(b): [zero_padding(speed_aligned[i], 2200) for i in idx]
    for b, idx in block_to_idx.items()
}

licks_by_block = {
    int(b): [first_licks[i] for i in idx]
    for b, idx in block_to_idx.items()
}


for b, _ in block_to_idx.items():
    speed_array = np.stack(speed_by_block[b])
    fig, ax = plt.subplots(dpi=300)
    # ax.plot(speed_array.T, lw=1, ls='--', color='grey')
    pf.plot_mean_trace(speed_array, ax, color='steelblue', lw=2)
    
    fls = np.hstack(licks_by_block[b])
    fig, ax = plt.subplots(figsize=(1, 2), dpi=200)
    ax.hist(fls, bins=10, range=(0, 220))

#%%    
align_by='time'
speed_aligned = extract_speed_trace(f_beh, align_by)
first_licks = extract_first_licks(f_beh, align_by)

speed_by_block = {
    int(b): [zero_padding(speed_aligned[i], 5000) for i in idx]
    for b, idx in block_to_idx.items()
}

licks_by_block = {
    int(b): [first_licks[i] for i in idx]
    for b, idx in block_to_idx.items()
}


for b, _ in block_to_idx.items():
    speed_array = np.stack(speed_by_block[b])
    speed_array = gaussian_filter1d(speed_array, sigma=50, axis=-1)
    fig, ax = plt.subplots(figsize=(2,2),dpi=300)
    # ax.plot(speed_array.T, lw=1, ls='--', color='grey')
    # pf.plot_mean_trace(speed_array, ax, color='steelblue', lw=2)
    ax.imshow(speed_array, cmap='Greys', aspect='auto', interpolation='none')
    
    fls = np.hstack(licks_by_block[b])
    fig, ax = plt.subplots(figsize=(1, 2), dpi=200)
    ax.hist(fls, bins=10, range=(0, 5000))    