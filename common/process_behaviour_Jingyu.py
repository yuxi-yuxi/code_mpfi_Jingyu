# -*- coding: utf-8 -*-
"""
Created on Fri May 2 14:34:28 2025

@author: Jingyu Cao, Dinghao Luo
"""
#%% imports
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import pickle 


# if (r"Z:\Dinghao\code_mpfi_dinghao\utils" in sys.path) == False:
#     sys.path.append(r"Z:\Dinghao\code_mpfi_dinghao\utils")
# import pre-processing functions 
# import behaviour_functions as bf
import common.behaviour_functions_Jingyu as bf

# block-number helpers
def assign_block_numbers(trial_start_times, block_statements):
    '''
    assign a block number to each trial using $BT block-start markers.

    each $BT line carries an explicit block number at index 2, e.g.
    `$BT,2216010.160,1,30,...` -> block 1 starts at t=2216010.160 ms.
    a trial belongs to the block declared by the most recent $BT whose
    timestamp is <= the trial start time. trials preceding the first $BT
    (or all trials, if no $BT exists) inherit the first block's number,
    defaulting to 1 when there are no $BT markers at all.

    parameters:
    - trial_start_times: list of trial start timestamps (in ms), one per trial.
    - block_statements: list of overflow-corrected $BT lines.

    returns:
    - list[int]: block number for each trial, same length as trial_start_times.
    '''
    if not block_statements:
        return [1] * len(trial_start_times)
    markers = [(float(b[1]), int(b[2])) for b in block_statements]
    first_block = markers[0][1]
    block_numbers = []
    for t in trial_start_times:
        current = first_block
        for ts, blk in markers:
            if ts <= t:
                current = blk
            else:
                break
        block_numbers.append(current)
    return block_numbers

def add_block_numbers(behavioural_data):
    '''
    derive per-trial block numbers from behavioural_data and attach in place.
    '''
    trial_start_times = [float(x[1]) for x in behavioural_data['trial_statements']]
    behavioural_data['block_numbers'] = assign_block_numbers(
        trial_start_times, behavioural_data['block_statements']
        )
#%% params
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\raw_data\lc_stim_gcamp\processed_data")
    
rec_lst = [
# 'AC319-20260319-02',
# 'AC319-20260319-04',
# 'AC319-20260320-02',
# 'AC319-20260320-04',
# 'AC319-20260320-06',  
     
# 'AC322-20260428-02',
# 'AC322-20260429-02',
# 'AC322-20260430-02',
# 'AC322-20260501-02',
# 'AC322-20260502-02',
# 'AC322-20260503-02',
# 'AC322-20260504-02',
# 'AC322-20260505-02',
# 'AC322-20260505-04',
# 'AC322-20260506-02',
# 'AC322-20260506-04',
# 'AC322-20260507-02',

# 'AC333-20260717-02',
# 'AC333-20260720-02',
# 'AC333-20260720-04',
# 'AC333-20260721-02',
# 'AC333-20260721-04',
# 'AC333-20260723-02',
# 'AC333-20260724-02',
# 'AC333-20260724-04',
# 'AC333-20260725-02',
# 'AC333-20260725-04',
'AC333-20260726-02',
'AC333-20260726-04',

# 'AC334-20260724-02',
# 'AC334-20260725-02'
'AC334-20260726-02'

# 'AC324-20260527-02',
# 'AC324-20260528-02',
# 'AC324-20260529-02',
# 'AC324-20260530-02',
# 'AC324-20260531-02',

# 'AC325-20260527-02',
# 'AC325-20260528-02',
# 'AC325-20260529-02',
# 'AC325-20260530-02',
# 'AC325-20260531-02',

# 'AC326-20260527-02',
# 'AC326-20260528-02',
# 'AC326-20260529-02',
# 'AC326-20260530-02',
# 'AC326-20260531-02',

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

# 'AC327-20260611-02',     
# 'AC330-20260611-02', 

# 'AC327-20260612-02',     
# 'AC330-20260612-02',      
    ]

# from Rdlight_imaging.rec_lst import all_rec

#%% main
for recname in rec_lst:
    
    OUT_DIR = OUT_DIR_RAW_DATA/recname
    if not OUT_DIR.exists():
        OUT_DIR.mkdir()
    out_f = OUT_DIR/ f'{recname}.pkl'
    
    # if not os.path.exists(out_f):
    print('processing {}...'.format(recname))
    txt_path = r"Z:\Jingyu\mice-expdata\{}\A{}T.txt".format(recname[0:5],recname[2:])
    if not os.path.exists(txt_path): # check if behaviour txt file for this session exists
        # df_anm_info.loc[recname, 'text_file']=-1
        print('no txt file!!!')

    behavioural_data = bf.process_behavioural_data_imaging(txt_path)
    add_block_numbers(behavioural_data)

    with open(out_f, 'wb') as f:
        pickle.dump(behavioural_data, f)