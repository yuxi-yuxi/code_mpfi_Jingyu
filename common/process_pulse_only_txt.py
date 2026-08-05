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


def process_imaging_block_pulse_session(txtfile):
    '''Parse imaging/block/pulse-only logs that have no behavioural trials.

    ``frame_times`` uses the ``$FM,...,0`` edge, as does the existing imaging
    parser. For these sensor logs, ``$PC`` is the leading pulse edge and
    ``$PO`` is the trailing edge, so they are exposed as pulse starts and
    ends. Raw split statements are retained for block and pulse metadata.
    '''
    overflow_period_ms = (2**32 - 1) / 1000
    wrap_offset_ms = 0.0
    previous_raw_time = None

    frames = []
    frame_rising_times = []
    frame_falling_times = []
    block_starts = []
    block_ends = []
    block_transitions = []
    pulse_commands = []
    pulse_trains = []
    pulse_records = []
    pc_times = []
    po_times = []
    train_start_times = []
    train_end_times = []
    session_starts = []
    session_ends = []
    malformed_lines = []

    with open(txtfile, 'r') as file:
        for line_number, raw_line in enumerate(file, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            statement = stripped.split(',')
            if len(statement) < 2:
                malformed_lines.append((line_number, stripped))
                continue
            try:
                raw_time = float(statement[1])
            except ValueError:
                malformed_lines.append((line_number, stripped))
                continue

            # A true Teensy wrap is a ~4.3e6-ms drop. The threshold prevents
            # small out-of-order writes from being mistaken for a wrap.
            if (previous_raw_time is not None
                    and previous_raw_time - raw_time > overflow_period_ms / 2):
                wrap_offset_ms += overflow_period_ms
            previous_raw_time = raw_time
            timestamp = raw_time + wrap_offset_ms
            statement[1] = str(timestamp)
            label = statement[0]

            if label == '$FM':
                frames.append(statement)
                if len(statement) >= 3 and statement[2] == '1':
                    frame_rising_times.append(timestamp)
                elif len(statement) >= 3 and statement[2] == '0':
                    frame_falling_times.append(timestamp)
            elif label == '$BT':
                block_starts.append(statement)
                block_transitions.append(statement)
            elif label == '$BE':
                block_ends.append(statement)
                block_transitions.append(statement)
            elif label == '$PP':
                pulse_commands.append(statement)
                pulse_records.append(statement)
            elif label == '$PT':
                pulse_trains.append(statement)
                pulse_records.append(statement)
                if len(statement) >= 3 and statement[2] == '1':
                    train_start_times.append(timestamp)
                elif len(statement) >= 3 and statement[2] == '0':
                    train_end_times.append(timestamp)
            elif label == '$PC':
                pc_times.append(timestamp)
                pulse_records.append(statement)
            elif label == '$PO':
                po_times.append(timestamp)
                pulse_records.append(statement)
            elif label == '$ST':
                session_starts.append(statement)
            elif label == '$TE':
                session_ends.append(statement)

    if not frame_falling_times:
        raise ValueError(f'No $FM,...,0 imaging-frame events found in {txtfile}')

    return {
        'frame_times': frame_falling_times,
        'frame_rising_times': frame_rising_times,
        'frame_falling_times': frame_falling_times,
        'frame_statements': frames,
        'block_statements': block_starts,
        'block_start_statements': block_starts,
        'block_end_statements': block_ends,
        'block_transition_statements': block_transitions,
        'block_start_times': [float(x[1]) for x in block_starts],
        'block_end_times': [float(x[1]) for x in block_ends],
        'block_ids': [int(x[2]) for x in block_starts],
        'pulse_descriptions': pulse_commands,
        'pulse_command_statements': pulse_commands,
        'pulse_train_statements': pulse_trains,
        'pulse_statements': pulse_records,
        'pulse_train_start_times': train_start_times,
        'pulse_train_end_times': train_end_times,
        'pc_times': pc_times,
        'po_times': po_times,
        'pulse_start_times': pc_times,
        'pulse_end_times': po_times,
        'session_start_statements': session_starts,
        'session_end_statements': session_ends,
        'malformed_lines': malformed_lines,
    }
#%% params
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\raw_data\lc_stim_sensor\processed_data")

rec_lst = [

'AD192-20260729-02',
'AD193-20260730-02',
    ]

# from Rdlight_imaging.rec_lst import all_rec

#%% main
for recname in rec_lst:
    
    OUT_DIR = OUT_DIR_RAW_DATA/recname
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True)

    out_f = OUT_DIR/ f'{recname}.pkl'
    
    # if not os.path.exists(out_f):
    print('processing {}...'.format(recname))
    txt_path = r"Z:\Jingyu\mice-expdata\{}\A{}T.txt".format(recname[0:5],recname[2:])
    if not os.path.exists(txt_path): # check if behaviour txt file for this session exists
        # df_anm_info.loc[recname, 'text_file']=-1
        print('no txt file!!!')

    behavioural_data = process_imaging_block_pulse_session(txt_path)
    print('  frames={}, blocks={}, pulses={}'.format(
        len(behavioural_data['frame_times']),
        len(behavioural_data['block_statements']),
        len(behavioural_data['pulse_start_times']),
        ))

    with open(out_f, 'wb') as f:
        pickle.dump(behavioural_data, f)
