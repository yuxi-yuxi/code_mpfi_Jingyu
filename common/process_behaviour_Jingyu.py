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


if (r"Z:\Dinghao\code_mpfi_dinghao\utils" in sys.path) == False:
    sys.path.append(r"Z:\Dinghao\code_mpfi_dinghao\utils")
# import pre-processing functions 
import behaviour_functions as bf
#%% params
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\rdlight_raw_data")

OUT_DIR = OUT_DIR_RAW_DATA / 'behaviour_profile'
if not OUT_DIR.exists():
    OUT_DIR.mkdir()
    
rec_lst = [
'AC319-20260319-02',
'AC319-20260319-04',
'AC319-20260320-02',
'AC319-20260320-04',
'AC319-20260320-06',       
    
    ]
#%% main 
for recname in rec_lst:
    
    out_f = OUT_DIR/ f'{recname}.pkl'
    
    # if not os.path.exists(out_f):
    print('processing {}...'.format(recname))
    txt_path = r"Z:\Jingyu\mice-expdata\{}\A{}T.txt".format(recname[0:5],recname[2:])
    if not os.path.exists(txt_path): # check if behaviour txt file for this session exists
        # df_anm_info.loc[recname, 'text_file']=-1
        print('no txt file!!!')
        
    behavioural_data = bf.process_behavioural_data_imaging(txt_path)
    with open(out_f, 'wb') as f:
        pickle.dump(behavioural_data, f)