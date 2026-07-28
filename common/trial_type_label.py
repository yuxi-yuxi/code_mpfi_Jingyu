# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:59:45 2026

@author: Jingyu Cao

generate behavioural trial label for each session:
- ealy/late lick trials (distance or time)
- speed matched ealy/late lick trials (distance or time)
- valid trials (reward&(~non_stop)&(~non_full_stop)&((t_rew - t_run)<10s))
- stim trials

for place cells, the trial numbers need to map to lap index 
"""

import numpy as np 
from pathlib import Path
import pandas as pd
from common.trial_selection import seperate_valid_trial
from common.utils_behaviour import speed_match, extract_first_licks

rec_id = ''
p_beh = ''
df_beh = pd.read_pickle(p_beh)
beh_valid_trials = seperate_valid_trial(df_beh, time_thresh=10000)