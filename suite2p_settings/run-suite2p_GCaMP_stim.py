# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:30:51 2024

@author: Jingyu Cao

MAKE SURE TO RUN THIS SCRIPT IN DEFAULT SUITE2P ENVIRONMENT!!

"""
#%%
import sys
sys.path.append("Z:\Jingyu\Code\Python")
import anm_list_running as anm

import suite2p
from pathlib import Path
import os
import numpy as np
from contextlib import redirect_stdout
from shutil import copytree, ignore_patterns

#%%

sessions = [
# 'AC322-20260428-02',
# 'AC322-20260507-02',
# 'AC322-20260506-04',
# 'AC333-20260717-02',
# 'AC333-20260720-02',
# 'AC333-20260720-04',
# 'AC333-20260721-02',
# 'AC333-20260721-04',
# 'AC333-20260723-02',

# 'AC334-20260724-02',
# 'AC334-20260725-02',
'AC334-20260726-02',

# 'AC333-20260724-02',
# 'AC333-20260724-04',
# 'AC333-20260725-02',
# 'AC333-20260725-04',
'AC333-20260726-02',
'AC333-20260726-04',


    ]
for s in sessions:
    
    p_data = os.path.join(r'Z:\Jingyu\2P_Recording',s[0:-12], s[:-3], s[-2:])
    ops = np.load(r"Z:\Jingyu\Code\suite2p_ops\suite2p.npy", allow_pickle=True).item()
    p_out = os.path.join(p_data, 'suite2p')
    # ops['th_badframes'] = 0
    ops['do_bidiphase'] = 1
    ops['roidetect'] = 0
        
    print(f'INFO: Running suite2p for {p_data}')
    
    # save output to 'run.log' file
    os.makedirs(p_out, exist_ok = True)
    p_log = p_out+r'/run_suite2p.log'    
    print(f'INFO: Saving text output to {p_log}')
    
    # run suite2p
    db = {
        'data_path': [ str(p_data) ],
        # 'save_path0': p_out,
        # 'save_path0': str(p_data)+r'\ROI_detection_test_2.0',
    }
    
    with open(p_log, 'w') as f:
        with redirect_stdout(f):
            print(f'Running suite2p v{suite2p.version} from Spyder')
            suite2p.run_s2p(ops=ops, db=db)
    
    print('***runing ROI detection***')
    # force roi detection to cover the full FOV
    p_ops = Path(p_data)/'suite2p'/'plane0'/'ops.npy'
    if p_ops.exists():
        ops = np.load(p_ops, allow_pickle=True).item()
        ops['xrange'] = [0, 512]
        ops['yrange'] = [0, 512]
        ops['roidetect'] = 1
        np.save(p_ops, np.array(ops, dtype='object'))
    with open(p_log, 'w') as f:
        with redirect_stdout(f):
            print(f'Running suite2p v{suite2p.version} from Spyder')
            suite2p.run_s2p(ops=ops, db=db)    
            
            suite2p_ori = p_out
            suite2p_new = p_data+r'\suite2p_func_detec'
            if not os.path.exists(suite2p_new+r'\plane0\stat.npy'):
                copytree(
                suite2p_ori,
                suite2p_new,
                # Any entry matching these names/patterns will be skipped
                ignore=ignore_patterns(
                    # directories to skip by name
                    "reg_tif", "reg_tif_chan2",
                    # files to skip by name or glob
                    "*.bin",
                ),
                dirs_exist_ok=True
                )
                    
    print(f'{s}------Finished------')

