# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:30:51 2024

@author: Jingyu Cao

only for newer suite2p version (>= V1.0.0.1)

"""
import sys
if "Z:\Jingyu\Code\Python" not in sys.path:
    sys.path.append("Z:\Jingyu\Code\Python")
# import anm_list_running as anm
# from utils_Jingyu import send_notification
import traceback
# import pandas as pd
import suite2p
from suite2p import default_settings, default_db
settings = default_settings()
db=default_db()
# db = {**default_db(), **db}
from pathlib import Path
import os
import numpy as np
import torch
from contextlib import redirect_stdout

from email.mime.text import MIMEText
import smtplib
to_email = "Jingyu.Cao@mpfi.org"
from_email = "midaure@gmail.com"
smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_username = "midlaure@gmail.com"
smtp_password = r"bksb vqmz dcqb mseb"
def send_notification(subject, message, to_email=to_email, from_email=from_email, 
                      smtp_server=smtp_server, smtp_port=smtp_port, smtp_username=smtp_username, smtp_password=smtp_password):
    """
    Sends an email using the provided SMTP server settings.
    """
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Secure the connection
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, [to_email], msg.as_string())

#%% params and sessions

# Basic
settings['torch_device']='cuda'
settings['run']['do_registration'] = 1
settings['run']['do_detection'] = 1
settings['fs']=30
# settings['tau'] = 1 # defualt = 1
# settings['diameter'] = [12.0, 12.0] # Expected cell diameter in pixels, passed to Cellpose, by default [12.0, 12.0]

# Registration
settings['registration']['align_by_chan2']=True # align by non-funcitonal channel
settings['registration']['do_difiphase']=True
settings['registration']['reg_tife']=True
settings['registration']['reg_tife_chan2']=True
settings['registration']['nimg_int']=300
settings['registration']['batch_size']=500
settings['registration']['smooth_sigma']=1.15
# settings['registration']['smooth_sigma_time']=0.35
settings['registration']['smooth_sigma_time']=0 # to skip the problem code
settings['registration']['block_size']=(64, 64)

# ROI detection
settings['detection']['algorithm'] = 'cellpose' # using cellpose to detect ROIs based on anatomical structures
settings['detection']['cellpose_settings']['img'] = 'meanImg' # Which image to segment: 'max_proj / meanImg', 'meanImg', or 'max_proj'
settings['detection']['cellpose_settings']['flow_threshold'] = 1.5 # default 0.4
settings['detection']['params'] = {'niter': 300, }
#%% 

rec_lst = [
# 'AC331-20260620-02',
# 'AC331-20260621-02',
# 'AC331-20260622-02',
# 'AC331-20260623-02',
# 'AC331-20260624-02',
# 'AC331-20260625-02',
# 'AC331-20260626-02',
# 'AC331-20260627-02',
# 'AC331-20260628-02',
'AC331-20260629-02',




# 'AC332-20260620-02',
# 'AC332-20260621-02',
# 'AC332-20260622-02',
# 'AC332-20260623-02',
# 'AC332-20260624-02',
# 'AC332-20260625-02',
# 'AC332-20260626-02',
# 'AC332-20260627-02',
# 'AC332-20260628-02',
'AC332-20260629-02',



           ]
base_dir = Path(r'Z:\Jingyu\2P_Recording')

run_detection_only = 0

if run_detection_only: # skip registraion and load bin file directly
    from suite2p import pipeline

#%%
for rec in rec_lst:
        anm, date, ss = rec.split('-')
        p_data = [
                    p for p in base_dir.iterdir()
                    if p.is_dir() and p.name.endswith(anm[-3:])
                ]
        if len(p_data)>1: # find more then one folder for this animal
            Warning('!!! find more then one folder for this anm, skip for nex anm')
            continue
        
        p_data = Path(p_data[0]/f'{anm}-{date}'/ss)
        
        if not run_detection_only:
            # db
            db['nchannels']=2
            db['functional_chan']=2
            db['save_path0']= str(p_data / r'anat_detect')
            db['data_path'] = [str(p_data),]
            suite2p.run_s2p(db=db, settings=settings)
        
        
        if run_detection_only: 
            p_suite2p = p_data/'nonrigid_reg_geco'/'suite2p'/'plane0'  
            f_bin_ch1 = p_suite2p/'data.bin'
            f_bin_ch2 = p_suite2p/'data_chan2.bin'
            data = np.memmap(f_bin_ch2, dtype='int16', mode='r', shape=(5000, 512, 512))
            
            p_out = p_data / r'anat_detect'
        
            if not p_out.exists():
                os.makedirs(p_out, exist_ok=0)
            
            save_path = p_out
            # settings['detection']['cellpose_settings']['img'] = 'meanImg' # Which image to segment: 'max_proj / meanImg', 'meanImg', or 'max_proj'
            # settings['detection']['cellpose_settings']['flow_threhold'] = 1.5 # default 0.4
            pipeline(save_path, f_reg=data, run_registration=False, settings=settings, device=torch.device("cuda"))
            
        
#%% run full suite2p
# rec_lst = [
#            # 'A223-20230519-02',
#            'A223-20230522-02', 
#            # 'A223-20230523-02',
#            # 'A223-20230523-04',
           
#            ]
# base_dir = Path(r'Z:\Jingyu\2P_Recording\GCamp')

# for rec in rec_lst:
#     # try:
        
#         # if exp=='axon-GCaMP_RdLight':
#         #     ops['align_by_chan']=2
        
        
#     anm, date, ss = rec.split('-')
#     p_data = [
#                 p for p in base_dir.iterdir()
#                 if p.is_dir() and p.name.endswith(anm[1:])
#             ]
#     if len(p_data)>1: # find more then one folder for this animal
#         Warning('!!! find more then one folder for this anm, skip for nex anm')
#         continue
    
#     p_data = Path(p_data[0]/f'{anm}-{date}'/ss)
#     p_suite2p = p_data/'suite2p'  
#     p_out = p_data / r'anat_detect'
    
#     if (p_suite2p/'plane0'/'stat.npy').exists():
#         os.makedirs(p_out, exist_ok=True)
#         # ops['path_roi_iterations'] = p_outops['save_path0']
#         # ops['save_path0'] = p_out
        
        
#         print('INFO: Running suite2p for {}'.format(p_data))
        
#         # save output to 'run.log' file
#         os.makedirs(p_out, exist_ok = True)
#         p_log = p_out/r'run_suite2p_anat_detec=3.log'    
#         print(f'INFO: Saving text output to {p_log}')
        
#         # run suite2p
#         db = {
#             'data_path': [ str(p_data) ],
#             'save_path0': str(p_out),
#             # 'save_path0': str(p_data)+r'\ROI_detection_test_2.0',
#         }
        
#         with open(p_log, 'w') as f:
#             with redirect_stdout(f):
#                 print(f'Running suite2p v{suite2p.version} from Spyder')
#                 suite2p.run_s2p(db=db, settings=settings)
            
#             print('Finished')
                
#             subject = "Registrations for\n{}\nFinished Successfully".format(s)
#             message = "Your Python script has finished running successfully."
            
#             reg_lst.append(s)
#             # try:
#             #     send_notification(subject, message)
#             #     print("Notification sent.")
#             # except Exception as notify_err:
#             #     print("Failed to send notification email:")
#             #     print(notify_err)
#         else:
#             print(f'--------{s}_RegOnly finished already-------------')
            
#     except Exception as e:
#         print(f'Error Raised-----------{s}-----------')
#         print(e)
#         subject = "suite2p registration Error Occurred for {}".format(s)
#         message = f"Your Python script encountered an error:\n\n{traceback.format_exc()}"
#         reg_f_lst.append(s)
#         try:
#             send_notification(subject, message)
#             print("Notification sent.")
#         except Exception as notify_err:
#             print("Failed to send notification email:")
#             print(notify_err)
        
# subject = "All Sessions - Registrations finished"
# message = "finished session list:\n{}\nerror sessions:\n{}".format(reg_lst, reg_f_lst)
# send_notification(subject, message)