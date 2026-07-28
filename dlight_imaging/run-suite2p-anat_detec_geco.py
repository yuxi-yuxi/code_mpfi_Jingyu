# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:30:51 2024

@author: Jingyu Cao



"""
import sys
if "Z:\Jingyu\Code\Python" not in sys.path:
    sys.path.append("Z:\Jingyu\Code\Python")
import anm_list_running as anm
# from utils_Jingyu import send_notification
import traceback
import pandas as pd
import suite2p
from pathlib import Path
import os
import numpy as np
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

#%%
exp = r'dlight_GECO_Ai14_Dbh'
# f_out_df_selected = r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_new.pkl".format(exp)
# df_selected = pd.read_pickle(f_out_df_selected)
# rec_lst = df_selected.index.tolist()

rec_lst = [
# 'AC327-20260602-02',     
# 'AC330-20260602-02',

# 'AC327-20260603-02',     
# 'AC330-20260603-02', 

'AC327-20260604-02',     
'AC330-20260604-02',        
    ]
#%% params and sessions
# ops = np.load(r"Z:\Jingyu\2P_Recording\AC918\AC918-20231028\04\ROI_detection_test_2.0\suite2p\plane0\ops.npy", allow_pickle=True).item()
ops = np.load(r"Z:\Jingyu\2P_Recording\suite2p_ops\RegOnly.npy", allow_pickle=True).item()

ops['do_registration']=1
ops['roidetect'] = 0
ops['reg_tif']=1
ops['reg_tif_chan2']=1
ops['align_by_chan']= 1 # align by dLight channel, Jingyu, 5/30/2025
ops['nonrigid']=1
ops['do_bidiphase']=1
# ops['smooth_sigma']=1.15
# ops['smooth_sigma_time']=0.35
# ops['batch_size'] = 500
# ops['block_size'] = (64, 64)
# ops['snr_thresh'] = 1.2

# exp='axon-GCaMP_RdLight'
if exp == r'dlight_GECO_Ai14_Dbh':
   # ops = np.load(r"Z:\Jingyu\2P_Recording\suite2p.npy", allow_pickle=True).item()
   ops['functional_chan']=2
   ops['roidetect'] = 1
   ops['anatomical_only']=3
   ops['sparse_mode'] = 1
   ops['spatial_scale']=0
   ops['denoise']=0
   ops['align_by_chan']=1

   ops['high_pass']=5
   ops['max_iteration']=20
   ops['max_overlap']=0.75
   ops['save_mat']=0
   #-----------for neuropil extraction--------------
   ops['circular_neuropil'] = True
   ops['inner_neuropil_radius']=2

#%% Main
reg_lst = []
reg_f_lst = []
for s in rec_lst:
    try:
        # if 'AC971' in s:
        #     ops['functional_chan']=2
        # else:
        #     ops['functional_chan']=1
        
        # if 'AC977' in s:
        #     ops['align_by_chan']=1
        
        if exp=='axon-GCaMP_RdLight':
            ops['align_by_chan']=2
        
        anm = s[:5]
        date = s[6:14]
        ss = s[15:]
        p_data = os.path.join(r'Z:\Jingyu\2P_Recording',anm, anm+'-'+date, ss)

    # ops['do_extraction'] = False
                
        p_out = p_data + r'\nonrigid_reg_geco'
        
        if os.path.exists(p_out+r'\suite2p\plane0\stat.npy') == False:
            os.makedirs(p_out, exist_ok=True)
            # ops['path_roi_iterations'] = p_outops['save_path0']
            ops['save_path0'] = p_out
            
            
            print('INFO: Running suite2p for {}'.format(p_data))
            
            # save output to 'run.log' file
            os.makedirs(p_out, exist_ok = True)
            p_log = p_out+r'/run_suite2p_anat_detec=3.log'    
            print(f'INFO: Saving text output to {p_log}')
            
            # run suite2p
            db = {
                'data_path': [ str(p_data) ],
                'save_path0': p_out,
                # 'save_path0': str(p_data)+r'\ROI_detection_test_2.0',
            }
            
            with open(p_log, 'w') as f:
                with redirect_stdout(f):
                    print(f'Running suite2p v{suite2p.version} from Spyder')
                    suite2p.run_s2p(ops=ops, db=db)
                
                print('Finished')
                
            subject = "Registrations for\n{}\nFinished Successfully".format(s)
            message = "Your Python script has finished running successfully."
            
            reg_lst.append(s)
            # try:
            #     send_notification(subject, message)
            #     print("Notification sent.")
            # except Exception as notify_err:
            #     print("Failed to send notification email:")
            #     print(notify_err)
        else:
            print(f'--------{s}_RegOnly finished already-------------')
            
    except Exception as e:
        print(f'Error Raised-----------{s}-----------')
        print(e)
        subject = "suite2p registration Error Occurred for {}".format(s)
        message = f"Your Python script encountered an error:\n\n{traceback.format_exc()}"
        reg_f_lst.append(s)
        try:
            send_notification(subject, message)
            print("Notification sent.")
        except Exception as notify_err:
            print("Failed to send notification email:")
            print(notify_err)
        
subject = "All Sessions - Registrations finished"
message = "finished session list:\n{}\nerror sessions:\n{}".format(reg_lst, reg_f_lst)
send_notification(subject, message)