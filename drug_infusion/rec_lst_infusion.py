# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:42:29 2026

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd

exp = r'GCaMP8s_infusion'

rec_lst_infusion = [
{'anm': 'AC986', 'date': '20250618', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 0.5, 0.5], 'latency': 20,}, 
{'anm': 'AC988', 'date': '20250618', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 0.5, 0.5], 'latency': 20,}, 
{'anm': 'AC986', 'date': '20250619', 'session': ['02', '04', ], 'label': ['baseline', 'SCH', ], 'conc': [np.nan, 1], 'latency': 20,}, 
{'anm': 'AC988', 'date': '20250619', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 1, 1], 'latency': 20,}, 
{'anm': 'AC986', 'date': '20250620', 'session': ['02', '04', '06'], 'label': ['baseline', 'ctrl', 'ctrl2'], 'conc': [np.nan, np.nan, np.nan], 'latency': 20,},
{'anm': 'AC988', 'date': '20250620', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour issue, running speed, licks
{'anm': 'AC986', 'date': '20250621', 'session': ['02', '04', ], 'label': ['baseline', 'SCH', ], 'conc': [np.nan, 1], 'latency': 20,}, # behaviour issue, second sessions stop licking
# {'anm': 'AC988', 'date': '20250623', 'session': ['02', '04', ], 'label': ['baseline', 'SCH', ], 'conc': [np.nan, 1, 1]}, # behaviour issue, running speed, imaging quality not good
{'anm': 'AC986', 'date': '20250624', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 1, 1], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250625', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250626', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # some long stops
# {'anm': 'AC986', 'date': '20250626', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,]}, # missed some trials, bad imaging quality
{'anm': 'AC986', 'date': '20250627', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250627', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250628', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC986', 'date': '20250628', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 0.5,]}, #????????? don't know why commented before
{'anm': 'AC988', 'date': '20250629', 'session': ['02', '04', '06'], 'label': ['baseline', 'ctrl', 'ctrl2'], 'conc': [np.nan, np.nan, np.nan], 'latency': 20,},
{'anm': 'AC989', 'date': '20250630', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250701', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC988', 'date': '20250702', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250702', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250703', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC986', 'date': '20250703', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250704', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250706', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250707', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250708', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,]},
{'anm': 'AC989', 'date': '20250709', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250710', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250711', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250712', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,]},
{'anm': 'AC301', 'date': '20250713', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,]}, # acclimation
{'anm': 'AC989', 'date': '20250715', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250716', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250717', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC996', 'date': '20250719', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},# behaviour not perfect
{'anm': 'AC996', 'date': '20250721', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},# behaviour not perfect
{'anm': 'AC996', 'date': '20250725', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC302', 'date': '20250725', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC996', 'date': '20250726', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 0.5,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC302', 'date': '20250726', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  # behaviour not perfect
{'anm': 'AC302', 'date': '20250727', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC302', 'date': '20250728', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC996', 'date': '20250728', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC302', 'date': '20250729', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 2,], 'latency': 20,},  
# {'anm': 'AC302', 'date': '20250730', 'session': ['02', '04',], 'label': ['baseline', 'muscimol',], 'conc': [np.nan, 2,], 'latency': 20,},  
{'anm': 'AC302', 'date': '20250731', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 2,], 'latency': 30,},  
# {'anm': 'AC302', 'date': '20250801', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 5,], 'latency': 30,},  #?????
{'anm': 'AC310', 'date': '20250818', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC310', 'date': '20250819', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250820', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, 
{'anm': 'AC310', 'date': '20250821', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250822', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 2,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250823', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, 
{'anm': 'AC310', 'date': '20250826', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 2,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250827', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 3,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250828', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 5,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250829', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, 
# {'anm': 'AC310', 'date': '20250830', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 5,], 'latency': 20,},  # bad imaging quality
{'anm': 'AC316', 'date': '20251023', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},  
{'anm': 'AC316', 'date': '20251024', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC316', 'date': '20251028', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC316', 'date': '20251029', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
# {'anm': 'AC316', 'date': '20251030', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  # bad imaging quality
{'anm': 'AC314', 'date': '20251030', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251031', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC316', 'date': '20251031', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},    
{'anm': 'AC314', 'date': '20251103', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251104', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},  
{'anm': 'AC316', 'date': '20251104', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251105', 'session': ['02', '04',], 'label': ['baseline', 'prazosin',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC316', 'date': '20251106', 'session': ['02', '04',], 'label': ['baseline', 'prazosin',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251106', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251107', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC317', 'date': '20251110', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251111', 'session': ['02', '04',], 'label': ['baseline', 'prazosin',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC317', 'date': '20251111', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC314', 'date': '20251113', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC317', 'date': '20251113', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC314', 'date': '20251114', 'session': ['02', '04',], 'label': ['baseline', 'prazosin',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251117', 'session': ['02', '04',], 'label': ['baseline', 'prazosin',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC317', 'date': '20251117', 'session': ['02', '04',], 'label': ['baseline', 'prazosin',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251118', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},  
{'anm': 'AC317', 'date': '20251118', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251119', 'session': ['02', '04',], 'label': ['baseline', 'prazosin',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC317', 'date': '20251119', 'session': ['02', '04',], 'label': ['baseline', 'prazosin',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC314', 'date': '20251120', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC317', 'date': '20251120', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC314', 'date': '20251122', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC317', 'date': '20251121', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC317', 'date': '20251125', 'session': ['02', '04',], 'label': ['baseline', 'prazosin',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC317', 'date': '20251126', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},  
  
    ]

muscimol_imaging = [
{'anm': 'AC986', 'date': '20250614', 'session': ['02', '04', '06'], 'label': ['baseline', 'muscimol', 'muscimol2'], 'conc': [np.nan, 0.25, 0.25]},    
{'anm': 'AC985', 'date': '20250614', 'session': ['02', '04', '06'], 'label': ['baseline', 'muscimol', 'muscimol2'], 'conc': [np.nan, 0.25, 0.25]},        
    ]

#%% processed sessions metadata
SESSION_INFO_PATH = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\infusion_session_info.parquet"
df_session_info = pd.read_parquet(SESSION_INFO_PATH)

def cond(conc):
    if conc is None:
        return False
    
    conc = np.array(conc, dtype=float)
    
    # condition 1: all nan
    all_nan = np.isnan(conc).all()
    
    # condition 2: second element > 0.5 (if exists)
    second_valid = len(conc) > 1 and not np.isnan(conc[1]) and conc[1] > 0.5
    
    return all_nan or second_valid

def _label_contains(label, pattern):
    return any(pattern in str(lbl) for lbl in (label if isinstance(label, list) else [label]))

def _conc_gt_thresh(conc, thresh=0.5):
    if conc is None:
        return False
    conc = np.asarray(conc, dtype=float)
    return (len(conc) > 0) and np.any(~np.isnan(conc)) and np.nanmax(conc) > thresh

def _count_cumulative_days(df, label_pattern, col_name, require_conc_gt=None):
    """Count cumulative treatment days per animal, one count per date."""
    df = df.sort_values(['anm', 'date']).copy()
    df[col_name] = 0

    for anm in df['anm'].unique():
        anm_mask = df['anm'] == anm

        # keep one row per date, but include conc too
        anm_dates = df.loc[anm_mask, ['date', 'label', 'conc']].drop_duplicates('date')

        if require_conc_gt is None:
            is_match = anm_dates['label'].apply(lambda x: _label_contains(x, label_pattern))
        else:
            is_match = anm_dates.apply(
                lambda row: _label_contains(row['label'], label_pattern) and _conc_gt_thresh(row['conc'], require_conc_gt),
                axis=1
            )

        cumsum = is_match.cumsum()
        date_to_cumsum = dict(zip(anm_dates['date'], cumsum))
        df.loc[anm_mask, col_name] = df.loc[anm_mask, 'date'].map(date_to_cumsum)

    return df

# Drug days: only count dates with matching drug label AND conc > 0.5
df_session_info = _count_cumulative_days(df_session_info, 'SCH', 'SCH_days', require_conc_gt=0.5)
df_session_info = _count_cumulative_days(df_session_info, 'propranolol', 'propranolol_days', require_conc_gt=0.5)
df_session_info = _count_cumulative_days(df_session_info, 'prazosin', 'prazosin_days', require_conc_gt=0.5)

# Ctrl days: count by label only
df_session_info = _count_cumulative_days(df_session_info, 'ctrl', 'ctrl_days', require_conc_gt=None)

# Set drug day counts to NaN on ctrl days, and ctrl_days to NaN on drug days
is_ctrl = df_session_info['label'].apply(lambda x: _label_contains(x, 'ctrl'))
is_SCH = df_session_info['label'].apply(lambda x: _label_contains(x, 'SCH'))
is_propranolol = df_session_info['label'].apply(lambda x: _label_contains(x, 'propranolol'))
is_prazosin = df_session_info['label'].apply(lambda x: _label_contains(x, 'prazosin'))
is_drug = is_SCH | is_propranolol | is_prazosin

df_session_info.loc[is_ctrl, 'SCH_days'] = np.nan
df_session_info.loc[is_ctrl, 'propranolol_days'] = np.nan
df_session_info.loc[is_ctrl, 'prazosin_days'] = np.nan
df_session_info.loc[is_drug, 'ctrl_days'] = np.nan

# Reorder columns
cols = df_session_info.columns.tolist()
label_idx = cols.index('label') + 1
day_cols = ['SCH_days', 'propranolol_days', 'prazosin_days', 'ctrl_days']
cols = [c for c in cols if c not in day_cols]
cols = cols[:label_idx] + day_cols + cols[label_idx:]
df_session_info = df_session_info[cols]

# Filter for high-quality sessions
rec_lst = df_session_info[
                    (df_session_info['perc_valid_trials_ss1']>0.6)&
                    (df_session_info['perc_valid_trials_ss2']>0.6)&
                    (df_session_info['lick_idx_ss1']>0.8)&
                    (df_session_info['lick_idx_ss2']>0.8)&
                    (df_session_info['latency']==20)&
                    (df_session_info['anm']!='AC996') # different strain: Ai14:D1R
                    # &(df_session_info['perc_good_trials_ss1']>.3)
                    # &(df_session_info['max_speed_median_ss1']>50)
                    ]

rec_SCH = rec_lst[
    rec_lst[["label", "conc"]].apply(
        lambda row: ("SCH" in str(row["label"])) and (np.nanmin(row["conc"]) >= 1),
        axis=1)
        ]
rec_praz = rec_lst[
    rec_lst[["label", "conc"]].apply(
        lambda row: ("prazosin" in str(row["label"])) and (np.nanmin(row["conc"]) >= 1),
        axis=1)
        ]
rec_prop = rec_lst[
    rec_lst[["label", "conc"]].apply(
        lambda row: ("propranolol" in str(row["label"])) and (np.nanmin(row["conc"]) >= 1),
        axis=1)
        ]

rec_ctrl      = rec_lst[rec_lst['label'].apply(lambda x: 'ctrl' in x)]
rec_SCH_ctrl  = rec_ctrl[rec_ctrl['anm'].isin(rec_SCH['anm'])]
rec_praz_ctrl = rec_ctrl[rec_ctrl['anm'].isin(rec_praz['anm'])]
rec_prop_ctrl = rec_ctrl[rec_ctrl['anm'].isin(rec_prop['anm'])]
