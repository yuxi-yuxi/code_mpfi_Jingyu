import pandas as pd
import numpy as np

df = pd.read_parquet(r'Z:\Jingyu\GCaMP_drug_infusion\place_cell_dataframe\AC989-20250711-04_place_cell_dataframe.parquet')

output = []
output.append(f'Shape: {df.shape}')
output.append('')
output.append('Columns and sample values:')
for col in df.columns:
    sample_val = df[col].iloc[0] if len(df) > 0 else None
    if isinstance(sample_val, np.ndarray):
        sample_str = f'array shape={sample_val.shape}'
    elif isinstance(sample_val, list):
        sample_str = f'list len={len(sample_val)}'
    else:
        sample_str = str(sample_val)[:50]
    output.append(f'  {col}: {df[col].dtype} | sample: {sample_str}')

with open(r'Z:\Jingyu\code_mpfi_Jingyu\place_cell_analysis\df_structure.txt', 'w') as f:
    f.write('\n'.join(output))
