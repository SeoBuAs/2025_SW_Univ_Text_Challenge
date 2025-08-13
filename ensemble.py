import pandas as pd
from pathlib import Path

csv_files = list(Path('./results').glob('predictions_*.csv'))
dfs = [pd.read_csv(f) for f in csv_files]

result = dfs[0].copy()
result['generated'] = sum(df['generated'] for df in dfs) / len(dfs)

result.to_csv('./results/final.csv', index=False)