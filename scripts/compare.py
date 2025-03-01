import json
import numpy as np
import pandas as pd

# JSONファイルを読み込む
with open('../results/pths/5_summary.json', 'r') as f:
    data = json.load(f)

# 結果を格納するリスト
results = []

# 各領域について処理
regions = ['LV', 'RV', 'LA', 'RA', 'AO', 'PA', 'SVC', 'IVC']
for region in regions:
    baseline_values = data['baseline_A1T0'][region]
    window_values = data['window_A1T0'][region]
    p_value = data['p-value'][region]
    
    result = {
        'Region': region,
        'Baseline Mean': np.mean(baseline_values),
        'Baseline Std': np.std(baseline_values),
        'Window Mean': np.mean(window_values),
        'Window Std': np.std(window_values),
        'p-value': p_value
    }
    results.append(result)

# DataFrameを作成して表示
df = pd.DataFrame(results)
df = df.round(4)  # 小数点4桁に丸める
print(df.to_string(index=False))