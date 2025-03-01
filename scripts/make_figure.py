# 各実験結果フォルダのjsonファイルから、boxplotを作成する
# 基本的な精度比較は、m5のbaselineとwindowのみで行う
# jsonファイルの中身は、group-value方式に書き換える
"""
data = {
    'Group': ['A']*100 + ['B']*100 + ['C']*100,
    'Value': np.concatenate([np.random.normal(0, 1, 100), 
                             np.random.normal(1, 2, 100), 
                             np.random.normal(2, 1.5, 100)])
}

AO, IVC, LA, LV, PA, RA, RV, SVC
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import seaborn as sns

# jsonファイルの読み込み
result_path = "/takaya_workspace/Sequential_UniverSeg/results/pths/"
m5_path = result_path + "5_summary.json"

with open(m5_path, "r") as f:
    m5 = json.load(f)
    # baselineの精度を取得（各臓器）
    baseline = m5["baseline_A1T0"]
    # windowの精度を取得（各臓器）
    window = m5["window_A1T0"]
    #print(baseline)

# group-value方式に変換
data = {
    "group": ["LV"] * len(baseline["LV"]) + ["LV"] * len(window["LV"]) + 
             ["RV"] * len(baseline["RV"]) + ["RV"] * len(window["RV"]) + 
             ["LA"] * len(baseline["LA"]) + ["LA"] * len(window["LA"]) + 
             ["RA"] * len(baseline["RA"]) + ["RA"] * len(window["RA"]) + 
             ["AO"] * len(baseline["AO"]) + ["AO"] * len(window["AO"]) + 
             ["PA"] * len(baseline["PA"]) + ["PA"] * len(window["PA"]) + 
             ["SVC"] * len(baseline["SVC"]) + ["SVC"] * len(window["SVC"]) +  # SVCを先に
             ["IVC"] * len(baseline["IVC"]) + ["IVC"] * len(window["IVC"]),   # IVCを後に
    "value": baseline["LV"] + window["LV"] + 
             baseline["RV"] + window["RV"] + 
             baseline["LA"] + window["LA"] + 
             baseline["RA"] + window["RA"] + 
             baseline["AO"] + window["AO"] + 
             baseline["PA"] + window["PA"] + 
             baseline["SVC"] + window["SVC"] +  # SVCを先に
             baseline["IVC"] + window["IVC"],   # IVCを後に
    "method": ["baseline"] * len(baseline["LV"]) + ["ICS"] * len(window["LV"]) +
              ["baseline"] * len(baseline["RV"]) + ["ICS"] * len(window["RV"]) +
              ["baseline"] * len(baseline["LA"]) + ["ICS"] * len(window["LA"]) +
              ["baseline"] * len(baseline["RA"]) + ["ICS"] * len(window["RA"]) +
              ["baseline"] * len(baseline["AO"]) + ["ICS"] * len(window["AO"]) +
              ["baseline"] * len(baseline["PA"]) + ["ICS"] * len(window["PA"]) +
              ["baseline"] * len(baseline["SVC"]) + ["ICS"] * len(window["SVC"]) +  # SVCを先に
              ["baseline"] * len(baseline["IVC"]) + ["ICS"] * len(window["IVC"])    # IVCを後に
}

# DataFrameに変換
df = pd.DataFrame(data)
sns.set_palette("Set2")
#print(df)
# boxplotを作成
sns.boxplot(x='group', y='value', data=df, hue='method', palette=["#4A90E2", "#E57373"], showfliers=False)
plt.xlabel('Region')
plt.ylabel('DSC')
plt.legend()
plt.savefig(result_path + 'm5_boxplot.png', dpi=300)
plt.close()


# m1~m5の比較
m1_path = result_path + "1_summary.json"
m2_path = result_path + "2_summary.json"
m3_path = result_path + "3_summary.json"
m4_path = result_path + "4_summary.json"

with open(m1_path, "r") as f:
    m1 = json.load(f)
    m1_baseline = m1["baseline_A1T0"]
    m1_window = m1["window_A1T0"]

with open(m2_path, "r") as f:
    m2 = json.load(f)
    m2_baseline = m2["baseline_A1T0"]
    m2_window = m2["window_A1T0"]

with open(m3_path, "r") as f:
    m3 = json.load(f)
    m3_baseline = m3["baseline_A1T0"]
    m3_window = m3["window_A1T0"]

with open(m4_path, "r") as f:
    m4 = json.load(f)
    m4_baseline = m4["baseline_A1T0"]
    m4_window = m4["window_A1T0"]

data = {
   "group": ["LV"] * len(m1_window["LV"]) + ["LV"] * len(m2_window["LV"]) + ["LV"] * len(m3_window["LV"]) + ["LV"] * len(m4_window["LV"]) + ["LV"] * len(window["LV"]) +
            ["RV"] * len(m1_window["RV"]) + ["RV"] * len(m2_window["RV"]) + ["RV"] * len(m3_window["RV"]) + ["RV"] * len(m4_window["RV"]) + ["RV"] * len(window["RV"]) +
            ["LA"] * len(m1_window["LA"]) + ["LA"] * len(m2_window["LA"]) + ["LA"] * len(m3_window["LA"]) + ["LA"] * len(m4_window["LA"]) + ["LA"] * len(window["LA"]) +
            ["RA"] * len(m1_window["RA"]) + ["RA"] * len(m2_window["RA"]) + ["RA"] * len(m3_window["RA"]) + ["RA"] * len(m4_window["RA"]) + ["RA"] * len(window["RA"]) +
            ["AO"] * len(m1_window["AO"]) + ["AO"] * len(m2_window["AO"]) + ["AO"] * len(m3_window["AO"]) + ["AO"] * len(m4_window["AO"]) + ["AO"] * len(window["AO"]) +
            ["PA"] * len(m1_window["PA"]) + ["PA"] * len(m2_window["PA"]) + ["PA"] * len(m3_window["PA"]) + ["PA"] * len(m4_window["PA"]) + ["PA"] * len(window["PA"]) +
            ["SVC"] * len(m1_window["SVC"]) + ["SVC"] * len(m2_window["SVC"]) + ["SVC"] * len(m3_window["SVC"]) + ["SVC"] * len(m4_window["SVC"]) + ["SVC"] * len(window["SVC"]) +
            ["IVC"] * len(m1_window["IVC"]) + ["IVC"] * len(m2_window["IVC"]) + ["IVC"] * len(m3_window["IVC"]) + ["IVC"] * len(m4_window["IVC"]) + ["IVC"] * len(window["IVC"]),
    "value": m1_window["LV"] + m2_window["LV"] + m3_window["LV"] + m4_window["LV"] + window["LV"] +
            m1_window["RV"] + m2_window["RV"] + m3_window["RV"] + m4_window["RV"] + window["RV"] +
            m1_window["LA"] + m2_window["LA"] + m3_window["LA"] + m4_window["LA"] + window["LA"] +
            m1_window["RA"] + m2_window["RA"] + m3_window["RA"] + m4_window["RA"] + window["RA"] +
            m1_window["AO"] + m2_window["AO"] + m3_window["AO"] + m4_window["AO"] + window["AO"] +
            m1_window["PA"] + m2_window["PA"] + m3_window["PA"] + m4_window["PA"] + window["PA"] +
            m1_window["SVC"] + m2_window["SVC"] + m3_window["SVC"] + m4_window["SVC"] + window["SVC"] +
            m1_window["IVC"] + m2_window["IVC"] + m3_window["IVC"] + m4_window["IVC"] + window["IVC"],
    "window_size": ["m1"] * len(m1_window["LV"]) + ["m2"] * len(m2_window["LV"]) + ["m3"] * len(m3_window["LV"]) + ["m4"] * len(m4_window["LV"]) + ["m5"] * len(window["LV"]) +
                  ["m1"] * len(m1_window["RV"]) + ["m2"] * len(m2_window["RV"]) + ["m3"] * len(m3_window["RV"]) + ["m4"] * len(m4_window["RV"]) + ["m5"] * len(window["RV"]) +
                  ["m1"] * len(m1_window["LA"]) + ["m2"] * len(m2_window["LA"]) + ["m3"] * len(m3_window["LA"]) + ["m4"] * len(m4_window["LA"]) + ["m5"] * len(window["LA"]) +
                  ["m1"] * len(m1_window["RA"]) + ["m2"] * len(m2_window["RA"]) + ["m3"] * len(m3_window["RA"]) + ["m4"] * len(m4_window["RA"]) + ["m5"] * len(window["RA"]) +
                  ["m1"] * len(m1_window["AO"]) + ["m2"] * len(m2_window["AO"]) + ["m3"] * len(m3_window["AO"]) + ["m4"] * len(m4_window["AO"]) + ["m5"] * len(window["AO"]) +
                  ["m1"] * len(m1_window["PA"]) + ["m2"] * len(m2_window["PA"]) + ["m3"] * len(m3_window["PA"]) + ["m4"] * len(m4_window["PA"]) + ["m5"] * len(window["PA"]) +
                  ["m1"] * len(m1_window["SVC"]) + ["m2"] * len(m2_window["SVC"]) + ["m3"] * len(m3_window["SVC"]) + ["m4"] * len(m4_window["SVC"]) + ["m5"] * len(window["SVC"]) +
                  ["m1"] * len(m1_window["IVC"]) + ["m2"] * len(m2_window["IVC"]) + ["m3"] * len(m3_window["IVC"]) + ["m4"] * len(m4_window["IVC"]) + ["m5"] * len(window["IVC"])
}

df = pd.DataFrame(data)
sns.set_palette("husl")
sns.boxplot(x='group', y='value', data=df, hue='window_size', palette=["#FFBB98", "#FFA07A", "#FF8C61", "#FF7043", "#FF5733"], showfliers=False)
plt.xlabel('Region')
plt.ylabel('DSC')
plt.legend(prop={'size': 8})
plt.savefig(result_path + 'm1_m5_boxplot.png', dpi=300)
plt.close()
