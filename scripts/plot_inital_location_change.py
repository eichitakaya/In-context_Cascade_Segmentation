import glob
import os
import math
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# seabornのスタイル設定
sns.set_style("whitegrid")

# サブプロットのグリッドを作成
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()  # 2次元配列を1次元に変換

titles = ['LV', 'RV', 'LA', 'RA', 'AO', 'PA', 'SVC', 'IVC']

for region in range(1, 9):
    files = glob.glob("../results/expe_20241128/w5_pat32_reg"+str(region)+"_metbaseline_s*.pth")
    files = sorted(files, key=lambda x: int(re.search(r's(\d+)\.pth$', x).group(1)))
    
    means_b, means_w = [], []
    
    for file_b in files:
        file_w = file_b.replace("metbaseline","metwindow")
        
        score_b = torch.load(file_b)
        score_w = torch.load(file_w)
    
        mean_b = np.nanmean(score_b)
        mean_w = np.nanmean(score_w)
    
        means_b.append(mean_b)
        means_w.append(mean_w)
    
    # インデックスは0から始まるので region-1
    ax = axes[region-1]
    sns.lineplot(data=means_b, color='#4A90E2', label="Baseline", ax=ax)  # 青
    sns.lineplot(data=means_w, color='#E57373', label="ICS", ax=ax)      # 赤
    ax.set_ylabel("DSC")
    ax.set_xlabel("Start position")
    ax.set_ylim(0, 1)
    ax.set_title(titles[region-1])
    ax.legend()

# サブプロット間の間隔を調整
plt.tight_layout()

# 保存
plt.savefig("../results/all_regions_inital_location_change.png", dpi=300)