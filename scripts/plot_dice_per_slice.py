import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# seabornのスタイル設定
sns.set_style("whitegrid")

# 1行2列のグラフを作成
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(wspace=0.3)

# 表示したい構造のインデックスとタイトル
plot_indices = [5, 0]  # LVは0番目、PAは5番目
titles = ['PA', 'LV']

for i, plot_idx in enumerate(plot_indices):
    score_b = torch.load(f"../results/pths/w5_pat32_reg{plot_idx+1}_metbaseline_sc.pth")
    score_w = torch.load(f"../results/pths/w5_pat32_reg{plot_idx+1}_metwindow_sc.pth")
    
    sns.lineplot(data=score_b, color='#4A90E2', label='Baseline', ax=axs[i])
    sns.lineplot(data=score_w, color='#E57373', label='ICS', ax=axs[i])
    
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Slice number')
    axs[i].set_ylabel('DSC')
    if i == 0:  # 最初のプロットにのみlegendを表示
        axs[i].legend()
    else:
        axs[i].get_legend().remove()  # 2つ目のプロットのlegendを削除
    axs[i].set_xlim(0, len(score_b))

plt.savefig("../results/dice_per_slice.png", dpi=300)