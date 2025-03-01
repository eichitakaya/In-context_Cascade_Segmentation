import numpy as np
import torch
import matplotlib.pyplot as plt

# 1行2列のグラフを作成
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(wspace=0.3)

# 表示したい構造のインデックスとタイトル
plot_indices = [0, 5]  # LVは0番目、PAは5番目
titles = ['LV', 'PA']

for i, plot_idx in enumerate(plot_indices):
    score_b = torch.load(f"../results/pths/w5_pat32_reg{plot_idx+1}_metbaseline_sc.pth")
    score_w = torch.load(f"../results/pths/w5_pat32_reg{plot_idx+1}_metwindow_sc.pth")
    
    # ラベル付きでプロット（PAのみラベルを付ける）
    if i == 1:  # PAの場合
        axs[i].plot(score_b, linestyle='-', color='k', label='baseline')
        axs[i].plot(score_w, linestyle='-', color='r', label='window')
        axs[i].legend(loc='upper right')  # PAの右上にlegendを配置
    else:  # LVの場合
        axs[i].plot(score_b, linestyle='-', color='k')
        axs[i].plot(score_w, linestyle='-', color='r')
    
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('slice number')
    axs[i].set_ylabel('dice')
    axs[i].grid(True)
    axs[i].set_xlim(0, len(score_b))

plt.savefig("../results/dice_per_slice_LVPA.png")