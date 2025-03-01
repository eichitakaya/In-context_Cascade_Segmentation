# 8つの領域について、全スライスの画像のraw, pred, maskを8枚ずつ並べる。
# ただし、中心のサポートスライスは除く。かつ0番目とN番目を含む。かつ残りの6枚は均等にする。
# pngファイルは../results/pat32/w5_pat32_reg{i+1}_metbaseline_scと../results/pat32/w5_pat32_reg{i+1}_metwindow_scに保存されている。

import matplotlib.pyplot as plt
import glob
import matplotlib.gridspec as gridspec

index_list_dic = {"reg1": [0, 13, 27, 40, 46, 60, 74, 88], 
                  "reg2": [0, 16, 32, 48, 54, 70, 86, 103], 
                  "reg3": [0, 19, 39, 59, 65, 85, 105, 125], 
                  "reg4": [0, 8, 16, 25, 31, 40, 48, 57], 
                  "reg5": [0, 14, 28, 41, 47, 61, 75, 89], 
                  "reg6": [0, 8, 16, 25, 31, 40, 48, 57], 
                  "reg7": [0, 8, 16, 24, 30, 38, 47, 56], 
                  "reg8": [0, 11, 22, 34, 40, 52, 64, 76]}

# 8回繰り返す。
for i in range(8):
    # 各領域の画像を読み込む。
    baseline_folder = f"../results/pat32/w5_pat32_reg{i+1}_metbaseline_sc/imgs/"
    window_folder = f"../results/pat32/w5_pat32_reg{i+1}_metwindow_sc/imgs/"

    # GridSpecを使用して4行×9列のグラフを作成する。
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(4, 9, width_ratios=[1, 1, 1, 1, 0.2, 1, 1, 1, 1], height_ratios=[1, 1, 1, 1])

    # 行間を狭くするためにhspaceを小さく設定
    plt.subplots_adjust(hspace=0.01, wspace=0.1)

    # 各行にラベルを追加
    fig.text(0.12, 0.80, 'Raw', va='center', ha='right', fontsize=10, rotation='vertical')
    fig.text(0.12, 0.60, 'Baseline', va='center', ha='right', fontsize=10, rotation='vertical')
    fig.text(0.12, 0.40, 'ICS', va='center', ha='right', fontsize=10, rotation='vertical')
    fig.text(0.12, 0.20, 'Label', va='center', ha='right', fontsize=10, rotation='vertical')

    # raw, pred_baseline, pred_window, labelで行を分け、8枚ずつ並べる。
    reg_key = f"reg{i+1}"
    for j in range(8):
        slice_index = index_list_dic[reg_key][j]
        
        axs = fig.add_subplot(gs[0, j if j < 4 else j + 1])
        raw_img = plt.imread(baseline_folder + f"raw{slice_index}.png")
        axs.imshow(raw_img)
        axs.axis('off')
        axs.set_title(f"Slice {slice_index}", fontsize=8)

        axs = fig.add_subplot(gs[1, j if j < 4 else j + 1])
        pred_baseline_img = plt.imread(baseline_folder + f"pred{slice_index}.png")
        axs.imshow(pred_baseline_img)
        axs.axis('off')

        axs = fig.add_subplot(gs[2, j if j < 4 else j + 1])
        pred_window_img = plt.imread(window_folder + f"pred{slice_index}.png")
        axs.imshow(pred_window_img)
        axs.axis('off')

        axs = fig.add_subplot(gs[3, j if j < 4 else j + 1])
        mask_img = plt.imread(baseline_folder + f"label{slice_index}.png")
        axs.imshow(mask_img)
        axs.axis('off')

    # グラフを保存する。
    plt.savefig(f"../results/pat32_reg{i+1}_aligned_imgs.png", dpi=300)