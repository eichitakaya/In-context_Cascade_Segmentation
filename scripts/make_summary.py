"""
必要な機能
- 結果のサマリを作成する
　- 全条件、全部位、全症例の結果をまとめる
    - 各条件について、部位ごとに全症例の統計情報をまとめ、エラーバーを作成してpngで保存する
- 各症例について、各条件および各部位におけるスライス毎dice係数を折れ線グラフで表し、pngで保存する Done
    - 症例×部位の数だけ、pngが作成される。1つのpngには、各条件におけるスライス毎dice係数が描画される Done

関数
- 任意の症例について、dice係数の統計情報を算出する
- 任意の部位について、全症例のdice係数の統計情報を算出する（平均の平均を求めることになる）
- 任意の症例について、各条件におけるスライス毎dice係数を折れ線グラフで表し、pngで保存する　Done
- 任意の症例について、各条件における推論結果とgroundtruthの画像を並べて表示し、pngで保存する

"""
import os
import json
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from dataset import label_dict

# 任意の症例について、各条件におけるスライス毎dice係数を折れ線グラフで表し、pngで保存する関数
def plot_dice_per_slice(base_path, patient_id, region):

    # 各条件におけるdice係数のデータを読み込む。diceがない場合もあるので例外処理
    try:
        #df_baseline00 = pd.read_table(f"{base_path}/baseline_A0T0/{region}/patient{patient_id}/dice", header=None)
        df_baseline10 = pd.read_table(f"{base_path}/baseline_A1T0/{region}/patient{patient_id}/dice", header=None)
        #df_baseline11 = pd.read_table(f"{base_path}/baseline_A1T1/{region}/patient{patient_id}/dice", header=None)
        #df_window00 = pd.read_table(f"{base_path}/window_A0T0/{region}/patient{patient_id}/dice", header=None)
        df_window10 = pd.read_table(f"{base_path}/window_A1T0/{region}/patient{patient_id}/dice", header=None)
        #df_window11 = pd.read_table(f"{base_path}/window_A1T1/{region}/patient{patient_id}/dice", header=None)
        #print(df_baseline10)
        #print(df_window10)
    except:
        return
    # スライス毎dice係数を折れ線グラフで表す
    #plt.plot(df_baseline00, label="baseline_A0T0")
    #plt.plot(df_window00, label="window_A0T0")
    plt.plot(df_baseline10, label="baseline")
    plt.plot(df_window10, label="window")
    #plt.plot(df_baseline11, label="baseline_A1T1")
    #plt.plot(df_window11, label="window_A1T1")


    # グラフのタイトル、軸ラベル、凡例を設定
    plt.title(f"Patient{patient_id} {region}")
    plt.xlabel("Slice")
    plt.ylabel("Dice coefficient")
    plt.legend()
    
    # グラフをpngで保存。フォルダが存在しない場合は作成する
    os.makedirs(f"{base_path}/dice_per_slice/patient{patient_id}", exist_ok=True)
    #print("hoge")
    plt.savefig(f"{base_path}/dice_per_slice/patient{patient_id}/{region}.png")
    plt.close()

def patient_summary(base_path, patient_id, region):
    # 各条件におけるdice係数のデータを読み込む。diceがない場合もあるので例外処理
    try:
        #df_baseline00 = pd.read_table(f"{base_path}/baseline_A0T0/{region}/patient{patient_id}/dice", header=None)[0]
        df_baseline10 = pd.read_table(f"{base_path}/baseline_A1T0/{region}/patient{patient_id}/dice", header=None)[0]
        #df_baseline11 = pd.read_table(f"{base_path}/baseline_A1T1/{region}/patient{patient_id}/dice", header=None)[0]
        #df_window00 = pd.read_table(f"{base_path}/window_A0T0/{region}/patient{patient_id}/dice", header=None)[0]
        df_window10 = pd.read_table(f"{base_path}/window_A1T0/{region}/patient{patient_id}/dice", header=None)[0]
        #df_window11 = pd.read_table(f"{base_path}/window_A1T1/{region}/patient{patient_id}/dice", header=None)[0]
    except:
        return
    # 各dfのnanを削除
    #df_baseline00 = df_baseline00.dropna()
    df_baseline10 = df_baseline10.dropna()
    #df_baseline11 = df_baseline11.dropna()
    #df_window00 = df_window00.dropna()
    df_window10 = df_window10.dropna()
    #df_window11 = df_window11.dropna()

    # 各条件におけるdice係数の統計情報を算出
    #baseline00_mean = df_baseline00.mean()
    #baseline00_std = df_baseline00.std()
    baseline10_mean = df_baseline10.mean()
    baseline10_std = df_baseline10.std()
    #baseline11_mean = df_baseline11.mean()
    #baseline11_std = df_baseline11.std()
    #window00_mean = df_window00.mean()
    #window00_std = df_window00.std()
    window10_mean = df_window10.mean()
    window10_std = df_window10.std()
    #window11_mean = df_window11.mean()
    #window11_std = df_window11.std()

    # 結果をまとめる
    result = {
        #"baseline_A0T0": [baseline00_mean, baseline00_std],
        "baseline_A1T0": [baseline10_mean, baseline10_std],
        #"baseline_A1T1": [baseline11_mean, baseline11_std],
        #"window_A0T0": [window00_mean, window00_std],
        "window_A1T0": [window10_mean, window10_std],
        #"window_A1T1": [window11_mean, window11_std]
    }

    return result

# 平均値の差の検定（t検定）を行う関数
# "list1の平均値の方が大きい"を帰無仮説とする場合、alternative="less"
# 両側検定の場合はtwo-sided
def t_test(list1, list2):
    pvalue = stats.ttest_rel(list1, list2, alternative="two-sided")
    return pvalue

def plot_and_summary(base_path):
    
    # 症例数×部位数の評価
    patients = range(60)
    # 1から8まで
    regions = range(1, 9)
    #for patient_id in patients:
    #    for region in regions:
    #        plot_dice_per_slice(base_path, patient_id, label_dict(region))
    
    # 部位ごとに、全条件全症例の結果をまとめる
    # 結果を格納する辞書を初期化
    results = {}
    for region in regions:
        results[label_dict(region)] = []
        
    for patient_id in patients:
        for region in regions:
            # 辞書式の結果をリストに追加
            result = patient_summary(base_path, patient_id, label_dict(region))
            if result:
                results[label_dict(region)].append(result)
    
    # 辞書の中身を集計
    # baselineA1T0, windowA1T0の平均と標準偏差を部位ごとに計算
    summary = {"baseline_A1T0": {}, "window_A1T0": {}, "p-value": {}}
    for region in regions:
        #print(region)
        summary["baseline_A1T0"][label_dict(region)] = []
        summary["window_A1T0"][label_dict(region)] = []
        summary["p-value"][label_dict(region)] = []
        baseline_results = []
        window_results = []
        for result in results[label_dict(region)]:
            if result:
                #print(result["baseline_A1T0"])
                baseline_results.append(result["baseline_A1T0"][0])
                window_results.append(result["window_A1T0"][0])
        baseline_mean = pd.DataFrame(baseline_results).mean()
        baseline_std = pd.DataFrame(baseline_results).std()
        window_mean = pd.DataFrame(window_results).mean()
        window_std = pd.DataFrame(window_results).std()
        # t検定を行い, p値を計算
        p = t_test(baseline_results, window_results)
        summary["baseline_A1T0"][label_dict(region)] = baseline_results
        summary["window_A1T0"][label_dict(region)] = window_results
        #print(baseline_results)
        #print(window_results)
        #print(p.pvalue)
        summary["p-value"][label_dict(region)] = p.pvalue
    
    # 結果の辞書をjson形式で保存
    with open(f"{base_path}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
            

if __name__ == "__main__":
    for i in range(1,6):
        base_path = f"../results/m{i}"
        plot_and_summary(base_path)
        for patient_id in range(60):
            for region in range(1, 9):
                plot_dice_per_slice(base_path, patient_id, label_dict(region))