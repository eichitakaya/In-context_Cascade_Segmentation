"""
HVSMRデータセットを呼び出すための関数
"""


import nibabel as nib
import numpy as np
import glob

def label_dict(label):
    # ラベル情報の辞書
    dict = {
        1: "LV",
        2: "RV",
        3: "LA",
        4: "RA",
        5: "AO",
        6: "PA",
        7: "SVC",
        8: "IVC"
    }
    return dict[label]

def pad_images_to_square(images):
    """
    画像のバッチを受け取り、大きい方の寸法に合わせて左上揃えのパディングを適用する。
    
    :param images: numpy配列としての画像のバッチ（形状：(batchsize, 1, width, height)）
    :return: パディング後の画像のバッチ
    """
    batchsize, channels, max_width, max_height = images.shape
    target_size = max(max_width, max_height)  # 最大の幅または高さをターゲットサイズとする

    # パディング後の画像を格納するための配列を初期化
    padded_images = np.zeros((batchsize, channels, target_size, target_size))

    for i, img in enumerate(images):
        _, width, height = img.shape
        padded_images[i, :, :width, :height] = img  # 左上に画像を配置し、必要に応じて右と下をパディング

    return padded_images


# 関心ラベルが含まれているスライスのみを抽出する関数
def extract_region_slices(region, slices, labels):
    """
    slices: np.array
    labels: np.array
    """
    # regionに対応するラベル以外を0に変換
    labels[labels != region] = 0
    
    memo = []
    for i in range(labels.shape[0]):
        # 非ゼロ要素が含まれている場合、スライスiをmemoに追加
        if np.count_nonzero(labels[i]) > 0:
            memo.append(i)
    
    # メモに含まれているスライスのみをsliceとlabelからそれぞれ抽出
    slices = slices[memo]
    labels = labels[memo]
    return slices, labels
    

def load_hvsmr_data(region, patient):
    """
    HVSMRデータセットを読み込む関数
    region: 部位名(1~8)
    patient: 患者番号(0~59)
    """
    # niiファイルの読み込み
    nii_path = f"/takaya_workspace/Medical_AI/data/HVSMR/orig/pat{patient}_orig.nii.gz"
    label_path = f"/takaya_workspace/Medical_AI/data/HVSMR/orig/pat{patient}_orig_seg.nii.gz"
    
    # niiファイルを読み込む
    nii_data = nib.load(nii_path)
    nii_data = nii_data.get_fdata()
    label_data = nib.load(label_path)
    label_data = label_data.get_fdata()
    
    # (x, y, z) -> (z, 1, x, y)に変換
    nii_data = np.transpose(nii_data, (2, 1, 0))
    nii_data = nii_data[:, np.newaxis, :, :]
    label_data = np.transpose(label_data, (2, 1, 0))
    label_data = label_data[:, np.newaxis, :, :]
    
    # ラベルが一切含まれていないスライスを削除
    nii_data, label_data = extract_region_slices(region, nii_data, label_data)
    
    # 左上揃えのパディングを適用
    nii_data = pad_images_to_square(nii_data)
    label_data = pad_images_to_square(label_data)

    return nii_data, label_data

if __name__ == "__main__":
    # テスト
    raw, label = load_hvsmr_data(3, 50)
    print(raw.shape, label.shape)