import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import glob
import json

def lps2voxel(lps_coords, pixel_spacing, origin):
    """
    LPS座標をボクセル座標に変換する関数
    :param lps_coords: LPS座標（numpy配列）
    :param pixel_spacing: ピクセル間隔（numpy配列）
    :param origin: 画像データの原点（numpy配列）
    :return: ボクセル座標（numpy配列）
    """
    # LPSからRASへの変換（X軸とY軸の符号を反転）
    ras_coords = np.array([-lps_coords[0], -lps_coords[1], lps_coords[2]])
    
    # 物理座標からボクセル座標への変換
    voxel_coords = (ras_coords - origin) / pixel_spacing
    return voxel_coords


dicom_path = "/takaya_workspace/Medical_AI/data/vestibular_schwannoma/Vestibular-Schwannoma-SEG"
label_path = "/takaya_workspace/Medical_AI/data/vestibular_schwannoma/contours"
csv_path = "/takaya_workspace/Medical_AI/data/vestibular_schwannoma/DirectoryNamesMappingModality.csv"

# csvファイルの読み込み
df = pd.read_csv(csv_path)
# ModalityからT1 RTSTRUCTの行を抽出
df = df[df["Modality"] == "T1 image"]
# Descriptive Directory Nameの列の中身を取り出し、リストに格納
dir_names = df["Descriptive Directory Name"].values

# dir_names内のフォルダ名を順に取り出し、それぞれのフォルダ内のDICOMファイルを取得してniftiファイルに変換。対応するラベルも同様に取得してniftiファイルに変換。
for dir_name in dir_names:
    dicom_dir = f"{dicom_path}/{dir_name}"
    label_dir = f"{label_path}/{dir_name}"
    # 半角スペースが含まれている場合は、半角スペースをアンダーバーに置換
    if " " in dicom_dir:
        dicom_dir = dicom_dir.replace(" ", "_")
    # "-"で区切ったときの7番目の要素の次にNAを挿入して、それをさらに"-"で結合
    dicom_dir = dicom_dir.split("-")
    dicom_dir.insert(7, "NA")
    dicom_dir = "-".join(dicom_dir)

    dicom_files = sorted(glob.glob(dicom_dir + "/*.dcm"))
    #print(dicom_dir)
    #print(dicom_files)

    # DICOMファイルをniftiファイルに変換
    # 最初のファイルを読み込んで、そのaffineを取得
    dicom = pydicom.dcmread(dicom_files[0])
    affine = np.eye(4)
    affine[0, 0] = dicom.PixelSpacing[0]
    affine[1, 1] = dicom.PixelSpacing[1]
    affine[2, 2] = dicom.SliceThickness

    dicom_data = []
    #for dicom_file in dicom_files:
    #    dicom = pydicom.dcmread(dicom_file)
    #    print(dicom.ImagePositionPatient)
    dicom_data = np.array(dicom_data)

    dicom_nii = nib.Nifti1Image(dicom_data, affine)
    
    # 保存名はVS-SEG-xxxとし、ゼロ埋めで3桁にする。
    save_name = dir_name.split("/")[0].split("-")[2].zfill(3)
    #nib.save(dicom_nii, f"/takaya_workspace/Medical_AI/data/vestibular_schwannoma/niis/{save_name}.nii.gz")
    
    # ラベルファイルをniftiファイルに変換
    # 患者no.を特定
    # ゼロ埋めを解除
    patient_num = str(int(save_name))
    print(patient_num)
    # label_dirにvs_gk_xxx_t1というフォルダがあるので、patient_numに従ってその中のcontours.jsonを取得
    label_file = f"{label_path}/vs_gk_{patient_num}_t1/contours.json"
    
    with open(label_file) as f:
        json_data = json.load(f)

    # ボリュームのサイズを取得
    slices, height, width = 120, 512, 512

    

    # contour_pointsを取り出す
    contour_points = json_data[0]["LPS_contour_points"]
    # contour_pointsは3次元配列で、3次元目に3つの要素が格納されている。これらを取り出してn×3の配列に変換
    # リスト内包表記は使わない
    contour_points_stack = []
    
    #origin = np.array([0, 0, 0])
    for i in range(len(contour_points)):
        for j in range(len(contour_points[i])):
            slice_i = int(contour_points[i][j][2])
            dicom = pydicom.dcmread(dicom_files[slice_i])
            spacing = [dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness]
            origin = dicom.ImagePositionPatient
            transformed_points = lps2voxel(contour_points[i][j], spacing, origin)
            contour_points_stack.append(transformed_points)
    
    # contour_pointsをnumpy配列に変換
    contour_points_stack = np.array(contour_points_stack)
    #print(contour_points_stack)
    
    # 画像の初期化
    label = np.zeros((slices, height, width))
    
    # ボリュームにcontour_pointsを描画（x, y, z）
    for point in contour_points_stack:
        x, y, z = point
        #print(point)
        label[int(z), int(y), int(x)] = 1
    
    # niftiファイルに変換
    #affine = np.eye(4)
    label_nii = nib.Nifti1Image(label, affine)
    
    # save_nameをゼロ埋めにし、xxx_label.nii.gzとして保存
    nib.save(label_nii, f"/takaya_workspace/Medical_AI/data/vestibular_schwannoma/niis/{save_name}_label.nii.gz")