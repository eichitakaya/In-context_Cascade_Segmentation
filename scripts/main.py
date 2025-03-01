import os
import sys
import glob
import math
import time
import torch
import itertools
import subprocess
import numpy as np
import einops as E
import nibabel as nib
from tqdm.auto import tqdm
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import f_oneway
from torchvision import transforms
import torchvision
from collections import defaultdict
from torchvision.transforms import functional as TF

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
    labels[labels == region] = 1
    
    memo = []
    for i in range(labels.shape[0]):
        # 非ゼロ要素が含まれている場合、スライスiをmemoに追加
        if np.count_nonzero(labels[i]) > 0:
            memo.append(i)
    
    # メモに含まれているスライスのみをsliceとlabelからそれぞれ抽出
    slices = slices[memo]
    labels = labels[memo]
    return slices, labels
    

def load_hvsmr_data(region, patient, nii_path, label_path):
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
    
def process_hvsmr_data(nii_path, region):
    patient = int(os.path.basename(nii_path).split('_')[0][3:])
    label_path = nii_path.replace('_orig.nii.gz', '_orig_seg.nii.gz')
    
    nii_data, label_data = load_hvsmr_data(region, patient, nii_path, label_path)
    nii_tensor = torch.from_numpy(nii_data).float()
    label_tensor = torch.from_numpy(label_data).long()

    return nii_tensor, label_tensor

# Dice metric for measuring volume agreement
def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2*(y_pred*y_true).sum() / (y_pred.sum() + y_true.sum())
    if (y_pred.sum() + y_true.sum()) == 0:
        score = 0
    else: score = score.item()
    return score

def rotation_yama(im, angle):
    im = TF.to_pil_image(im.squeeze())
    im = TF.rotate(im, angle)
    im = TF.to_tensor(im)
    return im

# run inference and compute losses for one test image
@torch.no_grad()
def inference_withoutTTA(model, image, label, support_images, support_labels):
    image, label = image.to(device), label.to(device)

    # inference
    logits = model(
        image[None],
        support_images[None],
        support_labels[None]
    )[0] # outputs are logits

    soft_pred = torch.sigmoid(logits)
    hard_pred = soft_pred.round().clip(0,1)

    #  score
    score = dice_score(hard_pred, label)

    # return a dictionary of all relevant variables
    return {'Image': image,
            'Soft Prediction': soft_pred,
            'Prediction': hard_pred,
            'Ground Truth': label,
            'score': score}

def process_aug(im, la, angles = [90,180,270]):
    im0 = im
    la0 = la
    for p in range(len(im0)):
        for angle in angles:
            im = torch.cat((im, rotation_yama(im0[p].cpu(), angle).to(device).unsqueeze(0)), dim=0)
            la = torch.cat((la, rotation_yama(la0[p].cpu(), angle).to(device).unsqueeze(0)), dim=0)
    return im, la

def calculator_v5_at_specific_m(images, labels, model, m, w, L, path, support_method, AUG):
    images = (images - images.min()) / (images.max() - images.min())

    scores_at_specific_m = np.full(L, np.nan).tolist()
    images_at_specific_m = torch.full((L, images.size()[2], images.size()[3]), float('nan'))
    label_at_specific_m = torch.full((L, images.size()[2], images.size()[3]), float('nan'))
    prediction_at_specific_m = torch.full((L, images.size()[2], images.size()[3]), float('nan'))
    time_at_specific_m = np.full(L, np.nan).tolist()

    if m+w <= L:
        # anterograde ##########################################
        if (m+w) != L:
            support_images = images[m:(m+w), :, :, :].float().to(device)
            support_labels = labels[m:(m+w), :, :, :].float().to(device)
            results_antero = defaultdict(list)

            for i in range(L-(m+w)):
                if AUG == 1:
                    support_images_2, support_labels_2 = process_aug(support_images, support_labels)
                image = images[(m+w+i), :, :, :].float()
                label = labels[(m+w+i), :, :, :].float()
                
                vals = inference_withoutTTA(model, image, label, support_images_2, support_labels_2)                

                for k, v in vals.items():
                    results_antero[k].append(v)
                if support_method == 'baseline':
                    support_images = support_images
                    support_labels = support_labels
                elif support_method == 'window':
                    support_images = torch.cat((support_images[1:, :, :, :], vals['Image'].unsqueeze(0)), dim=0)
                    support_labels = torch.cat((support_labels[1:, :, :, :], vals['Prediction'].unsqueeze(0)), dim=0)
                else:print('not valid support_method!')

                images_at_specific_m[(m+w+i), :, :] = vals['Image'].unsqueeze(0)
                label_at_specific_m[(m+w+i), :, :] = label
                prediction_at_specific_m[(m+w+i), :, :] = vals['Prediction'].unsqueeze(0)

                # image, label, predictionを画像として保存。ただしpathは、path_p_r_met_sc/imgs/raw1.png, path_p_r_met_sc/imgs/label1.png, path_p_r_met_sc/imgs/pred1.pngとする。
                os.makedirs(os.path.dirname(path.replace('.pth', '/imgs/')), exist_ok=True)
                img_path = path.replace('.pth', f'/imgs/raw{m+w+i}.png')
                label_path = path.replace('.pth', f'/imgs/label{m+w+i}.png')
                pred_path = path.replace('.pth', f'/imgs/pred{m+w+i}.png')
                torchvision.utils.save_image(vals['Image'], img_path)
                torchvision.utils.save_image(vals['Prediction'], pred_path)
                torchvision.utils.save_image(vals['Ground Truth'], label_path)
                
            scores_antero = results_antero.pop('score')
            scores_at_specific_m[(m+w):] = scores_antero


        # retrograde ##########################################
        if m != 0:
            support_images = images[m:(m+w), :, :, :].float().to(device)
            support_labels = labels[m:(m+w), :, :, :].float().to(device)
            results_retro = defaultdict(list)
                        
            for i in range(m):
                if AUG == 1:
                    support_images_2, support_labels_2 = process_aug(support_images, support_labels)
                image = images[(m-i-1), :, :, :].float()
                label = labels[(m-i-1), :, :, :].float()
                
                vals = inference_withoutTTA(model, image, label, support_images_2, support_labels_2)
                
                for k, v in vals.items():
                    results_retro[k].append(v)
                    
                if support_method == 'baseline':
                    support_images = support_images
                    support_labels = support_labels
                elif support_method == 'window':
                    support_images = torch.cat((vals['Image'].unsqueeze(0), support_images[:(w-1), :, :, :]), dim=0)
                    support_labels = torch.cat((vals['Prediction'].unsqueeze(0), support_labels[:(w-1), :, :, :]), dim=0)
                else:print('not valid support_method!')
                
                images_at_specific_m[(m-i-1), :, :] = vals['Image'].unsqueeze(0)
                label_at_specific_m[(m-i-1), :, :] = label
                prediction_at_specific_m[(m-i-1), :, :] = vals['Prediction'].unsqueeze(0)
                
                # image, label, predictionを画像として保存。ただしpathは、path_p_r_met_sc/imgs/raw.png, path_p_r_met_sc/imgs/label.png, path_p_r_met_sc/imgs/pred.pngとする。
                # imgsフォルダがなければ作る
                os.makedirs(os.path.dirname(path.replace('.pth', '/imgs/')), exist_ok=True)
                img_path = path.replace('.pth', f'/imgs/raw{m-i-1}.png')
                label_path = path.replace('.pth', f'/imgs/label{m-i-1}.png')
                pred_path = path.replace('.pth', f'/imgs/pred{m-i-1}.png')
          
                torchvision.utils.save_image(vals['Image'], img_path)
                torchvision.utils.save_image(vals['Prediction'], pred_path)
                torchvision.utils.save_image(vals['Ground Truth'], label_path)
            scores_retro = results_retro.pop('score')
            scores_at_specific_m[0:m] = scores_retro[::-1]

    return scores_at_specific_m

if __name__ == "__main__":
    ### UniverSeg setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sys.path.append('UniverSeg')
    from universeg import universeg
    model_universeg = universeg(pretrained=True)
    _ = model_universeg.to(device)

    ###parameter 
    expe = "../results/expe_20241204"
    support_methods = ['baseline','window']
    AUG = 1
    ###
    
    nii_files = glob.glob('../../Medical_AI/data/HVSMR/orig/pat*_orig.nii.gz')
    nii_files = sorted(nii_files)
    
    for nii_path in nii_files:
        print(nii_path)
        for region in range(1, 9):
            print(region)
            
            #dataを読み込む, 
            nii_tensor, label_tensor = process_hvsmr_data(nii_path, region)

            #Lを算出
            L = nii_tensor.size()[0]

            #有効なデータにのみ処理を行う.
            if nii_tensor.shape[0] != 0:
                # wを1から4まで変更しながらループ
                for w in range(1, 5):
                    # 有効・特定の症例・特定の領域に対しての処理
                    if w % 2 == 0:
                        start = int(L/2) - int(w//2)
                    elif w % 2 == 1:
                        start = int(L/2) - int((w//2) + 1)
                    for support_method in support_methods:
                        
                        path = os.path.join(expe, 'w' + str(w) + '_pat' + str(os.path.basename(nii_path).split('_')[0][3:]) + '_reg' +  str(region) + '_met' + support_method + '_sc' + '.pth')
                        
                        scores = calculator_v5_at_specific_m(nii_tensor, label_tensor, model_universeg, start, w, L, path, support_method=support_method, AUG = AUG)
                        torch.save(scores, path)
                
            elif nii_tensor.shape[0] == 0:
                print(f'Data for path_p_r NOT processed.')










