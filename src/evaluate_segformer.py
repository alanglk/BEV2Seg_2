from oldatasets.BEV import BEVDataset
from oldatasets.NuImages import NuImagesDataset
from oldatasets.NuImages import NuImagesFormattedDataset
from oldatasets.NuImages.nulabels import nuid2name as id2label

from bev2seg_2 import Raw2Seg_BEV, Raw_BEV2Seg

import cv2

from vcd import core
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pickle

def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path doesnt exist: {path}")

def plot_mean_iou_per_class(mean_iou_per_class, title = 'mean_iou_per_class', labels_list=None, colors=None):
    if labels_list is None:
        labels_list = range(len(mean_iou_per_class))
    if colors is None:
        colors = 'skyblue'

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(labels_list, mean_iou_per_class, color=colors)
    plt.xlabel('Classes')
    plt.xticks(rotation=90)
    plt.ylabel('IoU')
    plt.show()

def calculate_iou(predictions, annotations, num_classes):
        """
        Calcula el IoU por clase y el mean IoU en la GPU usando PyTorch.
        
        :param predictions: tensor de predicciones del modelo, de tamaño (H, W).
        :param annotations: tensor de anotaciones, de tamaño (H, W).
        :param num_classes: número total de clases (en este caso 27).
        :return: tensor con IoU por clase y el mean IoU.
        """
        # Aseguramos que los valores sean enteros
        predictions = predictions.to(torch.int64)
        annotations = annotations.to(torch.int64)

        iou_per_class = torch.zeros(num_classes, device=predictions.device)
        tp = torch.zeros(num_classes, device=predictions.device)
        fp = torch.zeros(num_classes, device=predictions.device)
        fn = torch.zeros(num_classes, device=predictions.device)

        for c in range(num_classes):
            pred_mask = (predictions == c)
            ann_mask = (annotations == c)
            
            # Calcular TP, FP, FN
            tp[c] = torch.sum(pred_mask & ann_mask)
            fp[c] = torch.sum(pred_mask & ~ann_mask)
            fn[c] = torch.sum(~pred_mask & ann_mask)

        for c in range(num_classes):
            denominator = tp[c] + fp[c] + fn[c]
            if denominator != 0:
                iou_per_class[c] = tp[c] / denominator
            else:
                # Si no hay píxeles de esa clase en la imagen, el IoU es 0
                iou_per_class[c] = 0  

        # Calcular el mean IoU (promedio de los IoUs de todas las clases)
        mean_iou = torch.mean(iou_per_class)

        return iou_per_class, mean_iou


##################################################################################
#                                      MAIN                                      #
##################################################################################
def main(bevdataset_path:str, 
         nudataset_path:str, 
         raw2segmodel_path:str, 
         bev2segmodel_path:str,
         dataset_versions:str = 'mini',
         save_results:bool = False,
         results_path:str = None,
         plot_results:bool = False):
    check_paths([bevdataset_path, nudataset_path, raw2segmodel_path, bev2segmodel_path])

    # Load Datasets
    bev_dataset     = BEVDataset(dataroot=bevdataset_path, version=dataset_versions)
    nu_dataset      = NuImagesFormattedDataset(dataroot=nudataset_path, version=dataset_versions)
    assert len(bev_dataset) == len(nu_dataset)

    # Load Models for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw2seg_bev = Raw2Seg_BEV(raw2segmodel_path, None, device=device)
    raw_seg2bev = Raw_BEV2Seg(bev2segmodel_path, None, device=device)

    labels_list = list(raw2seg_bev.id2label.values())
    num_labels = len(labels_list)
    colors      = list(raw2seg_bev.id2color.values())
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    # Metrics
    mean_iou = { 'rawseg': 0.0, 'raw2seg_bev': 0.0, 'raw2bev_seg': 0.0 }
    mean_iou_per_class = {
        'rawseg': np.zeros(num_labels),
        'raw2seg_bev': np.zeros(num_labels),
        'raw2bev_seg': np.zeros(num_labels)
        }

    # Run Inference instance by instance and compute metrics
    for index in tqdm(range(len(bev_dataset))):
        # Get paths
        _, real_bev_mask_path   = bev_dataset._get_item_paths(index)
        sample_token            = os.path.splitext(os.path.basename(real_bev_mask_path))[0].replace('_semantic', '')
        raw_image_path          = os.path.join(nu_dataset.dataroot, sample_token + "_raw" + nu_dataset.image_extension)
        real_raw_mask_path      = os.path.join(nu_dataset.dataroot, sample_token + "_semantic" + nu_dataset.image_extension)
        openlabel_path          = os.path.join(bev_dataset.dataroot, bev_dataset.data_tokens[index] + ".json")
        check_paths([raw_image_path, real_raw_mask_path, real_bev_mask_path, openlabel_path])

        # Load files
        vcd = core.VCD()
        vcd.load_from_file(openlabel_path)
        raw_image       = cv2.imread(raw_image_path)
        real_raw_mask   = torch.tensor(cv2.imread(real_raw_mask_path)[:, :, 0], device=device)
        real_bev_mask   = torch.tensor(cv2.imread(real_bev_mask_path)[:, :, 0], device=device)
        
        # cv2.namedWindow("raw_image", cv2.WINDOW_NORMAL)
        # cv2.imshow("raw_image", raw_image)

        # Raw -> Seg -> BEV | Comparar inf_raw_mask con real_raw_mask e inf_bev_mask con real_bev_mask
        raw2seg_bev.set_openlabel(vcd)
        inf_raw_mask, inf_bev_mask = raw2seg_bev.generate_bev_segmentation(raw_image, 'CAM_FRONT')
        inf_raw_mask = torch.tensor(inf_raw_mask, device=device)
        inf_bev_mask = torch.tensor(inf_bev_mask, device=device)

        iou_per_class, m_iou = calculate_iou(inf_raw_mask, real_raw_mask, num_labels)
        mean_iou_per_class['rawseg'] += iou_per_class.cpu().detach().numpy()
        mean_iou['rawseg'] += m_iou.cpu().detach().numpy()
         
        iou_per_class, m_iou = calculate_iou(inf_bev_mask, real_bev_mask, num_labels)
        mean_iou_per_class['raw2seg_bev'] += iou_per_class.cpu().detach().numpy()
        mean_iou['raw2seg_bev'] += m_iou.cpu().detach().numpy()


        # Raw -> BEV -> Seg | Comparar inf_bev_mask con real_bev_mask  
        raw_seg2bev.set_openlabel(vcd)
        inf_bev_mask = raw_seg2bev.generate_bev_segmentation(raw_image, 'CAM_FRONT')
        inf_bev_mask = torch.tensor(inf_bev_mask, device=device)

        iou_per_class, m_iou = calculate_iou(inf_bev_mask, real_bev_mask, num_labels)
        mean_iou_per_class['raw2bev_seg'] += iou_per_class.cpu().detach().numpy()
        mean_iou['raw2bev_seg'] += m_iou.cpu().detach().numpy()
        

        # cv2.namedWindow("real_mask", cv2.WINDOW_NORMAL)
        # cv2.imshow("real_mask", raw2seg_bev.mask2image(real_bev_mask))
        # cv2.namedWindow("inf_mask", cv2.WINDOW_NORMAL)
        # cv2.imshow("inf_mask", raw2seg_bev.mask2image(inf_bev_mask))
        # real_labels = np.unique(real_bev_mask) #[raw_seg2bev.id2label[l] for l in np.unique(real_raw_mask)]
        # inf_labels = np.unique(inf_bev_mask) #[raw_seg2bev.id2label[l] for l in np.unique(inf_raw_mask)]
        # print(f"real labels in image: {real_labels}")
        # print(f"inf labels in image: {inf_labels}")
        # cv2.waitKey(0)
        

    for k in mean_iou_per_class.keys():
        mean_iou_per_class[k] /= len(bev_dataset)
    for k in mean_iou.keys():
        mean_iou[k] /= len(bev_dataset)

    print(f"FINAL mean_iou_per_class: {mean_iou_per_class}")
    print(f"FINAL mean_iou: {mean_iou}")

    if save_results and results_path is not None:
        with open(results_path, "wb") as f:
            pickle.dump({'mean_iou': mean_iou, 'mean_iou_per_class': mean_iou_per_class}, f)

    if plot_results:
        plot_mean_iou_per_class(mean_iou_per_class['rawseg'], 'RAW -> SEG mean_iou_per_class', labels_list=labels_list)
        plot_mean_iou_per_class(mean_iou_per_class['raw2seg_bev'], 'RAW -> SEG -> BEV mean_iou_per_class', labels_list=labels_list)
        plot_mean_iou_per_class(mean_iou_per_class['raw2bev_seg'], 'RAW -> BEV -> SEG mean_iou_per_class', labels_list=labels_list)


if __name__ == "__main__":
    BEVDATASET_PATH    = "tmp/BEVDataset"
    NUDATASET_PATH     = "tmp/NuImagesFormatted"
    #RAW2SEGMODEL_PATH  = "models/segformer_nu_formatted_test/overfitted_model_NoReduceLabels"
    #BEV2SEGMODEL_PATH  = "models/segformer_bev_test/overfitted_model_NoReduceLabels"
    RAW2SEGMODEL_PATH  = "models/segformer_nu_formatted/raw2seg_bev_v0.2"
    BEV2SEGMODEL_PATH  = "models/segformer_bev/raw2bevseg_v0.3"
    
    OUT_PATH = "./data/mean_iou.pt"

    main(BEVDATASET_PATH, 
         NUDATASET_PATH, 
         RAW2SEGMODEL_PATH, 
         BEV2SEGMODEL_PATH, 
         dataset_versions='val',
         plot_results=False,
         save_results=True,
         results_path=OUT_PATH)