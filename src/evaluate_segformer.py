from oldatasets.BEV import BEVDataset
from oldatasets.NuImages import NuImagesDataset
from oldatasets.NuImages import NuImagesFormattedDataset
from oldatasets.NuImages.nulabels import nuid2name as id2label

from bev2seg_2 import Raw2Seg_BEV, Raw_BEV2Seg

import evaluate
import cv2

from vcd import core
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

num_labels = len(id2label)
mean_iou = evaluate.load("mean_iou")

def check_paths(paths: List[str]):
    for path in paths:
        assert os.path.exists(path)

BEV_DATASET_PATH    = "tmp/BEVDataset"
NU_DATASET_PATH     = "tmp/BEVDataset"
RAW2SEG_MODEL_PATH  = "models/segformer_nu_formatted_test/overfitted_model_NoReduceLabels"
BEV2SEG_MODEL_PATH  = "models/segformer_bev_test/overfitted_model_NoReduceLabels"
BEV2SEG_MODEL_PATH  = "models/segformer_bev/raw2bevseg_v0.2"

check_paths([BEV_DATASET_PATH, NU_DATASET_PATH, RAW2SEG_MODEL_PATH, BEV2SEG_MODEL_PATH])


bev_dataset     = BEVDataset(dataroot=BEV_DATASET_PATH, version='mini')
nu_dataset      = NuImagesFormattedDataset(dataroot=NU_DATASET_PATH, version='mini')

raw2seg_bev = Raw2Seg_BEV(RAW2SEG_MODEL_PATH, None)
raw_seg2bev = Raw_BEV2Seg(BEV2SEG_MODEL_PATH, None)
labels_list = list(raw2seg_bev.id2label.values())
colors      = list(raw2seg_bev.id2color.values())
colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

assert len(bev_dataset) == len(nu_dataset)
for index in range(len(bev_dataset)):
    # Get paths
    raw_image_path, real_raw_mask_path = nu_dataset._get_item_paths(index)
    bev_image_path, real_bev_mask_path = bev_dataset._get_item_paths(index)
    openlabel_path = os.path.join(bev_dataset.dataroot, bev_dataset.data_tokens[index] + ".json")

    # Load files
    vcd = core.VCD()
    vcd.load_from_file(openlabel_path)
    raw_image       = cv2.imread(raw_image_path)
    real_raw_mask   = cv2.imread(real_raw_mask_path)[:, :, 0]
    real_bev_mask   = cv2.imread(real_bev_mask_path)[:, :, 0]

    # cv2.namedWindow("raw_image", cv2.WINDOW_NORMAL)
    # cv2.imshow("raw_image", raw_image)

    # Raw -> Seg -> BEV
    # comparar inf_raw_mask con real_raw_mask e inf_bev_mask con real_bev_mask
    # raw2seg_bev.set_openlabel(vcd)
    # inf_raw_mask, inf_bev_mask = raw2seg_bev.generate_bev_segmentation(raw_image, 'CAM_FRONT')
    # cv2.namedWindow("raw2seg_bev-inf_raw_mask", cv2.WINDOW_NORMAL)
    # cv2.imshow("raw2seg_bev-inf_raw_mask", raw2seg_bev.mask2image(inf_raw_mask))
    # cv2.namedWindow("raw2seg_bev-inf_bev_mask", cv2.WINDOW_NORMAL)
    # cv2.imshow("raw2seg_bev-inf_bev_mask", raw2seg_bev.mask2image(inf_bev_mask))
    # results_rawseg = mean_iou.compute(predictions=inf_raw_mask, references=real_raw_mask, num_labels=num_labels, ignore_index=255, reduce_labels=False)
    # results_rawsegbev = mean_iou.compute(predictions=inf_bev_mask, references=real_bev_mask, num_labels=num_labels, ignore_index=255, reduce_labels=False)

    # Raw -> BEV -> Seg
    # Comparar inf_bev_mask con real_bev_mask  
    raw_seg2bev.set_openlabel(vcd)
    inf_bev_mask = raw_seg2bev.generate_bev_segmentation(raw_image, 'CAM_FRONT')
    results_rawbevseg = mean_iou.compute(predictions=inf_bev_mask, references=real_bev_mask, num_labels=num_labels, ignore_index=255, reduce_labels=False)
    print(results_rawbevseg)
    
    print(np.unique(real_bev_mask))
    print(np.unique(inf_bev_mask))
    real_labels = [raw_seg2bev.id2label[l] for l in np.unique(real_bev_mask)]
    inf_labels = [raw_seg2bev.id2label[l] for l in np.unique(inf_bev_mask)]
    print(f"real labels in image: {real_labels}")
    print(f"inf labels in image: {inf_labels}")


    
    iou_per_category = results_rawbevseg['per_category_iou']
    plt.figure(figsize=(10, 6))
    plt.title('Raw -> BEV -> Seg')
    plt.bar(labels_list, iou_per_category, color=colors)
    # plt.bar(range(len(iou_per_category)), iou_per_category, color=colors)
    plt.xlabel('Classes')
    plt.xticks(rotation=45)
    plt.ylabel('IoU')

    # for i, v in enumerate(iou_per_category):
    #     if v > 0:
    #         label = labels_list[i]
    #         plt.text(i, v + 0.02, f'{label}: {v:.2f}', ha='center', va='bottom')
    plt.show()
    # cv2.namedWindow("raw_seg2bev-inf_bev_mask", cv2.WINDOW_NORMAL)
    # cv2.imshow("raw_seg2bev-inf_bev_mask", raw_seg2bev.mask2image(inf_bev_mask))
    # cv2.waitKey(0)

