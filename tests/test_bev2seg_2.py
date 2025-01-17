
from src.bev2seg_2 import Raw_BEV2Seg, Raw2Seg_BEV

# NuImages color palette
#from oldatasets.NuImages.nulabels import nuid2color

from typing import List
import numpy as np
import cv2
import os


from PIL import Image

BEV_DATASET_PATH    = "tmp/BEVDataset"
RAW2SEG_MODEL_PATH  = "models/segformer_nu_formatted/raw2seg_bev_mit-b0_v0.2"
#RAW2SEG_MODEL_PATH  = "models/segformer_nu_formatted_test/overfitted_model_NoReduceLabels"
BEV2SEG_MODEL_PATH  = "models/segformer_bev/raw2bevseg_mit-b0_v0.3"
#BEV2SEG_MODEL_PATH  = "models/segformer_bev_test/overfitted_model_NoReduceLabels"

def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path: {path} doesnt exist")

def test_raw2segbev():
    """Test of the Raw -> Seg -> Bev pipeline"""
    image_id = "60d367ec0c7e445d8f92fbc4a993c67e"
    test_image_path     = os.path.join(BEV_DATASET_PATH, "mini", image_id + "_raw.png") 
    test_openlabel_path = os.path.join(BEV_DATASET_PATH, "mini", image_id + ".json") 
    check_paths([BEV_DATASET_PATH, RAW2SEG_MODEL_PATH, test_image_path, test_openlabel_path])
    
    # Load image
    image = cv2.imread(test_image_path)
    # cv2.imshow("Input Image", image)
    # cv2.waitKey(0)

    # raw2seg_bev instance
    raw2seg_bev = Raw2Seg_BEV(RAW2SEG_MODEL_PATH, test_openlabel_path)
    raw_mask, bev_mask = raw2seg_bev.generate_bev_segmentation(image, 'CAM_FRONT')
    bev_labels = np.unique(bev_mask)
    bev_labelnames = [ raw2seg_bev.id2label[l] for l in bev_labels ]
    raw_color = raw2seg_bev.mask2image(raw_mask)
    bev_color = raw2seg_bev.mask2image(bev_mask)

    print(f"SegFormer id2label:\n{raw2seg_bev.id2label}")
    print(f"SegFormer id2color: {raw2seg_bev.id2color}")
    print(f"unique labels: {bev_labels}")
    print(f"label names: {bev_labelnames}")

    cv2.imshow("Normal Segmentation mask", raw_color)
    cv2.imshow("BEV Segmentation mask", bev_color)
    cv2.waitKey(0)
    

def test_rawbev2seg():
    """Test of the Raw -> Bev -> Seg pipeline"""
    image_id = "60d367ec0c7e445d8f92fbc4a993c67e"
    # image_id = "2940ea9eda6c4cf6b6a30a47d579cd1c"
    test_image_path     = os.path.join(BEV_DATASET_PATH, "mini", image_id + "_raw.png") 
    test_openlabel_path = os.path.join(BEV_DATASET_PATH, "mini", image_id + ".json") 
    check_paths([BEV_DATASET_PATH, BEV2SEG_MODEL_PATH, test_image_path, test_openlabel_path])
    
    # Load image
    image = cv2.imread(test_image_path)
    # cv2.imshow("Input Image", image)
    # cv2.waitKey(0)

    # raw_seg2bev instance
    raw_seg2bev = Raw_BEV2Seg(BEV2SEG_MODEL_PATH, test_openlabel_path)
    
    bev_mask = raw_seg2bev.generate_bev_segmentation(image, 'CAM_FRONT')

    bev_labels = np.unique(bev_mask)
    bev_labelnames = [ raw_seg2bev.id2label[l] for l in bev_labels ]
    bev_color = raw_seg2bev.mask2image(bev_mask)
    
    print(f"SegFormer id2label:\n{raw_seg2bev.id2label}")
    print(f"SegFormer id2color: {raw_seg2bev.id2color}")
    print(f"unique labels: {bev_labels}")
    print(f"label names: {bev_labelnames}")

    cv2.imshow("BEV Segmentation mask", bev_color)
    cv2.waitKey(0)
    # assert False