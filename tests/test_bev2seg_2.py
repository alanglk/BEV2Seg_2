
from src.bev2seg_2 import Raw_BEV2Seg#, Raw2Seg_BEV

# NuImages color palette
from oldatasets.NuImages.nulabels import nuid2color

from typing import List
import numpy as np
import cv2
import os


from PIL import Image

BEV_DATASET_PATH    = "tmp/BEVDataset"
RAW2SEG_MODEL_PATH  = "tmp/models/segformer_bev_test/checkpoint-9" # Change this
BEV2SEG_MODEL_PATH  = "tmp/models/segformer_bev_test/checkpoint-9"

def check_paths(paths: List[str]):
    for path in paths:
        assert os.path.exists(path)

# def test_raw2segbev():
#     """Test of the Raw -> Seg -> Bev pipeline"""
#     image_id = "60d367ec0c7e445d8f92fbc4a993c67e"
#     test_image_path     = os.path.join(BEV_DATASET_PATH, "mini", image_id + "_raw.png") 
#     test_openlabel_path = os.path.join(BEV_DATASET_PATH, "mini", image_id + ".json") 
#     check_paths([BEV_DATASET_PATH, RAW2SEG_MODEL_PATH, test_image_path, test_openlabel_path])
#     
#     # Load image
#     image = Image.open(test_image_path)
# 
#     # raw2seg_bev instance
#     raw2seg_bev = Raw2Seg_BEV(RAW2SEG_MODEL_PATH, test_openlabel_path)
#     bev_mask = raw2seg_bev.generate_bev_segmentation(image, 'CAM_FRONT')
# 
#     cv2.imshow("Segmentation mask", raw2seg_bev.mask2image(bev_mask, nuid2color))
#     cv2.waitKey(0)
    

def test_rawbev2seg():
    """Test of the Raw -> Bev -> Seg pipeline"""
    image_id = "60d367ec0c7e445d8f92fbc4a993c67e"
    test_image_path     = os.path.join(BEV_DATASET_PATH, "mini", image_id + "_raw.png") 
    test_openlabel_path = os.path.join(BEV_DATASET_PATH, "mini", image_id + ".json") 
    check_paths([BEV_DATASET_PATH, RAW2SEG_MODEL_PATH, test_image_path, test_openlabel_path])
    
    # Load image
    image = cv2.imread(test_image_path)
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)

    # raw_seg2bev instance
    raw_seg2bev = Raw_BEV2Seg(RAW2SEG_MODEL_PATH, test_openlabel_path)
    bev_mask = raw_seg2bev.generate_bev_segmentation(image, 'CAM_FRONT')

    cv2.imshow("BEV Segmentation mask", raw_seg2bev.mask2image(bev_mask, nuid2color))
    cv2.waitKey(0)
    