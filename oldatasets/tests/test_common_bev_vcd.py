import os
import cv2
from vcd import core, scl

from oldatasets.common import Dataset2BEV, display_image

BEV_DATASET         = "./tests/tmp/BEVDataset"
NUIMAGESFORMATTED   = "./tests/tmp/NuImagesFormatted"
SAMPLE_TOKEN        = "3e422fcb9c2a49639ca24c8ff43f3d67"
SAMPLE_CAM          = "CAM_FRONT" 

DISPLAY_IMAGES = True

######################## TESTS ########################
def test_img2bev():
    sample_image = os.path.join(BEV_DATASET, 'mini', SAMPLE_TOKEN + "_raw.png")
    sample_vcd   = os.path.join(BEV_DATASET, 'mini', SAMPLE_TOKEN + ".json") 
    assert os.path.exists(sample_image)
    assert os.path.exists(sample_vcd)

    # Read data
    sample_image = cv2.imread(sample_image)
    vcd = core.VCD()
    vcd.load_from_file(sample_vcd)
    sample_scene = scl.Scene(vcd)

    d2bev = Dataset2BEV(
        cam_name=SAMPLE_CAM, 
        scene=sample_scene,
        bev_max_distance=30)
    
    samble_bev = d2bev._img2bev(sample_image, framenum=0)
    assert samble_bev is not None

    # Display BEV image
    if DISPLAY_IMAGES:
        display_image("test_img2bev", samble_bev)