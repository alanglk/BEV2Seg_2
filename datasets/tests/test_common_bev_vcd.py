import os
import cv2
from vcd import core, scl
from .utils import display_test_image

from datasets.common import Dataset2BEV

NUIMAGES_PATH   = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/nuImages"
TMP_DIR         = "./tests/tmp/BEVDataset"
SAMPLE_IMAGE    = "samples/CAM_FRONT_RIGHT/n013-2018-08-27-14-41-26+0800__CAM_FRONT_RIGHT__1535352274870176.jpg"
SAMPLE_TOKEN    = "0128b121887b4d0d86b8b1a43ac001e9"
SAMPLE_CAM      = "CAM_FRONT_RIGHT" 

DISPLAY_IMAGES = True

######################## TESTS ########################
def test_img2bev():
    sample_image = os.path.join(NUIMAGES_PATH, SAMPLE_IMAGE)
    sample_vcd   = os.path.join(TMP_DIR, SAMPLE_TOKEN + ".json") 
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
        display_test_image("test_img2bev", samble_bev)
