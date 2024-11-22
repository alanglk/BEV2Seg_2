
import os
from common_test_utils import display_test_image ,display_test_images

from datasets.NuImages import NuImagesBEVDataset

NUIMAGES_PATH = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/nuImages"
TMP_DIR = "./tests/tmp/NuImages/OpenLABEL"

DISPLAY_IMAGES = True

######################## TESTS ########################
def test_import_datasets():
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    assert os.path.exists(NUIMAGES_PATH)
    assert NuImagesBEVDataset is not None

def test_generate_openlabel_one_sample():
    dataset = NuImagesBEVDataset(
        dataroot=NUIMAGES_PATH, 
        openlabelroot=TMP_DIR, 
        save_openlabel=True)

    for i in range(1):
        image, target = dataset.__getitem__(i)
    assert image is not None
    assert target is not None

    if DISPLAY_IMAGES:
        target = dataset.target2image(target)
        display_test_images("test_generate_openlabel_one_sample", [image, target])

def test_load_openlabel_one_sample():
    assert os.path.exists(TMP_DIR)
    dataset = NuImagesBEVDataset(
        dataroot=NUIMAGES_PATH, 
        openlabelroot=TMP_DIR)
      
    num_images = 0
    ex_found = False # no ex :D !!
    for i in range(3):
        try:
            image, target = dataset.__getitem__(i)
            num_images += 1
        except:
             # This is normal as just the first image has
             # a generated openlabel file from the previous
             # test
             ex_found = True

    assert num_images == 1 and ex_found

    if DISPLAY_IMAGES:
        display_test_image("test_load_openlabel_one_sample", image)
        #display_test_image("TARGET test_load_openlabel_one_sample", image)
