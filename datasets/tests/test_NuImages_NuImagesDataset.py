
import os
from .utils import display_test_image ,display_test_images

from datasets.NuImages import NuImagesDataset

NUIMAGES_PATH = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/nuImages"

DISPLAY_IMAGES = False

######################## TESTS ########################
def test_import_datasets():
    assert os.path.exists(NUIMAGES_PATH)
    assert NuImagesDataset is not None

def test_get_one_sample():
    dataset = NuImagesDataset(dataroot=NUIMAGES_PATH)

    for i in range(1):
        image, target = dataset.__getitem__(i)
    assert image is not None
    assert target is not None

    if DISPLAY_IMAGES:
        target = dataset.target2image(target)
        display_test_images("test_get_one_sample", [image, target])

def test_get_multiple_samples():
    dataset = NuImagesDataset(
        dataroot=NUIMAGES_PATH, 
        version='mini')
    assert len(dataset) == 50 # just 50 on the mini dataset
    
    num_samples = len(dataset)
    i = 0
    for i in range(num_samples):
        image, target = dataset.__getitem__(i)
    
    assert i == num_samples-1

def test_generate_CAM_FRONT():
    """
    Generate samples from an specific channel (camera)
    """
    assert os.path.exists(NUIMAGES_PATH)

    dataset = NuImagesDataset(
        dataroot=NUIMAGES_PATH, 
        version='mini',
        camera='CAM_FRONT')
    
    num_samples = len(dataset)
    i = 0
    for i in range(num_samples):
        image, target = dataset.__getitem__(i)

        if DISPLAY_IMAGES:
            target = dataset.target2image(target)
            display_test_images("test_get_one_sample", [image, target])
    
    print(f"Num CAM_FRONT images: {num_samples}")
    assert i == num_samples-1 # 'mini' has 8 CAM_FRONT images