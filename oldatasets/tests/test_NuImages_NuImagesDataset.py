
import os

from oldatasets.common import display_images
from oldatasets.NuImages import NuImagesDataset

NUIMAGES_PATH = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/nuImages"
OUTPUT_PATH = "./tests/tmp/NuImages/mini"

DISPLAY_IMAGES = True

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
        display_images("test_get_one_sample", [image, target])
    
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
    assert os.path.exists(OUTPUT_PATH)


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
            display_images("test_get_one_sample", [image, target])
    
    print(f"Num CAM_FRONT images: {num_samples}")
    assert i == num_samples-1 # 'mini' has 8 CAM_FRONT images

def test_save_generated_CAM_FRONT():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH) # Create the tmp folder
    else:
        files = os.listdir(OUTPUT_PATH) # Clear the tmp folder
        for f in files:
            os.remove(os.path.join(OUTPUT_PATH, f))
    
    assert os.path.exists(NUIMAGES_PATH)
    assert os.path.exists(OUTPUT_PATH)

    dataset = NuImagesDataset(
        dataroot=NUIMAGES_PATH, 
        version='mini',
        camera='CAM_FRONT',
        save_dataset=True,
        output_path=OUTPUT_PATH)
    
    num_samples = len(dataset)
    i = 0
    for i in range(num_samples):
        image, target = dataset.__getitem__(i)

        if DISPLAY_IMAGES:
            target = dataset.target2image(target)
            display_images("test_get_one_sample", [image, target])
    
    files = os.listdir(OUTPUT_PATH)
    samples = [os.path.splitext(f)[0] for f in files if f.endswith('_raw.png')]

    assert len(samples) == 8 # Se han generado los 8 CAM_FRONT de mini