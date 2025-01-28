
import os
from oldatasets.common import display_image, display_images
from oldatasets.NuImages import NuImagesDataset, NuImagesBEVDataset, generate_BEVDataset_from_NuImages
import numpy as np

NUIMAGES_PATH = "./tests/tmp/NuImages"
NUIMAGES_PATH = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/nuImages"
TMP_DIR = "./tests/tmp/trash"

DISPLAY_IMAGES = False

######################## TESTS ########################
def test_import_datasets():
    if not os.path.exists(TMP_DIR):
        # Create the tmp folder
        os.makedirs(TMP_DIR)
    else:
        # Clear the tmp folder
        files = os.listdir(TMP_DIR)
        for f in files:
            os.remove(os.path.join(TMP_DIR, f))
    assert os.path.exists(NUIMAGES_PATH)
    assert NuImagesBEVDataset is not None

def test_generate_openlabel_one_sample():
    """
    Test for generating and saving one sample of BEVDataset from the NuImages Dataset
    """
    dataset_nu = NuImagesDataset(dataroot=NUIMAGES_PATH, version='mini')
    dataset_bev = NuImagesBEVDataset(
        dataroot=NUIMAGES_PATH,
        version='mini', 
        output_path=TMP_DIR, 
        save_bevdataset=True)

    for i in range(1):
        image_nu, target_nu = dataset_nu.__getitem__(i)
        image_bev, target_bev = dataset_bev.__getitem__(i)
        print(f"labels on normal image: {np.unique(target_nu)}")
        print(f"labels on bev image: {np.unique(target_bev)}")
        # assert np.array_equal(np.unique(target_nu), np.unique(target_bev)) # No tienen porqué ser exactamente iguales

    if DISPLAY_IMAGES:
        target = dataset_bev.target2image(target_bev)
        display_images("test_generate_openlabel_one_sample", [image_bev, target_bev])

def test_load_openlabel_one_sample():
    """
    Test for generating BEVDataset on the loop by reading previously
    generated OpenLABEL files
    """
    assert os.path.exists(TMP_DIR)
    dataset = NuImagesBEVDataset(
        dataroot=NUIMAGES_PATH, 
        output_path=TMP_DIR)
      
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
        display_image("test_load_openlabel_one_sample", image)

def test_distorsion_error_on_sample():
    """
    Con la instancia 38 en 'mini' siempre se genera error. 
    Hay que arreglar esto en caso de que se quiera utilizar
    NuImagesBEVDataset en un Dataloader de pytorch. Para la 
    generación del BEVDataset se puede utilizar la función
    generate_BEVDataset_from_NuImages()
    """

    assert os.path.exists(TMP_DIR)
    dataset = NuImagesBEVDataset(dataroot=NUIMAGES_PATH)
    
    item_index = 38
    assert item_index < len(dataset)
    error = False
    try:
        image, target = dataset.__getitem__(item_index)
    except:
        error = True
    assert not error

def test_generate_BEVDataset():
    """
    Test for generating a complete BEVDataset from a NuImages dataset
    """
    assert os.path.exists(TMP_DIR)

    num_generated, _ = generate_BEVDataset_from_NuImages(
                        dataset_path=NUIMAGES_PATH, 
                        out_path=TMP_DIR, 
                        version='mini')
    
    files = os.listdir(TMP_DIR)

    samples = [os.path.splitext(f)[0] for f in files if f.endswith('.json')]

    assert num_generated == len(samples) 
    assert num_generated == 49 # mini tiene 50 samples pero 1 da error

