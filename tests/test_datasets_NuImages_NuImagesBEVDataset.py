
import os
from tests.utils import display_test_image ,display_test_images

from datasets.NuImages import NuImagesBEVDataset, generate_BEVDataset_from_NuImages

NUIMAGES_PATH = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/nuImages"
TMP_DIR = "./tests/tmp/BEVDataset"

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
    dataset = NuImagesBEVDataset(
        dataroot=NUIMAGES_PATH, 
        output_path=TMP_DIR, 
        save_bevdataset=True)

    for i in range(1):
        image, target = dataset.__getitem__(i)
    assert image is not None
    assert target is not None

    if DISPLAY_IMAGES:
        target = dataset.target2image(target)
        display_test_images("test_generate_openlabel_one_sample", [image, target])

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
        display_test_image("test_load_openlabel_one_sample", image)
        #display_test_image("TARGET test_load_openlabel_one_sample", image)

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

