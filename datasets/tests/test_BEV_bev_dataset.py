from datasets.BEV import BEVDataset
from .utils import display_test_images

TMP_DIR = "./tests/tmp/BEVDataset"

def test_load_bev_dataset():
    """
    Test for loading a previously generated BEVDataset
    """
    dataset = BEVDataset(dataroot=TMP_DIR)
    return
    for i in range(len(dataset)):
        image, target = dataset.__getitem__(i)
        display_test_images("test_load_bev_dataset", [image, target])