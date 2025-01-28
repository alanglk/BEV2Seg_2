from oldatasets.postprocessing import DatasetSplitter
import os

BEV_DATASET         = "./tests/tmp/BEVDataset"
NUIMAGESFORMATTED   = "./tests/tmp/NuImagesFormatted"

def test_dataset_splitter():
    bev_split_out   = os.path.join(BEV_DATASET, 'val')
    nu_split_out    = os.path.join(NUIMAGESFORMATTED, 'val')
    os.makedirs(bev_split_out)
    os.makedirs(nu_split_out)

    dataset_paths = [
        {"dataroot": BEV_DATASET, "type": "BEVDataset", "split_dest_path": bev_split_out},
        {"dataroot": NUIMAGESFORMATTED, "type": "NuImagesFormattedDataset", "split_dest_path": nu_split_out}
    ]
    dataset_splitter = DatasetSplitter(dataset_paths, version='mini')

    dataset_splitter.move_split_to_path()