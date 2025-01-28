
from typing import List

from oldatasets.BEV import BEVDataset
from oldatasets.NuImages import NuImagesFormattedDataset

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import shutil
import os

"""
´´´
.../BEVDataset/
    mini/
        - token1.json
        - token1_bev.png
        - token1_raw.png
        - token1_color.png
        - token1_semantic.png
        ...
    train/
    test/
´´´

´´´
.../NuImagesFormatted/
    mini/
        - token1_raw.png
        - token1_color.png
        - token1_semantic.png
        ...
    train/
    test/
´´´
"""

class DatasetSplitter():
    SUPPORTED_DATASETS = {
        "BEVDataset": BEVDataset,
        "NuImagesFormattedDataset": NuImagesFormattedDataset
    }
    
    def __init__(self, dataset_paths: List[dict], version:str, set2_size:float = 0.5, random_state:int = None):
        """
        It takes a dataset path as input and generates a split. if multiple paths are provided from supported dataset formats, it ensures
        that the same split is made on those datasets. 
        
        Supported Dataset Formats:
            - BEVDataset
            - NuImagesFormattedDataset

        There are some configuration parameters:
            - dataset_paths: List of dicts with this format [{"dataroot": path, "type": supported_types, "split_dest_path": dest_directory}]
            - version: dataset version from where the split is going to be made.
            - set2_size: size of the division result of the split of the input dataset version. 50% by default.
            - random_state: random seed for shuffling the data.
        """
        
        self.datasets_info = []
        for dpaths in dataset_paths:
            dataroot = dpaths['dataroot']
            split_dest_path = dpaths['split_dest_path']
            dtype = dpaths['type']

            # Check paths and versions
            if not os.path.isdir(dataroot):
                raise Exception(f"Dataset path {dataroot} not found")
            if not os.path.isdir(split_dest_path):
                raise Exception(f"Split destination dir path {split_dest_path} not found")
            if dtype not in DatasetSplitter.SUPPORTED_DATASETS:
                raise Exception(f"Type {dtype} not supported. Supported types are {list(DatasetSplitter.SUPPORTED_DATASETS.keys())}")
            dataset = DatasetSplitter.SUPPORTED_DATASETS[dtype]
            if version not in dataset.DATASET_VERSIONS:
                raise Exception(f"Version {version} not supported by dataset {dtype}")
            data_path = os.path.join(dataroot, version)
            if not os.path.isdir(dataroot):
                raise Exception(f"Data path {data_path} not found. Ensure there exists version {version} in dataset folder")
            files = os.listdir(data_path)
            if not len(files) > 0:
                raise Exception(f"Data path {data_path} is empty")
            
            image_extension = None
            for f in files:
                image_extension = f".{os.path.basename(f).split('.')[-1]}"
                if image_extension not in [".json"]:
                    break
            
            dinfo = {
                "dataroot": dataroot,
                "split_dest_path": split_dest_path,
                "type": dtype,
                "type_obj": dataset,
                "version": version,
                "data_path": data_path,
                "image_extension": image_extension,
                "data_tokens": []
            }
        
            # Load samples from paths
            if dataset == BEVDataset:
                dinfo['data_tokens'] = dataset.get_data_tokens(data_path)
            elif dataset == NuImagesFormattedDataset:
                
                dinfo['data_tokens'] = dataset.get_data_tokens(data_path, image_extension)
            
            # Check if there are samples to load
            if len(dinfo['data_tokens']) == 0:
                raise Exception(f"No data loaded for {dtype} {version}")
            dinfo['data_tokens'].sort() # Sort the list
            self.datasets_info.append(dinfo)

        # Assert the same samples are on all the datasets
        for di in self.datasets_info:
            for dj in self.datasets_info:
                if di == dj:
                    continue # Skip equals
                assert len(di['data_tokens']) == len(dj['data_tokens'])
                for index in range(len(di['data_tokens'])):
                    if di['data_tokens'][index] != dj['data_tokens'][index]:
                        raise Exception(f"Token at index {index} is '{di['data_tokens'][index]}' in {di['type']} but '{dj['data_tokens'][index]}' in {dj['type']}")
        
        # Make the sample index split
        indexes = range(len(self.datasets_info[0]['data_tokens']))
        self.set1_indexes, self.set2_indexes = train_test_split(indexes, test_size=set2_size, shuffle=True, random_state=random_state)
        print(f"[DatasetSplitter]   Loaded {len(self.datasets_info[0]['data_tokens'])} samples")
        # print(f"[DatasetSplitter]   Moving indexes:")
        # print(self.set2_indexes)
        print()

    def move_split_to_path(self):
        """
        The idea is to take the set2 split and move onto another folder dest for all the provided datasets.
        the destination path is provided in the constructor
        For example:
        ```
        ./Dataset1/
            mini/
                - sample_1
                - sample_2
                - sample_3
                - sample_4
            dest/
        
        Result:
        ./Dataset1/
            mini/
                - sample_1
                - sample_2
            dest/
                - sample_3
                - sample_4
        ```
        """

        for dinfo in self.datasets_info:
            print(f"[DatasetSplitter]   Moving samples from {dinfo['type']} {dinfo['version']} to {dinfo['split_dest_path']}")
            for i in tqdm(self.set2_indexes, desc=f"Samples"):
                data_token = dinfo['data_tokens'][i]

                moving_files = []
                if dinfo['type_obj'] == BEVDataset:
                    f1 = os.path.join(dinfo['data_path'], data_token + ".json")
                    f2 = os.path.join(dinfo['data_path'], data_token + "_bev"       + dinfo['image_extension'])
                    f3 = os.path.join(dinfo['data_path'], data_token + "_raw"       + dinfo['image_extension'])
                    f4 = os.path.join(dinfo['data_path'], data_token + "_color"     + dinfo['image_extension'])
                    f5 = os.path.join(dinfo['data_path'], data_token + "_semantic"  + dinfo['image_extension'])
                    moving_files = [f1, f2, f3, f4, f5]
                elif dinfo['type_obj'] == NuImagesFormattedDataset:
                    f1 = os.path.join(dinfo['data_path'], data_token + "_raw"       + dinfo['image_extension'])
                    f2 = os.path.join(dinfo['data_path'], data_token + "_color"     + dinfo['image_extension'])
                    f3 = os.path.join(dinfo['data_path'], data_token + "_semantic"  + dinfo['image_extension'])
                    moving_files = [f1, f2, f3]
                
                for f_src in moving_files:
                    shutil.move(f_src, dinfo['split_dest_path'])
            print()