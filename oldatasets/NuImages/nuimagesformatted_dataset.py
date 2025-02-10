import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from oldatasets.NuImages.nulabels import nulabels, nuname2label, nuid2name, nuid2color
from oldatasets.common import Dataset2BEV, progress_bar, target2image

import os
from typing import Union, List

class NuImagesFormattedDataset(Dataset):
    """
    Loads a parsed NuImagesFormatted Dataset:
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
    DATASET_VERSIONS = ['mini', 'train', 'val', 'test']

    @staticmethod
    def get_data_tokens(data_path:str, file_extension:str = '.png') -> List:
        """
        Return the list of sample tokens of a NuImagesFormatted Dataset 
        """
        # Load all the tokens from the dataroot folder
        if not os.path.isdir(data_path):
            raise Exception(f"NuImagesFormatted data path not found: {data_path}")
        files = os.listdir(data_path)
        data_tokens = [os.path.splitext(f)[0].replace('_raw', '') for f in files if f.endswith('_raw' + file_extension)]
        return data_tokens

    def __init__(self, dataroot, version, image_extension = '.png', transforms = None, id2label = nuid2name, id2color = nuid2color):
        """
        BGR format!!!

        INPUT:
            - dataroot: root path of the BEVDataset
            - version: ['mini', 'train', 'val', 'test']
            - image_extension: extension of image files
            - transforms: torchvision transforms
            - id2label: {0: "road"...}. By default is the NuImages id2label
            - id2color: {0: rgb, 1: rgb...}. By default is the NuImages id2color
        """
        super().__init__()
        assert version in NuImagesFormattedDataset.DATASET_VERSIONS

        dataroot = os.path.join(dataroot, version)
        self.dataroot = os.path.abspath(dataroot)
        self.version = version
        self.image_extension = image_extension
        self.transforms = transforms
        self.data_tokens = [] # All the tokens in the dataset

        # Save the id label mapping
        self.id2label = id2label
        self.label2id = { v : k for k, v in self.id2label.items() }
        self.id2color = id2color

        # Load all the tokens from the dataroot folder
        self.data_tokens = NuImagesFormattedDataset.get_data_tokens(self.dataroot, self.image_extension)
        
    def __len__(self):
        return len(self.data_tokens)
    
    def target2image(self, target: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Converts target (seg mask) into BGR image
        - target: torch.tensor or numpy.ndarray (H, W, 3)

        returns BGR image!!!
        """ 
        #target_1C = target[:, :, 0] # Get just the first channel
        return target2image(target, self.id2color)

    def _get_item_paths(self, index):
        """
        INPUT:
            Index of the current sample in the dataset
        OUTPUT:
            raw_path    -> path of the raw image
            target_path -> path of the annotations 
        """
        assert index < len(self)
        
        sample_token = self.data_tokens[index]
        
        raw_path = os.path.join(self.dataroot, sample_token + "_raw" + self.image_extension)
        semantic_path = os.path.join(self.dataroot, sample_token + "_semantic" + self.image_extension)

        if not os.path.isfile(raw_path):
            raise Exception(f"Image file file not found: {raw_path}")

        if not os.path.isfile(semantic_path):
            raise Exception(f"Semantic mask file not found: {semantic_path}")

        return raw_path, semantic_path

    def __getitem__(self, index):
        """
        INPUT:
            Index of the current sample in the dataset
        OUTPUT:
            image   -> torch.Tensor (H, W, 3) # RGB
            target  -> annotations of the image ("mask"): torch.Tensor (H, W)
        """
        bev_path, semantic_path = self._get_item_paths(index)

        image   = Image.open(bev_path)      # RGB
        target  = Image.open(semantic_path) # RGB
        
        # Apply transforms if necessary
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # image   = torch.tensor( np.array( image ) )
        # target  = torch.tensor( np.array( target ) )

        return image, target

import cv2
class NuImagesFormattedFeatureExtractionDataset(NuImagesFormattedDataset):
    """Image (semantic) segmentation dataset. BGR Format!!!"""

    def __init__(self, dataroot, version, image_processor, image_extension='.png', transforms=None, id2label=nuid2name):
        super().__init__(dataroot, version, image_extension, transforms, id2label)
        self.image_processor = image_processor
        
        if image_processor.do_reduce_labels:
            self.id2label = {k-1: v for k, v in self.id2label.items()}
            self.label2id = {k: v-1 for k, v in self.label2id.items()}
            self.id2color = {k-1: v for k, v in self.id2color.items()}
        
        self.id2label[255] = 'ignore'
        self.label2id['ignore'] = 255
        # self.id2color[255] = (255, 255, 255)
    
    def __getitem__(self, index):
        """
        INPUT:
            Index of the current dataset sample
        OUTPUT:
            encoded bev image/target as follows:
            encoded_inputs = {
                "pixel_values": BGR image!!!
                "labels": target
            }
        """
        # image, target = super().__getitem__(index)
        raw_path, semantic_path = self._get_item_paths(index)
        image   = Image.open(raw_path)      # RGB (1024, 1024, 3)
        target  = Image.open(semantic_path) # RGB (1024, 1024, 3)
        
        # cv2.namedWindow("DEBUG_IMAGE", cv2.WINDOW_NORMAL)
        # cv2.imshow("DEBUG_IMAGE", cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        # Data Augmentations
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        # cv2.imshow("DEBUG_IMAGE", cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)


        # Perform data preparation with image_processor 
        # (it shoul be from transformers:SegformerImageProcessor)
        encoded_inputs = self.image_processor(image, target, return_tensors="pt")
        
        # Remove the batch_dim from each sample
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()

        return encoded_inputs