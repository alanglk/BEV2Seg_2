import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import os

class BEVDataset(Dataset):
    """
    Loads a previously generated BEVDataset. The expected structure is:
    ´´´
    /dataroot/
        - token1.json
        - token1_raw.png
        - token1_color.png
        - token1_semantic.png
        ...
    ´´´
    BEVDataset requires the following structure despite OpenLABEL files are not neccesary
    for training Networks.
    """
    def __init__(self, dataroot = '/data/sets/nuimages', image_extension = '.png', transforms = None):
        """
        INPUT:
            - dataroot: root path of the BEVDataset
            - image_extension: extension of image files
            - transforms: torchvision transforms
        """
        super().__init__()
        self.dataroot = dataroot
        self.image_extension = image_extension
        self.transforms = transforms
        self.data_tokens = [] # All the tokens in the dataset

        # Load all the tokens from the dataroot folder
        if os.path.isdir(self.dataroot):
            files = os.listdir(dataroot)
            tokens = [os.path.splitext(f)[0] for f in files if files.endswith('.json')]
            self.data_tokens = tokens
        
    def __len__(self):
        return len(self.data_tokens)
    
    def __getitem__(self, index):
        """
        INPUT:
            Index of the current sample in the dataset
        OUTPUT:
            image   -> BEV torch.Tensor (H, W, 3) # BGR
            target  -> BEV annotations of the image ("mask"): torch.Tensor (H, W)
        """
        assert index < len(self)
        
        sample_token = self.data_tokens[index]
        
        image_path = os.path.join(self.dataroot, sample_token + self.image_extension)
        semantic_path = os.path.join(self.dataroot, sample_token + "_semantic" + self.image_extension)

        if not os.path.isfile(image_path) or not os.path.isfile(semantic_path):
            raise Exception(f"Image file or Semantic mask file not found: {image_path} | {semantic_path}")
        
        image   = torch.tensor( cv2.imread(image_path).convert("RGB") )
        target  = torch.tensor( cv2.imread(semantic_path).convert("RGB") )

        # Apply transforms if necessary
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target