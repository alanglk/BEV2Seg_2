import torch
from torch.utils.data import Dataset, DataLoader

from datasets.NuImages.nulabels import nuid2name

from PIL import Image
import cv2
import os

class BEVDataset(Dataset):
    """
    Loads a previously generated BEVDataset. The expected structure is:
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
    BEVDataset requires the following structure despite OpenLABEL files are not neccesary
    for training Networks.
    """
    def __init__(self, dataroot, version, image_extension = '.png', transforms = None, id2label = nuid2name):
        """
        INPUT:
            - dataroot: root path of the BEVDataset
            - version: ['mini', 'train', 'val', 'test']
            - image_extension: extension of image files
            - transforms: torchvision transforms
            - id2label: {0: "road"...}. By default is the NuImages id2label
        """
        super().__init__()
        dataroot = os.path.join(dataroot, version)
        self.dataroot = os.path.abspath(dataroot)
        self.version = version
        self.image_extension = image_extension
        self.transforms = transforms
        self.data_tokens = [] # All the tokens in the dataset

        # Save the id label mapping
        self.id2label = id2label

        # Load all the tokens from the dataroot folder
        if os.path.isdir(self.dataroot):
            files = os.listdir(self.dataroot)
            self.data_tokens = [os.path.splitext(f)[0] for f in files if f.endswith('.json')]
        else:
            raise Exception(f"BEVDataset path not found: {self.dataroot}")
        
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
        
        bev_path = os.path.join(self.dataroot, sample_token + "_bev" + self.image_extension)
        semantic_path = os.path.join(self.dataroot, sample_token + "_semantic" + self.image_extension)

        if not os.path.isfile(bev_path):
            raise Exception(f"Image file file not found: {bev_path}")

        if not os.path.isfile(semantic_path):
            raise Exception(f"Semantic mask file not found: {semantic_path}")
        
        image   = torch.tensor( cv2.imread(bev_path) )      # BGR
        target  = torch.tensor( cv2.imread(semantic_path) ) # BGR

        # Apply transforms if necessary
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

class BEVFeatureExtractionDataset(BEVDataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, dataroot, version, feature_extractor, image_extension='.png', transforms=None, id2label=...):
        super().__init__(dataroot, version, image_extension, transforms, id2label)
        self.feature_extractor = feature_extractor
    
    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        encoded_inputs = self.feature_extractor(image, target, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()

        return encoded_inputs