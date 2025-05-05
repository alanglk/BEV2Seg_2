import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as v1
import torchvision.transforms.v2 as v2

from oldatasets.NuImages.nulabels import nuid2name, nuid2color
from oldatasets.common import target2image

from typing import Union, List

# Imports para chapuza dataAugmentations
from vcd import core, scl, utils, draw
import matplotlib.pyplot as plt
from oldatasets.common import Dataset2BEV

import numpy as np
from PIL import Image
import cv2
import os
import time

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
            - token1_raw_semantic.png (for data Augmentations)
            ...
        train/
        test/
    ´´´
    BEVDataset requires the following structure despite OpenLABEL files are not neccesary
    for training Networks.
    """
    DATASET_VERSIONS = ['mini', 'train', 'val', 'test']
    @staticmethod
    def get_data_tokens(data_path:str) -> List:
        """
        Return the list of sample tokens of a BEVDataset 
        """
        # Load all the tokens from the dataroot folder
        if not os.path.isdir(data_path):
            raise Exception(f"BEVDataset data path not found: {data_path}")
        files = os.listdir(data_path)
        data_tokens = [os.path.splitext(f)[0] for f in files if f.endswith('.json')]
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
        dataroot = os.path.join(dataroot, version)
        assert version in BEVDataset.DATASET_VERSIONS

        self.dataroot = os.path.abspath(dataroot)
        self.version = version
        self.image_extension = image_extension
        
        self.transforms = transforms
        self.prev_to_bev_transform = False 
        # If prev_to_bev_transform is set
        # apply custom geometric transform by modifiying the camera extrinsic parameters before reprojecting to BEV
        if self.transforms is not None and isinstance(self.transforms, dict):
            assert 'rx' in self.transforms 
            assert 'ry' in self.transforms 
            assert 'rz' in self.transforms 
            if 'multiple_rotations' not in self.transforms:
                self.transforms['multiple_rotations'] = False
            if 'use_random' not in self.transforms:
                self.transforms['use_random'] = True
            if 'save_scene_path' not in self.transforms:
                self.transforms['save_scene_path'] = None
            self.prev_to_bev_transform = True
        
        self.data_tokens = [] # All the tokens in the dataset

        # Save the id label mapping
        self.id2label = id2label
        self.label2id = { v : k for k, v in self.id2label.items() }
        self.id2color = id2color

        # Load all the tokens from the dataroot folder
        if os.path.isdir(self.dataroot):
            self.data_tokens = self.get_data_tokens(self.dataroot)
        
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

    def _data_augmentation(self, image:Image, target:Image, vcd:core.OpenLABEL, camera_name:str='CAM_FRONT'):
        """
        Codigo guarro de cojones. BEVDataset no estaba pensado para utilizar el openlabel
        de las imágenes.

        INPUT: PIL image and target (en raw)
        OUTPUT: PIL augmented image and target (en bev)
        """
        
        coord_sys = vcd.get_coordinate_system(camera_name)
        T_4x4 = np.array(coord_sys['pose_wrt_parent']['matrix4x4']).reshape(4, 4)
        R = T_4x4[:3, :3].reshape((3, 3))
        C = T_4x4[:3, 3].reshape((3, 1))

        # Modificar extrínsecos de la cámara
        rvec = utils.R2rvec(R).flatten().tolist() # [rx, ry, rz]
        rx, ry, rz = 0.0, 0.0, 0.0
        if self.transforms['multiple_rotations']:
            if self.transforms['use_random']:
                rx = np.random.uniform(self.transforms['rx'][0], self.transforms['rx'][1])
                ry = np.random.uniform(self.transforms['ry'][0], self.transforms['ry'][1])
                rz = np.random.uniform(self.transforms['rz'][0], self.transforms['rz'][1])
            else:
                rx = np.abs( self.transforms['rx'][0] ) if isinstance(self.transforms['rx'], (tuple, list)) else float(self.transforms['rx']) 
                ry = np.abs( self.transforms['ry'][0] ) if isinstance(self.transforms['ry'], (tuple, list)) else float(self.transforms['ry']) 
                rz = np.abs( self.transforms['rz'][0] ) if isinstance(self.transforms['rz'], (tuple, list)) else float(self.transforms['rz']) 
        else:
            sel = np.random.randint(0, 3)
            if sel == 0:
                rx = np.random.uniform(self.transforms['rx'][0], self.transforms['rx'][1])
            elif sel == 1:
                ry = np.random.uniform(self.transforms['ry'][0], self.transforms['ry'][1])
            elif sel == 2:
                rz = np.random.uniform(self.transforms['rz'][0], self.transforms['rz'][1])

        rvec[0] += rx # Roll
        rvec[1] += ry # Pitch 
        rvec[2] += rz # Yaw

        R = utils.euler2R(rvec)
        T_4x4 = utils.create_pose(R, C)
        vcd.data['openlabel']['coordinate_systems'][camera_name]['pose_wrt_parent']['matrix4x4'] = T_4x4.flatten()

        # Generar BEV image y target
        scene = scl.Scene(vcd)

        if self.transforms['save_scene_path'] is not None:
            setup_viewer = draw.SetupViewer(scene=scene, coordinate_system="vehicle-iso8855")
            fig = setup_viewer.plot_setup()
            fig.savefig(self.transforms['save_scene_path'])

        dbev = Dataset2BEV(cam_name=camera_name, scene=scene)
        
        # Transformar a BEV
        image_bev, target_bev = dbev.convert2bev(np.array(image), np.array(target))
        image_bev   = Image.fromarray(image_bev)
        target_bev  = Image.fromarray(target_bev)

        return image_bev, target_bev

    def _get_item_paths(self, index):
        """
        INPUT:
            Index of the current sample in the dataset
        OUTPUT:
            bev_path    -> path of the bev image
            target_path -> path of the annotations 
        """
        assert index < len(self)
        sample_token = self.data_tokens[index]
        bev_path = os.path.join(self.dataroot, sample_token + "_bev" + self.image_extension)
        semantic_path = os.path.join(self.dataroot, sample_token + "_semantic" + self.image_extension)

        if not os.path.isfile(bev_path):
            raise Exception(f"Image file file not found: {bev_path}")
        if not os.path.isfile(semantic_path):
            raise Exception(f"Semantic mask file not found: {semantic_path}")
        return bev_path, semantic_path

    def _get_item_raw_paths(self, index):
        """
        Más chapuzas de época para el data Augmentation
        """
        assert index < len(self)
        sample_token = self.data_tokens[index]
        raw_path = os.path.join(self.dataroot, sample_token + "_raw" + self.image_extension)
        raw_semantic_path = os.path.join(self.dataroot, sample_token + "_raw_semantic" + self.image_extension)

        if not os.path.isfile(raw_path):
            raise Exception(f"Image file file not found: {raw_path}")
        if not os.path.isfile(raw_semantic_path):
            raise Exception(f"Semantic mask file not found: {raw_semantic_path}")
        return raw_path, raw_semantic_path

    def _get_item_openlabel_path(self, index):
        """
        Used for data geometric augmentations
        INPUT:
            Index of the current sample in the dataset
        OUTPUT:
            openlabel_path
        """
        assert index < len(self)
        sample_token = self.data_tokens[index]
        openlabel_path = os.path.join(self.dataroot, sample_token + ".json")

        if not os.path.isfile(openlabel_path):
            raise Exception(f"OpenLABEL file file not found: {openlabel_path}")
        return openlabel_path

    def __getitem__(self, index):
        """
        INPUT:
            Index of the current sample in the dataset
        OUTPUT:
            image   -> BEV torch.Tensor (H, W, 3) # BGR
            target  -> BEV annotations of the image ("mask"): torch.Tensor (H, W)
        """
        
        if self.transforms is not None and self.prev_to_bev_transform:
            raw_path, raw_semantic_path = self._get_item_raw_paths(index)
            openlabel_path = self._get_item_openlabel_path(index)
            vcd = core.OpenLABEL()
            
            # Load files
            image   = Image.open(raw_path)          # BGR
            target  = Image.open(raw_semantic_path) # BGR
            vcd.load_from_file( openlabel_path )

            # Data Augmentation and transform to BEV
            image, target = self._data_augmentation(image, target, vcd, camera_name='CAM_FRONT')            
            return image, target

        bev_path, semantic_path = self._get_item_paths(index)
        image   = Image.open(bev_path)      # BGR
        target  = Image.open(semantic_path) # BGR

        # Apply transforms if necessary
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        # image   = torch.tensor( np.array( image ) )
        # target  = torch.tensor( np.array( target ) )
        return image, target

class BEVFeatureExtractionDataset(BEVDataset):
    """Image (semantic) segmentation dataset. BGR Format!!!"""

    def __init__(self, dataroot, version, image_processor, image_extension='.png', transforms=None, id2label=nuid2name, id2color=nuid2color, merging_lut_ids:dict = None):
        super().__init__(dataroot, version, image_extension, transforms, id2label, id2color)
        self.image_processor = image_processor
        # IMPORTANTE:
        #   Se considera que 0 es el background, pero en nuestro caso
        #   no queremos la clase background. Cuando generamos el BEVDataset se
        #   pone como 0 las regiones que no interesan así que hay que mappear los
        #   0s a 255 ("ignore")
        
        if image_processor.do_reduce_labels:
            self.id2label = {k-1: v for k, v in self.id2label.items()}
            self.label2id = {k: v-1 for k, v in self.label2id.items()}
            self.id2color = {k-1: v for k, v in self.id2color.items()}
        
        self.id2label[255] = 'ignore'
        self.label2id['ignore'] = 255
        # self.id2color[255] = (255, 255, 255)
        self.merging_lut_ids = merging_lut_ids
    
    def merge_semantic_labels(self, semantic_mask:Image.Image):
        if isinstance(semantic_mask, Image.Image):
            semantic_mask = np.array(semantic_mask)

        # Merge labels
        for src_id, res_id in self.merging_lut_ids.items():
            semantic_mask[semantic_mask == src_id] = res_id

        return Image.fromarray( semantic_mask )

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

        # Load BEV Image and BEV target
        if self.transforms is not None and self.prev_to_bev_transform:
            # Custom Data Augmentations from Raw Image
            raw_path, raw_semantic_path = self._get_item_raw_paths(index)
            openlabel_path = self._get_item_openlabel_path(index)
            vcd = core.OpenLABEL()
            
            # Load files
            image   = Image.open(raw_path)          # RGB
            target  = Image.open(raw_semantic_path) # RGB
            vcd.load_from_file( openlabel_path )

            # Data Augmentation and transform to BEV
            image, target = self._data_augmentation(image, target, vcd, camera_name='CAM_FRONT') 
        
        else:
            bev_path, semantic_path = self._get_item_paths(index)

            image   = Image.open(bev_path)      # RGB (1024, 1024, 3)
            target  = Image.open(semantic_path) # RGB (1024, 1024, 3)

            # Normal Data Augmentations
            if self.transforms is not None:
                image, target = self.transforms(image, target)
        
        # cv2.namedWindow("DEBUG_IMAGE", cv2.WINDOW_NORMAL)
        # cv2.imshow("DEBUG_IMAGE", cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        # Merge semantic labels
        if self.merging_lut_ids is not None:
            target = self.merge_semantic_labels(target)

        # Perform data preparation with image_processor 
        # (it shoul be from transformers:SegformerImageProcessor)
        encoded_inputs = self.image_processor(image, target, return_tensors="pt")
        
        # Remove the batch_dim from each sample
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()
        
        return encoded_inputs
    
