from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
import torch

import cv2
import numpy as np
from vcd import core, utils, draw, scl


from typing import Union
from abc import ABC, abstractmethod

import sys
import os

# TODO: images from multiple types not only Pillow
# ImageInput = Union[
#     "PIL.Image.Image", np.ndarray, "torch.Tensor", List["PIL.Image.Image"], List[np.ndarray], List["torch.Tensor"]
# ]  # noqa

# TODO: possibility to pass a batch as input
# TODO: check the types

class BEV2SEG_2_Interface(ABC):
    BEV_MAX_DISTANCE = 30
    BEV_WIDTH = 1024
    BEV_HEIGH = 1024

    def __init__(self, model_path:str, openlabel_path: str):
        # SegFormer Model
        self.image_processor = SegformerImageProcessor(reduce_labels=True)
        self.model_path = model_path
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_path)
        

        # BEV Parameters
        bev_aspect_ratio = self.BEV_WIDTH / self.BEV_HEIGH
        bev_x_range = (-1.0, self.BEV_MAX_DISTANCE)
        bev_y_range = (-((bev_x_range[1] - bev_x_range[0]) / bev_aspect_ratio) / 2,
                        ((bev_x_range[1] - bev_x_range[0]) / bev_aspect_ratio) / 2)
        self.bev_parameters = draw.TopView.Params(
            color_map           =   utils.COLORMAP_1, # In the case the OpenLABEL has defined objects
            topview_size        =   (self.BEV_WIDTH, self.BEV_HEIGH),
            background_color    =   0,
            range_x             =   bev_x_range,
            range_y             =   bev_y_range,
            step_x              =   1.0,
            step_y              =   1.0,
            draw_grid           =   True
        )

        # Set the openlabel and create the BEV Drawer
        if openlabel_path is not None:
            vcd = core.VCD()
            vcd.load_from_file(openlabel_path)
            self.set_openlabel(vcd)

    def set_openlabel(self, openlabel: Union[core.OpenLABEL, core.VCD]):
        """Set the camera parameters for the IPM"""
        self.vcd = openlabel
        self.scene = scl.Scene(openlabel)
        self.drawer = draw.TopView(scene=self.scene, coordinate_system="vehicle-iso8855", params=self.bev_parameters)

    def inverse_perspective_mapping(self, image: np.ndarray, camera_name: str, frame_num: int = 0) -> np.ndarray:
        """IPM to BEV perspective
        INPUT: 
        """        
        sys.stdout = open(os.devnull, 'w') # Redirigir stdout a os.devnull para ignorar la salida
        self.drawer.add_images(imgs = {f"{camera_name}": image}, frame_num = frame_num)
        sys.stdout = sys.__stdout__ # Restablecer la salida estándar para que vuelva a imprimir en pantalla
        
        self.drawer.draw_bevs(_frame_num=frame_num)
        return self.drawer.topView

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Image preprocess for inference"""
        
        encoded_inputs = self.image_processor(image, return_tensors="pt")
        # Remove the batch_dim from each sample
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()

        # TODO: add support for batches. If it is just one image it is neccesary to create a minibatch
        # Currently only one image
        encoded_inputs['pixel_values'] = encoded_inputs['pixel_values'].unsqueeze(0) # Add batch dimension

        return encoded_inputs
    
    def mask2image(self, mask: Union[torch.Tensor, np.ndarray], colormap: dict) -> np.ndarray:
        """
        Converts seg mask into BGR image!!!
        INPUT:
            mask: torch.tensor (H, W)
            colormap: dict {0: RGB, 1: RGB...}
        OUTPUT:
            np.array BGR Image
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        res_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label, rgb_col in colormap.items():
            res_mask[mask == label] = rgb_col

        # Convert to BGR (for display with opencv)
        res_mask = cv2.cvtColor(res_mask, cv2.COLOR_RGB2BGR)
        return res_mask

    @abstractmethod
    def generate_bev_segmentation(self, image: np.ndarray, camera_name:str, openlabel: core.OpenLABEL) -> None:
        """
        INPUT: Raw Image and corresponding OpenLABEL with intrinsic and extrinsic
        parameters.
        OUTPUT: Bird's Eye View Semantic Segmentation
        """
        raise NotImplementedError

# class Raw2Seg_BEV(BEV2SEG_2_Interface):
# 
#     def __init__(self, model_path: str, openlabel_path: str):
#         """
#         Semantic Segmentation on Raw image and then IPM to Bird's Eye View.
#         """
#         super().__init__(model_path, openlabel_path)
#     
#     def generate_bev_segmentation(self, image: np.ndarray, camera_name:str, openlabel: core.OpenLABEL = None):
#         
#         # Inference
#         encoded = super().preprocess_image(image)
#         with torch.no_grad():
#             outputs = self.model( encoded['pixel_values'] )
#         raw_mask = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
#         raw_mask = raw_mask.cpu().numpy() # expected (batch_dim, height, width); received (H, W)
#         print(raw_mask.shape)
# 
#         # Generate BEV mask
#         # TODO: complete this
#         raw_mask = np.expand_dims(raw_mask, 0).repeat(3, axis=0)
#         raw_mask = np.moveaxis(raw_mask, 0, -1) # expected (batch_dim, H, W, 3)
#         
#         print(raw_mask.shape)
#         bev_mask = self.inverse_perspective_mapping(raw_mask, camera_name)
#         return bev_mask
# 
# 


class Raw_BEV2Seg(BEV2SEG_2_Interface):

    def __init__(self, model_path: str, openlabel_path: str):
        """
        IPM to Bird's Eye View and then Semantic Segmentation on the BEV Space
        """
        super().__init__(model_path, openlabel_path)
    
    def generate_bev_segmentation(self, image: np.ndarray, camera_name:str, openlabel: core.OpenLABEL = None):
        # IPM to BEV image
        bev_image = self.inverse_perspective_mapping(image, camera_name)
        # cv2.imshow("BEV image", bev_image)
        # cv2.waitKey(0)

        # Inference
        encoded = super().preprocess_image(bev_image)
        with torch.no_grad():
            outputs = self.model( encoded['pixel_values'] )
        bev_mask = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[bev_image.shape[:2]])[0]
        bev_mask = bev_mask.cpu().numpy()

        # cv2.imshow("Segmentation mask", self.mask2image(bev_mask, nuid2color))
        # cv2.waitKey(0)
        return bev_mask



class BEV2SEG_2():
    def __init__(self, 
                 raw2seg_model_path: str, 
                 bev2seg_model_path: str):
        pass
    
    def _fusion():
        pass

    def generate_bev_segmentation(self, image: np.ndarray, openlabel: core.OpenLABEL):
        pass

