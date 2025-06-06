import cv2
import numpy as np
from vcd import core, types, utils, draw, scl
from PIL import Image
from typing import Union, List

import os 
import sys

class Dataset2BEV():
    def __init__(self, cam_name: str, scene:scl.Scene, bev_max_distance = 30.0, bev_width = 1024, bev_heigh = 1024):
        self.camera_name = cam_name
        self.scene = scene

        bev_aspect_ratio = bev_width / bev_heigh
        self.bev_x_range = (-1.0, bev_max_distance)
        #self.bev_x_range = (- bev_max_distance, bev_max_distance)
        #self.bev_x_range = (- 5.0, 10.0)
        self.bev_y_range = (-((self.bev_x_range[1] - self.bev_x_range[0]) / bev_aspect_ratio) / 2,
                        ((self.bev_x_range[1] - self.bev_x_range[0]) / bev_aspect_ratio) / 2)
        self.bev_parameters = draw.TopView.Params(
                                color_map=utils.COLORMAP_1,
                                topview_size=(bev_width, bev_heigh),
                                background_color=0,
                                range_x=self.bev_x_range,
                                range_y=self.bev_y_range,
                                step_x=1.0,
                                step_y=1.0,
                                draw_grid=True
                            )
        
        self.drawer = draw.TopView(scene=self.scene, 
                              coordinate_system="vehicle-iso8855", 
                              params=self.bev_parameters)

    def _img2bev(self, image: np.ndarray, framenum: int = 0):
        # print(f"DEBUG: image 2 bec image shape {image.shape}")
        cam = self.scene.get_camera(camera_name=self.camera_name, frame_num=framenum)
        # print(f"DEBUG: camera distorsion: {cam.d_1xN}")
        
        sys.stdout = open(os.devnull, 'w') # Redirigir stdout a os.devnull para ignorar la salida
        self.drawer.add_images(imgs={f"{self.camera_name}": image}, 
                          frame_num=framenum)
        sys.stdout = sys.__stdout__ # Restablecer la salida estándar para que vuelva a imprimir en pantalla
        
        self.drawer.draw_bevs(_frame_num=framenum)
        return self.drawer.topView

    def _target2bev(self, target: np.ndarray) -> np.ndarray:
        """
            target (seg mask) (H, W)
        """
        target = np.expand_dims(target, 0).repeat(3, axis=0)
        target = np.moveaxis(target, 0, -1)
        # target (H, W, 3)
        
        map_x = self.drawer.images[self.camera_name]["mapX"]
        map_y = self.drawer.images[self.camera_name]["mapY"]
        bev = cv2.remap(target, map_x, map_y,
            interpolation=cv2.INTER_NEAREST, # INTER_LINEAR
            borderMode=cv2.BORDER_CONSTANT,
        )

        bev32 = np.array(bev, np.float32)
        if "weights" in self.drawer.images[self.camera_name]:
            cv2.multiply(self.drawer.images[self.camera_name]["weights"], bev32, bev32)

        target = bev32.astype(np.uint8)
        return target[:, :, 0] # return (H, W)
    
    def convert2bev(self, image, target):
        image_bev = self._img2bev(image)
        target_bev = self._target2bev(target)
        return image_bev, target_bev
