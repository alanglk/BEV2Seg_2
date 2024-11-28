import sys

import cv2
import torch
import numpy as np
from typing import Union

def progress_bar(total, current, long=50):
    porcentaje = (current / total) * 100
    bloques = int((current / total) * long)
    barra = f"[{'#' * bloques}{'.' * (50 - bloques)}] {porcentaje:.2f}%"
    
    # Imprime la barra de progreso en la misma línea
    sys.stdout.write(f"\r{barra}")
    sys.stdout.flush()

    if current == total:
        print()
    
def target2image(target: Union[torch.Tensor, np.ndarray], colormap: dict) -> np.ndarray:
    """
     Converts target (seg mask) into BGR image!!!
    INPUT:
        target: torch.tensor (H, W)
        colormap: dict {0: RGB, 1: RGB...}
    OUTPUT:
        np.array BGR Image
    """
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    
    res_mask = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    for label, rgb_col in colormap.items():
        res_mask[target == label] = rgb_col

    # Convert to BGR (for display with opencv)
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_RGB2BGR) 
    return res_mask

def display_image(window_name: str, image:np.ndarray):
    """
    Display one image using OpenCV2.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

def display_images(window_name: str, images:list):
    """
    Displays multiple images using OpenCV2.
    INPUT:
        - window_name: name of the window. Multiple image will set their window names to
            window_name_index
        - images: list of images in numpy BGR format
    """
    for i, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        wn = window_name + f"_{i+1}"
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        cv2.imshow(wn, image)
    cv2.waitKey(0)
