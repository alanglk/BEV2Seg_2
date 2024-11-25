import cv2
import numpy as np


def display_test_image(window_name: str, image:np.ndarray):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

def display_test_images(window_name: str, images:list):
    for i, image in enumerate(images):
        wn = window_name + f"_{i+1}"
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        cv2.imshow(wn, image)
    cv2.waitKey(0)
