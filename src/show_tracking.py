import numpy as np
from vcd import utils

import matplotlib.pyplot as plt
import cv2

from my_utils import check_paths
from bev2seg_2 import Raw2Seg_BEV

import argparse
import os

"""
    Scene folder structure:
        - scene_path/
            - original_openlabel.json
            - nuscenes_sequence/
                - annotated_openlabel.json
                - structure.json
                - images/ # vcd-uri dependent
                    - image1.jpg

            - debug/
                - bev_cuboids/
                - bev_reproj_cuboids/
                - raw_cuboids/
                - vehicle_pcd/
            - generated/
                - semantic/
                    - image1_semantic.png
                    - image1_bev_sb.png
                    - image1_bev_bs.png
                - depth/
                    - image1.tiff
                - pointloud/
                    - image1.pcd
                - instances/
                    - image1.plk
                - tracking/
                    - frame_0.txt
                
    Notation:
        sb: normal -> semantic -> bev
        bs: normal -> bev -> semantic
"""

"""
Frame_num: fk
Image_path: <image-path>
Detections --------------------
|  detection (x, y, z)  | semantic_label |object_id        |
|-----------------------|----------------|-----------------|
| x y z                 | pedestrian     | 0_pedestrian_0  |
| x y z                 | vehicle.car    | 0_vehicle.car_0 |
| x y z                 | vehicle.car    | 0_vehicle.car_1 |
Tracks --------------------
|  prediction (x, y, dx, dy) | frame_start | frame_end | tracking_id | semantic_label | associated detections  |
|----------------------------|-------------|-----------|-------------|----------------|------------------------|
| x y dx dy                  | 0           | 0         | -1          | pedestrian     | [ 0_pedestrian_0,...]  |
| x y dx dy                  | 0           | 0         | -2          | vehicle.car    | [ 0_vehicle.car_0,...] |
| x y dx dy                  | 0           | 0         | -3          | vehicle.car    | [ 0_vehicle.car_1,...] |
"""


class Detection:
    def __init__(self, x, y, z, semantic_label, object_id):
        self.x = x
        self.y = y
        self.z = z
        self.semantic_label = semantic_label
        self.object_id = object_id
    def __str__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z} semantic_label: {self.semantic_label} index_pos: {self.index_pos}\n"
    def __repr__(self):
        return self.__str__()
class Track:
    def __init__(self, x, y, dx, dy, frame_start, frame_end, tracking_id, semantic_label, object_id):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.tracking_id = tracking_id
        self.semantic_label = semantic_label
        self.object_id = object_id

from typing import List, Tuple

def load_tracking_data(file_path) -> Tuple[int, List[Detection], List[Track]]:
    detections = []
    tracks = []
    frame_num = -1
    image_path = None
    parsing_detections = False
    parsing_tracks = False
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith("Frame_num:"):
            frame_num = int(line.split()[1])
            continue
        if line.startswith("Image_path:"):
            image_path = line.split()[1]
            continue
        if line.startswith("Detections"):
            parsing_detections = True
            parsing_tracks = False
            continue
        if line.startswith("Tracks"):
            parsing_detections = False
            parsing_tracks = True
            continue
        if line.startswith("|") or line == "":
            continue  # Skip table headers and empty lines
        
        parts = line.split()
        if parsing_detections:
            x, y, z = map(float, parts[:3])
            semantic_label = parts[3]
            object_id = parts[4]
            detections.append(Detection(x, y, z, semantic_label, object_id))
        elif parsing_tracks:
            x, y, dx, dy = map(float, parts[:4])
            frame_start = int(parts[4])
            frame_end = int(parts[5])
            tracking_id = int(parts[6])
            semantic_label = parts[7]
            object_id = parts[8]
            tracks.append(Track(x, y, dx, dy, frame_start, frame_end, tracking_id, semantic_label, object_id))
    return frame_num, image_path, detections, tracks


def main(scene_folder_path:str):
    tracking_folder = os.path.join(scene_folder_path, "generated", "tracking")
    check_paths([scene_folder_path, tracking_folder])
    
    tracking_paths = os.listdir(tracking_folder)
    
    frame_data = {}
    for file_name in tracking_paths:
        file_path = os.path.join(tracking_folder, file_name)
        frame_num, image_path, detections, tracks = load_tracking_data(file_path)
        if frame_num not in frame_data:
            frame_data[frame_num] = {}
        frame_data[frame_num]["image_path"] = image_path
        frame_data[frame_num]["detections"] = detections
        frame_data[frame_num]["tracks"] = tracks
    

    # Visualization
    scene_openlabel_path    = os.path.join(scene_folder_path, "original_openlabel.json")
    raw2seg_bev = Raw2Seg_BEV("models/segformer_nu_formatted/raw2segbev_mit-b0_v0.2", scene_openlabel_path, device=None)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.ion()
    plt.show()

    frame_keys = list(frame_data.keys())
    frame_keys.sort()
    for fk in frame_keys:
        detections  = frame_data[fk]["detections"]
        tracks      = frame_data[fk]["tracks"]
        ax.cla() # Clear the current axes
        
        image_path = frame_data[fk]["image_path"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        

        ax.imshow(image)
        
        for track in tracks:
            track.frame_start
            
            aux = "fk_label_index"
            
            trk_st_frame, trk_label, trk_st_index = track.object_id.split('_')


        points3d_Nx3 = np.array( [ (det.x, det.y, det.z)  for det in detections ] )
        camera = raw2seg_bev.scene.get_camera('CAM_FRONT', frame_num=fk)
        points3d_4xN = utils.add_homogeneous_row(points3d_Nx3.T)
        points3d_4xN, remove_outside = camera.project_points3d(points3d_4xN)
        ax.scatter(points3d_4xN[0, :], points3d_4xN[1, :])

        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show tracking")
    parser.add_argument('scene_folder', type=str, help="Scene folder path")
    args = parser.parse_args()
    
    main(args.scene_folder)