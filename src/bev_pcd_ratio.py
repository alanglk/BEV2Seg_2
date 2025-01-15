import cv2
import numpy as np
import matplotlib.pyplot as plt
from vcd import core, types, utils, draw, scl
import os
import json

import argparse

MAX_DISTANCE = 30.0
BEV_HEIGHT = 1024
BEV_WIDTH = 1024

def check_json_structure(json_data:dict):
    root_atribs = ["bev_samples", "pcd_samples"]
    image_token_atribs = ["image_path", "openlabel_path", "camera_name", "samples"]
    sample_atribs = ["A", "B", "dist", "description"]

    for r_atrib in root_atribs:
        if r_atrib == "pcd_samples":
            continue
        if r_atrib not in json_data:
            raise Exception(f"Root json doesnt have '{r_atrib}' entry")
        
        for token, image_data in json_data[r_atrib].items():
            for i_atrib in image_token_atribs:
                if i_atrib not in image_data:
                    raise Exception(f"Image {token} in '{r_atrib}' doesnt have '{i_atrib}' entry")
            
            for i, sample in enumerate(image_data['samples']):
                for s_atrib in sample_atribs:
                    if s_atrib not in sample:
                        raise Exception(f"Sample {i} of image {token} in '{r_atrib}' doesnt have '{s_atrib}' entry")


class BEVPointSelector():
    def __init__(self, data_folder,  image_token, cam_name, output_json_path, image_extension = ".png"):
        self.camera_name = cam_name
        self.image_extension = image_extension
        self.image_token = image_token

        self.image_path = os.path.join(data_folder, "input", "images", f"{self.image_token}_raw{self.image_extension}")  
        self.openlabel_path = os.path.join(data_folder, "input", "openlabel", f"{self.image_token}.json")
        self.output_json_path = output_json_path
        
        # Load json data and check format
        try:
            with open(self.output_json_path, 'r') as f:
                try:
                    self.json_data = json.load(f)
                except:
                    self.json_data = {"bev_samples":{}, "pcd_samples":{}}
                check_json_structure(self.json_data)
        except FileNotFoundError:
            self.json_data = {"bev_samples":{}, "pcd_samples":{}}

        # Load input image
        self.img = cv2.imread(self.image_path)

        # Load openlabel
        self.vcd = core.VCD()
        self.vcd.load_from_file(self.openlabel_path)
        self.scene = scl.Scene(self.vcd)

        # Compute BEV params
        bev_aspect_ratio = BEV_WIDTH / BEV_HEIGHT
        bev_x_range = (0.0, MAX_DISTANCE)
        bev_y_range = (-((bev_x_range[1] - bev_x_range[0]) / bev_aspect_ratio) / 2,
                        ((bev_x_range[1] - bev_x_range[0]) / bev_aspect_ratio) / 2)
        self.bev_parameters = draw.TopView.Params(
                                color_map=utils.COLORMAP_1,
                                topview_size=(BEV_WIDTH, BEV_HEIGHT),
                                background_color=255,
                                range_x=bev_x_range,
                                range_y=bev_y_range,
                                step_x=1.0,
                                step_y=1.0,
                                draw_grid=True
                            )

    def register_sample(self, A, B, dist, description):
        # Reload json data
        try:
            with open(self.output_json_path, 'r') as f:
                try:
                    self.json_data = json.load(f)
                except:
                    self.json_data = {"bev_samples":{}, "pcd_samples":{}}
                check_json_structure(self.json_data)
        except FileNotFoundError:
            self.json_data = {"bev_samples":{}, "pcd_samples":{}}
        
        # Register on json
        if self.image_token not in self.json_data["bev_samples"]:
            data = {
                "image_path": self.image_path,
                "openlabel_path": self.openlabel_path,
                "camera_name": self.camera_name,
                "samples": []
            }
            self.json_data["bev_samples"][self.image_token] = data

        data = {
            "A": [A[0], A[1], 0.0],
            "B": [B[0], B[1], 0.0],
            "dist": dist,
            "description": description
        }
        self.json_data["bev_samples"][self.image_token]["samples"].append(data)
        
        # Dump into file
        with open(self.output_json_path, 'w') as f:
            json.dump(self.json_data, f, indent=4)

    def run(self):
        drawer = draw.TopView(scene=self.scene, coordinate_system="vehicle-iso8855", params=self.bev_parameters)
        
        drawer.add_images(imgs={f"{self.camera_name}": self.img}, frame_num=0)
        drawer.draw_bevs(_frame_num=0)
        drawer.draw_topview_base()
        
        bev_img     = drawer.topView
        buffer_img  = bev_img.copy() 

        picked_pixels = []
        def select_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                picked_pixels.append((x, y))
                print(f"Point selected: ({x}, {y})")
                cv2.circle(buffer_img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                if len(picked_pixels) >= 2:
                    cv2.line(buffer_img, picked_pixels[0], picked_pixels[1], color=(0, 0, 255), thickness=1)
                cv2.imshow("bird's-eye-view", buffer_img)

        cv2.namedWindow("bird's-eye-view", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("bird's-eye-view", select_points)
        
        while True:
            cv2.imshow("bird's-eye-view", buffer_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

            if len(picked_pixels) >= 2:
                # Get metric distance between chosen pixels:
                picked_pix_homog = utils.add_homogeneous_row(np.array(picked_pixels).T)
                picked_3d_coords = utils.inv(drawer.params.S).dot(picked_pix_homog)
                dist = np.linalg.norm(picked_3d_coords[:,0] - picked_3d_coords[:,1])
                print(f"Distance between A-B points: {dist}")

                res = input(f"Use A: {picked_pixels[0]} and B: {picked_pixels[1]}? [y/N]")
                if res == "y":
                    desc = input("Description of the taken sample: ")
                    self.register_sample(picked_pixels[0], picked_pixels[1], dist, description=desc)

                picked_pixels = []
                buffer_img = bev_img.copy()


def main(data_folder,  image_token, cam_name, output_json_path):
    print("Script Configuration:")
    print(f"Data folder: {data_folder}")
    print(f"Image token: {image_token}")
    print(f"Camera name: {cam_name}")
    print(f"Output JSON path: {output_json_path}")
    print()


    bev_ps = BEVPointSelector(data_folder=data_folder, 
                              image_token=image_token, 
                              cam_name=cam_name, 
                              output_json_path=output_json_path)
    # bev_ps.run()
    
    # Compute 2D - 3D ratio
    data = None
    with open(output_json_path, "r") as f:
        data = json.load(f)
    
    len_bevsamples = len(data['bev_samples'][image_token]['samples'])
    len_pcdsamples = len(data['pcd_samples'][image_token]['samples'])
    assert len_bevsamples == len_pcdsamples

    ratio_2d3d = 0.0
    for i in range(len_pcdsamples):
        bev_sample = data['bev_samples'][image_token]['samples'][i]
        pcd_sample = data['pcd_samples'][image_token]['samples'][i]
        
        bev_A, bev_B = bev_sample['A'], bev_sample['B']
        pcd_A, pcd_B = pcd_sample['A'], pcd_sample['B']
        
        ratio_2d3d = bev_sample['dist'] / pcd_sample['dist']
        ratio_3d2d = pcd_sample['dist'] / bev_sample['dist']
        
        print(f"2d: {bev_A}-{bev_B} \t 3d: {pcd_A}-{pcd_B} \t ratio_2d3d: {ratio_2d3d} \t ratio_3d2d: {ratio_3d2d}")



if __name__ == "__main__":
    
    # python3 src/bev_pcd_ratio.py --data_folder ./data --output_json_path ./data/bev_pcd_ratio.json --image_token 60d367ec0c7e445d8f92fbc4a993c67e

    # Definir los argumentos
    parser = argparse.ArgumentParser(description="Calcular distancias métricas en BEV.")
    parser.add_argument('--data_folder', type=str, required=True, help='Path a la carpeta de datos.')
    parser.add_argument('--image_token', type=str, required=True, help='Token de la imagen a procesar.')
    parser.add_argument('--cam_name', type=str, default='CAM_FRONT', help='Nombre de la cámara (por defecto es "CAM_FRONT").')
    parser.add_argument('--output_json_path', type=str, required=True, help='Path del archivo .json de salida.')
    args = parser.parse_args()

    main(args.data_folder, args.image_token, args.cam_name, args.output_json_path)
