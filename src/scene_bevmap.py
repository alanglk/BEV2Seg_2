from vcd import core, scl, draw, utils
from bev2seg_2 import Raw2Seg_BEV, Raw_BEV2Seg
import numpy as np
import torch

import matplotlib.pyplot as plt

from tqdm import tqdm
import argparse
import cv2
import os


def main(scene_path:str, raw2segmodel_path, bev2segmodel_path, tmp_path:str = None):
    
    # Paths
    scene_openlabel_path = os.path.join(scene_path, "nuscenes_openlabel_complete_sequence.json")
    assert os.path.exists(scene_openlabel_path)

    # Load OpenLABEL Scene
    vcd = core.OpenLABEL()
    vcd.load_from_file(scene_openlabel_path)
    scene = scl.Scene(vcd=vcd)
    camera_name = 'CAM_FRONT'

    cam  = scene.get_camera(camera_name=camera_name)
    print(f"cam {cam}")
    setup_viewer = draw.SetupViewer(scene=scene, 
                                    coordinate_system="odom")
    setup_viewer.plot_setup()
    plt.title('Scene setup')
    plt.show()
    

    # Open-CV windows
    window_raw_name = f"raw_images_{camera_name}"
    window_bev_name = f"bev_images_{camera_name}"
    window_s_name   = f"sedrawmantic_images_{camera_name}"
    window_sb_name  = f"semantic2bev_{camera_name}"
    window_bs_name  = f"bev2semantic_{camera_name}"

    cv2.namedWindow(window_raw_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_bev_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(window_s_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(window_sb_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(window_bs_name, cv2.WINDOW_NORMAL)


    # Load Models for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    raw2seg_bev = Raw2Seg_BEV(raw2segmodel_path, None, device=device)
    raw_seg2bev = Raw_BEV2Seg(bev2segmodel_path, None, device=device)
    raw2seg_bev.set_openlabel(vcd)
    raw_seg2bev.set_openlabel(vcd)

    # Scene frame iteration
    frame_keys = vcd.data['openlabel']['frames'].keys()
    for fk in tqdm(frame_keys, desc="frames"):
        frame = vcd.get_frame(frame_num=fk)
        frame_properties    = frame['frame_properties']
        image_path          = frame_properties['streams'][camera_name]['stream_properties']['uri']
        image_path          = os.path.join(scene_path, image_path)
        sample_token        = frame_properties['sample_token']

        print(f"sample_token: {sample_token}")

        raw_image = cv2.imread(image_path)
        cv2.imshow(window_raw_name, raw_image)


        # Generate semantic masks
        # raw_mask, bev_mask_sb  = raw2seg_bev.generate_bev_segmentation(raw_image,camera_name, frame_num=fk)
        # bev_image, bev_mask_bs = raw_seg2bev.generate_bev_segmentation(raw_image, camera_name, frame_num=fk)
        bev_image = raw2seg_bev.inverse_perspective_mapping(raw_image, camera_name=camera_name, frame_num=0)
        


        setup_viewer = draw.SetupViewer(scene=raw2seg_bev.scene, 
                                        coordinate_system="vehicle-iso8855")
        setup_viewer.plot_setup()
        plt.title('Scene setup')
        plt.show()

        #cv2.imshow(window_s_name,   raw_mask)
        cv2.imshow(window_bev_name, bev_image.astype(np.uint8))
        # cv2.imshow(window_sb_name,  raw2seg_bev.mask2image(bev_mask_sb))
        # cv2.imshow(window_bs_name,  raw_seg2bev.mask2image(bev_mask_bs))


        # Depth estimation

        # Generate semantic pointcloud dict

        # Generate panoptic pointcloud dict



        camera      = scene.get_camera(camera_name=camera_name, frame_num=fk)
        transform   = scene.get_transform(cs_src=camera_name, cs_dst="vehicle-iso8855", frame_num=fk)
        cv2.waitKey(0)
        print()
    

if __name__ == "__main__":

    # Definir los argumentos
    # parser = argparse.ArgumentParser(description="Scene BEV Map")
    # parser.add_argument('--scene_path', type=str, required=True, help='Path to the scene OpenLABEL file.')
    # parser.add_argument('--tmp_path', type=str, required=False, default=None, help='Path for saving temp files.')
    # args = parser.parse_args()

    scene_path          = "./tmp/nuscenes_sequence" # args.scene_path
    raw2segmodel_path   = "models/segformer_nu_formatted/raw2seg_bev_mit-b0_v0.2"
    bev2segmodel_path   = "models/segformer_bev/raw2bevseg_mit-b0_v0.3"
    tmp_path            = "" # args.tmp_path

    main(scene_path=scene_path, raw2segmodel_path=raw2segmodel_path, bev2segmodel_path= bev2segmodel_path, tmp_path=tmp_path)
