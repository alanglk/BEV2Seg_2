from vcd import core, scl, draw, utils, types
### Add this to vcd.scl line 1641 for compatibility with non dynamic camera intrinsics
# intrinsic_types = ["intrinsics_pinhole" "intrinsics_fisheye", "intrinsics_cylindrical", "intrinsics_orthographic", "intrinsics_cubemap"]
# sp = vcd_frame["frame_properties"]["streams"][camera_name]["stream_properties"]
# for it in intrinsic_types:
#     # SO, there are dynamic intrinsics!
#     if it in sp:
#         dynamic_intrinsics = True  
#         break
#
from depth_pro.depth_pro import DepthProConfig
import depth_pro

from oldatasets.NuImages.nulabels import nuid2color, nuid2name, nuid2dynamic
from oldatasets.common.utils import target2image
from my_utils import check_paths, merge_semantic_labels, get_blended_image, DEFAULT_MERGE_DICT
from bev2seg_2 import Raw2Seg_BEV, Raw_BEV2Seg

from bevmap_manager import BEVMapManager
from depth_pcd import DepthEstimation, ScenePCD
from instances import InstanceScenePCD, InstanceBEVMasks, InstanceRAWDrawer, OdometryStitching


from sklearn.cluster import DBSCAN
from PIL import Image
import open3d as o3d
import numpy as np
import torch
import cv2

from scene_renderer import DebugBEVMap
import matplotlib.pyplot as plt

from collections import defaultdict
from typing import List

from tqdm import tqdm
import argparse
import pickle
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
                
    Notation:
        sb: normal -> semantic -> bev
        bs: normal -> bev -> semantic
"""

def annotate_cuboids_on_vcd(instance_pcds:dict, vcd: core.OpenLABEL, frame_num:int, transform_4x4:np.ndarray, cuboid_semantic_labels:List[str] = None, initial_traslation_4x1: np.ndarray = None):
            global uid
            initial_traslation_4x1 = np.zeros((4, 1)) if initial_traslation_4x1 is None else initial_traslation_4x1
            frame_properties = vcd.get_frame(frame_num=frame_num)['frame_properties']

            for semantic_pcd in instance_pcds:
                # skip non dynamic objects
                if not semantic_pcd['dynamic']:
                    continue 
                # skip non selected semantic classes
                if cuboid_semantic_labels is not None and semantic_pcd['label'] not in cuboid_semantic_labels:
                    continue
                cs_src = semantic_pcd['camera_name']
                for bbox in semantic_pcd['instance_3dboxes']:
                    instance_name_id = f"bbox_{bbox['inst_id']}_{semantic_pcd['label_id']}_{frame_num}"
                    uid += 1
                    vcd.add_object(name=instance_name_id, semantic_type=f"annotated/{semantic_pcd['label']}", uid=uid, frame_value=frame_num)
                    
                    # Transform to odom
                    frame_transforms = frame_properties['transforms']
                    frame_odometry = frame_transforms['vehicle-iso8855_to_odom']['odometry_xyzypr']

                    t_vec =  frame_odometry[:3]
                    ypr = frame_odometry[3:]
                    r_3x3 = utils.euler2R(ypr)
                    pose_4x4 = utils.create_pose(R=r_3x3, C=np.array([t_vec]).reshape(3, 1))

                    center_4x1 = utils.add_homogeneous_row(np.array(bbox['center']).reshape(3, -1))
                    center_trans_4x1 = transform_4x4 @ center_4x1
                    center_1x3 = center_trans_4x1[:3].T

                    bbox_cuboid_img = types.cuboid(name="bbox3D",
                                                   val=(center_1x3[0, 0], 
                                                        center_1x3[0, 1], 
                                                        center_1x3[0, 2], 
                                                        ypr[2], ypr[1], ypr[0],
                                                        bbox['dimensions'][2],
                                                        bbox['dimensions'][0],
                                                        bbox['dimensions'][1]
                                                        ),
                                                    coordinate_system="odom")
                    vcd.add_object_data(uid, bbox_cuboid_img, frame_value=frame_num)
            return vcd






def main(scene_path:str, raw2segmodel_path:str, bev2segmodel_path:str, depth_pro_path:str, merge_semantic_labels_flag:bool = True):
    # Paths
    scene_openlabel_path    = os.path.join(scene_path, "original_openlabel.json")
    print(f"scene_openlabel_path: {scene_openlabel_path}")
    check_paths([scene_openlabel_path, raw2segmodel_path, bev2segmodel_path, depth_pro_path])

    # Load OpenLABEL Scene
    vcd = core.OpenLABEL()
    vcd.load_from_file(scene_openlabel_path)
    scene = scl.Scene(vcd=vcd)
    frame_keys = vcd.data['openlabel']['frames'].keys()
    
    # Params
    model_config = {
        'scene':{
            'scene_path':scene_path,
            'camera_name': 'CAM_FRONT'
        },
        'semantic':{
            'raw2segmodel_path': raw2segmodel_path,
            'bev2segmodel_path': bev2segmodel_path,
            'merge_semantic_labels_flag': merge_semantic_labels_flag,
            'merge_dict': DEFAULT_MERGE_DICT
        },
        'depth_estimation':{
            'depth_pro_path': depth_pro_path
        },
        'scene_pcd':{},
        'instance_scene_pcd':{
            'dbscan_samples': 15,
            'dbscan_eps': 0.1,
            'dbscan_jobs': None,
            'lims': (np.inf, np.inf, np.inf),
            'min_samples_per_instance': 250,
            'max_distance': 50.0,
            'max_height': 2.0
        }
    }
    
    camera_name                 = model_config['scene']['camera_name']
    dbscan_samples              = model_config['instance_scene_pcd']['dbscan_samples']
    dbscan_eps                  = model_config['instance_scene_pcd']['dbscan_eps']
    dbscan_jobs                 = model_config['instance_scene_pcd']['dbscan_jobs']
    lims                        = model_config['instance_scene_pcd']['lims']
    min_samples_per_instance    = model_config['instance_scene_pcd']['min_samples_per_instance']
    max_distance                = model_config['instance_scene_pcd']['max_distance']
    max_height                  = model_config['instance_scene_pcd']['max_height']
    
    # Save model_config in metadata:
    vcd.add_metadata_properties({'model_config': model_config})

    # Open-CV windows
    cv2.namedWindow("DEBUG", cv2.WINDOW_NORMAL)
    
    # Create device for model inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create instances
    raw2seg_bev = Raw2Seg_BEV(raw2segmodel_path, None, device=device)
    raw_seg2bev = Raw_BEV2Seg(bev2segmodel_path, None, device=device)
    raw2seg_bev.set_openlabel(vcd)
    raw_seg2bev.set_openlabel(vcd)
    
    BMM = BEVMapManager(scene_path=scene_path, gen_flags={
            'all': False, 
            'pointcloud': False, 
            'instances': False, 
            'occ_bev_mask': False, 
            'tracking': False
        })
    DE  = DepthEstimation(model_path=depth_pro_path, device=device)
    SP  = ScenePCD(scene=scene)
    ISP = InstanceScenePCD(dbscan_samples=dbscan_samples, dbscan_eps=dbscan_eps, dbscan_jobs=dbscan_jobs)
    IBM = InstanceBEVMasks(scene=scene, bev_parameters=raw_seg2bev.bev_parameters)
    IRD = InstanceRAWDrawer(scene=scene)

    # PCD odometry stitching
    ODS = OdometryStitching(
        scene,
        vcd.get_frame(frame_num=0),
        frame_length = len(frame_keys),
        pcd_semantic_labels = ['flat.driveable_surface', 'movable_object.barrier'],
        cuboid_semantic_labels = ['vehicle.car']
    )

    global uid
    uid = len(vcd.get_objects().keys())

    # Scene frame iteration
    for fk in tqdm(frame_keys, desc="frames"):
        frame = vcd.get_frame(frame_num=fk)
        frame_properties    = frame['frame_properties']
        raw_image_path      = frame_properties['streams'][camera_name]['stream_properties']['uri']
        raw_image_path      = os.path.join(scene_path, "nuscenes_sequence", raw_image_path)
        # sample_token        = frame_properties['sample_token']

        # ##############################################################
        # Load input image #############################################
        print(f"# Load input image {'#'*45}")
        raw_image = cv2.imread(raw_image_path)

        # ##############################################################
        # Generate semantic masks ######################################
        print(f"# Generate semantic masks {'#'*38}")
        if BMM.gen_flags['all'] or BMM.gen_flags['semantic']: 
            raw_mask, bev_mask_sb  = raw2seg_bev.generate_bev_segmentation(raw_image,camera_name, frame_num=fk)
            bev_image, bev_mask_bs = raw_seg2bev.generate_bev_segmentation(raw_image, camera_name, frame_num=fk)
            BMM.save_semantic_images(image_name=raw_image_path, images=[raw_mask, bev_mask_sb, bev_mask_bs])
        else:
            bev_image = raw_seg2bev.inverse_perspective_mapping(raw_image, camera_name=camera_name, frame_num=fk)
            raw_mask, bev_mask_sb, bev_mask_bs = BMM.load_semantic_images(image_name=raw_image_path)
        
        # Merge semantic labels
        if merge_semantic_labels_flag:
            raw_mask    = merge_semantic_labels(raw_mask,       raw_seg2bev.label2id, merge_dict=model_config['semantic']['merge_dict'])
            # bev_mask_bs = merge_semantic_labels(bev_mask_bs,    raw_seg2bev.label2id, merge_dict=model_config['semantic']['merge_dict'])
            bev_mask_sb = merge_semantic_labels(bev_mask_sb,    raw_seg2bev.label2id, merge_dict=model_config['semantic']['merge_dict'])

        # ##############################################################
        # Depth estimation #############################################
        print(f"# Depth estimation {'#'*45}")
        if BMM.gen_flags['all'] or BMM.gen_flags['depth']:
            depth_dmap = DE.run(raw_image_path)
            BMM.save_depth_image(raw_image_path, depth_dmap)
        else:
            depth_dmap = BMM.load_depth_image(raw_image_path)

        # ##############################################################
        # Generate pointcloud ##########################################
        print(f"# Generate pointcloud {'#'*42}")
        if BMM.gen_flags['all'] or BMM.gen_flags['pointcloud']:
            blended_image = get_blended_image(raw_image, raw2seg_bev.mask2image(raw_mask))
            pcd = SP.run(depth_dmap, camera_name, color_image=blended_image)
            BMM.save_pointcloud(raw_image_path, pcd)
        else:
            pass
            # pcd = BMM.load_pointcloud(raw_image_path)

        # ##############################################################
        # Generate panoptic pointcloud dict ############################
        print(f"# Generate panoptic pointcloud dict {'#'*28}")
        if BMM.gen_flags['all'] or BMM.gen_flags['instances']:
            instance_pcds = ISP.run(pcd, 
                                    raw_mask, 
                                    camera_name, 
                                    lims=lims,
                                    min_samples_per_instance=min_samples_per_instance,
                                    max_distance=max_distance,
                                    max_height=max_height,
                                    verbose=True)
            BMM.save_instance_pcds(raw_image_path, instance_pcds)
        else:
            instance_pcds = BMM.load_instance_pcds(raw_image_path)
        # save_class_pcds(instance_pcds, fk, semantic_labels=["vehicle.car"]) # Save class pcds
        
        # ##############################################################
        # Draw cuboids and Occupancy/Oclusion on RAW image #############
        # Transform cuboids to BEV, compute ConectedComponents of the bev_mask and
        # calc occupancy/occlusion masks of each instance
        print(f"# Generate Occupancy/Oclusion Instances {'#'*23}")
        bev_image_cuboids = bev_image.copy()
        if BMM.gen_flags['all'] or BMM.gen_flags['occ_bev_mask']:
            instance_pcds, bev_image_occ_ocl = IBM.run(bev_mask_sb, instance_pcds, frame_num=fk, bev_image=bev_image_cuboids)
            BMM.save_occ_bev_masks(raw_image_path, instance_pcds)
        else:
            instance_pcds = BMM.load_occ_bev_masks(raw_image_path)
        # bev_blended = get_blended_image(bev_image_cuboids, raw2seg_bev.mask2image(bev_mask_sb))
        # Uncomment this for visualization
        # instance_pcds, bev_image_occ_ocl = IBM.run(bev_mask_sb, instance_pcds, frame_num=fk, bev_image=bev_image_cuboids)
        
        # ##############################################################
        # Apply tracking data to instances #############################
        print(f"# Apply tracking data to instance dict {'#'*24}")
        if BMM.gen_flags['all'] or BMM.gen_flags['tracking']:
            BMM.save_tracking_frame(frame_num=fk, image_path=raw_image_path, instance_pcds=instance_pcds)
        BMM.load_tracking_frame(frame_num=fk, instance_pcds=instance_pcds)

        # ##############################################################
        # Draw cuboids on RAW image ####################################
        # print(f"# Draw cuboids on RAW image {'#'*36}")
        raw_blended = get_blended_image(raw_image, raw2seg_bev.mask2image(raw_mask))
        raw_image_cuboids = IRD.run_on_image(raw_blended, instance_pcds, frame_num=fk)
        # pcd_semantic, pcd_cuboids = IRD.run_on_pointcloud(instance_pcds)
        bev_repoj_cuboids = raw2seg_bev.inverse_perspective_mapping(raw_image_cuboids, camera_name, fk) # Reproyectar cuboides en raw a bev
        
        # ##############################################################
        # Odometry Stitching ###########################################
        # print(f"# Draw cuboids on RAW image {'#'*36}")
        # accum_pcd = ODS.add_frame_pcd(instance_pcds, camera_name, fk, use_frame_color = False)
        # accum_cuboids = ODS.add_frame_cuboids(instance_pcds, camera_name, fk)
        
        # ##############################################################
        # Visualization ################################################
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # all_geometries = pcd_semantic + pcd_cuboids + [create_plane_at_y(2.0)] + [coordinate_frame]
        # all_geometries = accum_cuboids + [accum_pcd]
        # all_geometries = []
        # o3d.visualization.draw_geometries(all_geometries, window_name="DEBUG") 
        
        # cv2.imshow("DEBUG", bev_image_occ_ocl)
        # cv2.waitKey(0)
        # debug_name = f"{os.path.splitext(os.path.basename(raw_image_path))[0]}.png"
        # cv2.imwrite(os.path.join(scene_path, "debug", "bev_cuboids", f"bev_cuboid_{fk+1}.png"), bev_image_cuboids)
        # cv2.imwrite(os.path.join(scene_path, "debug", "bev_occupancy_oclusion", f"bev_occ_{fk+1}.png"), bev_image_occ_ocl)
        # cv2.imwrite(os.path.join(scene_path, "debug", "raw_cuboids", f"raw_cuboid_{fk+1}.png"), raw_image_cuboids)
        # cv2.imwrite(os.path.join(scene_path, "debug", "bev_reproj_cuboids", f"bev_reproj_cuboid_{fk+1}.png"), bev_repoj_cuboids)
        
        
        # cv2.imwrite(os.path.join(scene_path, "paola", "image_raw_cam_front",    f"{fk+1}.png"), raw_image)
        # cv2.imwrite(os.path.join(scene_path, "paola", "image_bev_cam_front",    f"{fk+1}.png"), bev_image)
        # cv2.imwrite(os.path.join(scene_path, "paola", "semantic_raw_cam_front", f"{fk+1}.png"), raw_mask)
        # cv2.imwrite(os.path.join(scene_path, "paola", "semantic_colored_raw_cam_front", f"{fk+1}.png"),raw2seg_bev.mask2image(raw_mask))
        # cv2.imwrite(os.path.join(scene_path, "paola", "semantic_bev_cam_front", f"{fk+1}.png"), bev_mask_sb)
        # cv2.imwrite(os.path.join(scene_path, "paola", "semantic_colored_bev_cam_front", f"{fk+1}.png"), raw2seg_bev.mask2image(bev_mask_sb) )
        # cv2.imwrite(os.path.join(scene_path, "paola", "cuboids_raw_cam_front",  f"{fk+1}.png"), raw_image_cuboids)
        # cv2.imwrite(os.path.join(scene_path, "paola", "cuboids_bev_cam_front",  f"{fk+1}.png"), bev_repoj_cuboids)
        
        # ##############################################################
        # Annotations ##################################################
        print(f"# Annotating cuboids on vcd {'#'*36}")
        transform_4x4, _ = scene.get_transform(cs_src=camera_name, cs_dst="odom", frame_num=fk)
        annotate_cuboids_on_vcd(instance_pcds, vcd, fk, transform_4x4, cuboid_semantic_labels=['vehicle.car'], initial_traslation_4x1=ODS.initial_translation_4x1)
        vcd.save(os.path.join(scene_path, "nuscenes_sequence", "detections_openlabel.json"))
        
        # print()
        # Check for a key press (if a key is pressed, it returns the ASCII code)
        # if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit
        #     break
        
    # Release resources
    cv2.destroyAllWindows()
    

if __name__ == "__main__":

    # Definir los argumentos
    # parser = argparse.ArgumentParser(description="Scene BEV Map")
    # parser.add_argument('--scene_path', type=str, required=True, help='Path to the scene OpenLABEL file.')
    # parser.add_argument('--tmp_path', type=str, required=False, default=None, help='Path for saving temp files.')
    # args = parser.parse_args()

    scene_path              = "./tmp/my_scene" # args.scene_path
    raw2segmodel_path       = "models/segformer_nu_formatted/raw2segbev_mit-b0_v0.2"
    bev2segmodel_path       = "models/segformer_bev/raw2bevseg_mit-b0_v0.3"
    depth_pro_path          = "./models/ml_depth_pro/depth_pro.pt" 
    tmp_path                = "" # args.tmp_path
    
    main(scene_path=scene_path, raw2segmodel_path=raw2segmodel_path, bev2segmodel_path= bev2segmodel_path, depth_pro_path=depth_pro_path)
