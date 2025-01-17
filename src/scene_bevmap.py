from vcd import core, scl, draw, utils
### Add this to vcd.scl line 1636 for compatibility with non dynamic camera intrinsics
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

from bev2seg_2 import Raw2Seg_BEV, Raw_BEV2Seg
from sklearn.cluster import DBSCAN
from PIL import Image
import open3d as o3d
import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt

from collections import defaultdict
from typing import List

from tqdm import tqdm
import argparse
import pickle
import os


"""
    Scene folder structure:
        nuscenes_sequence/
            - openlabel.json
            - images/ # vcd-uri dependent
                - image1.jpg
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

"""
    instance_pointclouds: 
                [{  'label': str, 
                    'label_id': int,
                    'camera_name': str,
                    'dynamic': bool, 
                    'pcd': np.ndarray, 
                    'pcd_colors': np.ndarray,
                    'instance_pcds': [{
                        'inst_id': int, 
                        'pcd': np.ndarray, 
                        'pcd_colors': np.ndarray
                    }],
                    'instance_3dboxes':[{
                        'inst_id': int, 
                        'center': (x_pos, y_pos, z_pos), 
                        'dimensions': (bbox_width, bbox_height, bbox_depth)  
                    }]
                }]
"""

def check_paths(paths: List[str]) -> List[bool]:
    """
        INPUT: path list of dirs and files
        OUPUT: flag_list with wheter paths exists or not. 
            If a folder path doesn't exist, it is created.
    """
    flag_list = []
    for path in paths:
        if os.path.exists(path):
            flag_list.append(True) # File or Folder exist
            continue
        
        # Set as non existing path and create it if it is a folder path
        flag_list.append(False)
        is_folder_path = not os.path.splitext(path)[1]
        if is_folder_path:  
            os.makedirs(path, exist_ok=True)  # use makedirs for creating intermediate directories
            print(f"Directory '{path}' was created.")
            continue
        
        raise Exception(f"Path '{path}' does not exist and it is not going to be created.")

    return flag_list

def get_blended_image(image_a, image_b:np.ndarray, alpha:float=0.5):
    """
    INPUT: raw_image is image_a and semantic mask is image_b 
    OUPUT: blended image
    """
    if len(image_b.shape) > 2:
        image_b = image_b[:, :, 0]

    image_a_float = image_a.astype(np.float32)
    semantic_mask_rgb = target2image(image_b, nuid2color)
    semantic_mask_rgb_float = semantic_mask_rgb.astype(np.float32)
    blended_image = cv2.addWeighted(image_a_float, 1 - alpha, semantic_mask_rgb_float, alpha, 0)
    return blended_image

class BEVMapManager():
    GEN_FOLDERS = ['semantic', 'depth', 'pointcloud', 'instances']
    
    def __init__(self, 
                 scene_path: str, 
                 gen_flags: dict = {}):
        # check if scene_path exists
        if not os.path.exists(scene_path):
            raise Exception(f"scene_path doesnt exist: {scene_path}")

        # Define Generation paths
        self.gen_folder_path  = os.path.join(scene_path, 'generated')
        self.gen_paths = {self.GEN_FOLDERS[i]: os.path.join(self.gen_folder_path, self.GEN_FOLDERS[i]) for i in range(len(self.GEN_FOLDERS))}
        gen_all_flag = not check_paths([self.gen_folder_path])[0]
        
        # check if the generation paths already exists.
        # if a path doesnt exist, create the folder and mark it to regenerate all that type data
        flag_list   = check_paths([p for _, p in self.gen_paths.items()])
        empty_list  = [ len(os.listdir(p)) == 0 for _, p in self.gen_paths.items()]
        self.gen_flags = {self.GEN_FOLDERS[i]: not flag_list[i] or empty_list[i] for i in range(len(self.GEN_FOLDERS))}
        self.gen_flags['all'] = gen_all_flag
        
        # Check user generation flags
        for flag, val in gen_flags.items():
            if flag not in self.gen_flags:
                raise Exception(f"[BEVMapManager]\t Unknown user flag '{flag}'.")
            if not self.gen_flags[flag]:
                self.gen_flags[flag] = val # Override default gen_flag whit user's one

        # DEBUG INFO
        for flag, val in self.gen_flags.items():
            if val == True:
                print(f"[BEVMapManager]\t {flag} is going to be regenerated.")

    def _get_path(self, image_name:str, gen_type:str, file_extension:str) -> str:
        assert gen_type in self.GEN_FOLDERS
        image_name = os.path.splitext(os.path.basename(image_name))[0]
        return os.path.join(self.gen_paths[gen_type],f"{image_name}{file_extension}")

    def exist_gen_file(self, image_name:str, gen_type, file_extension:str) -> bool:
        assert gen_type in self.GEN_FOLDERS
        image_name = os.path.splitext(os.path.basename(image_name))[0]
        path = os.path.join(self.gen_paths[gen_type],f"{image_name}{file_extension}")

        try:
            return check_paths([path]) # always returning true if no exception
        except:
            return False

    def load_semantic_images(self, image_name: str) -> List[np.ndarray]:
        """
            INPUT: image basename. 
                Ej ->  n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg
            OUTPUT: list [image_semantic, image_sb, image_bs] 
            
            Notation:
                sb: normal -> semantic -> bev 
                bs: normal -> bev -> semantic
        """
        image_paths = [
            self._get_path(image_name, 'semantic', '_semantic.png'),
            self._get_path(image_name, 'semantic', '_sb.png'),
            self._get_path(image_name, 'semantic', '_bs.png') ]
        check_paths(image_paths)
        return [cv2.imread(p) for p in image_paths]

    def load_depth_image(self, image_name:str) -> np.ndarray:
        depth_image_path = self._get_path(image_name, 'depth', '.tiff')
        check_paths([depth_image_path])
        return cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    def load_pointcloud(self, image_name:str):
        pcd_path = self._get_path(image_name, 'pointcloud', '.pcd')
        check_paths([pcd_path])
        pcd = o3d.io.read_point_cloud(pcd_path)
        return pcd

    def load_instance_pcds(self, image_name:str) -> dict:
        panoptic_pcd_path = self._get_path(image_name, 'instances', '.plk')
        check_paths([panoptic_pcd_path])

        with open(panoptic_pcd_path, "rb") as f:
            instance_pcds = pickle.load(f)
        return instance_pcds
    
    def save_semantic_images(self, image_name:str, images:List[np.ndarray]):
        """
        INPUT:
            image_name: path or name of input image
            images: list of [raw semantic image, sb image, bs image]
        """
        image_paths = [
            self._get_path(image_name, 'semantic', '_semantic.png'),
            self._get_path(image_name, 'semantic', '_sb.png'),
            self._get_path(image_name, 'semantic', '_bs.png') ]
        
        for i, p in enumerate(image_paths):
            cv2.imwrite(p, images[i])

    def save_depth_image(self, image_name:str, depth_dmap: np.ndarray):
        depth_image_path = self._get_path(image_name, 'depth', '.tiff')
        depth_image_pil = Image.fromarray(depth_dmap.astype(np.float32), mode='F')
        depth_image_pil.save(depth_image_path)

    def save_pointcloud(self, image_name:str, pcd: o3d.geometry.PointCloud):
        pcd_path = self._get_path(image_name, 'pointcloud', '.pcd')
        o3d.io.write_point_cloud(filename=pcd_path, pointcloud=pcd, write_ascii=True)      

    def save_instance_pcds(self, image_name:str, instance_pcds: dict):
        inst_path = self._get_path(image_name, 'instances', '.plk')
        with open(inst_path, "wb") as f:
            pickle.dump(instance_pcds, f)

class DepthEstimation():
    def __init__(self, model_path:str, device:torch.DeviceObjType):
        self.device = device
        self.model = None
        self.DEFAULT_MONODEPTH_CONFIG_DICT = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=model_path,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )

    def run(self, raw_image_path: str):
        """
        INPUT:
            raw_image: token of RGB input image
        """
        if self.model is None:
            print("[DepthEstimation]\t Loading model...")
            self.model, self.transform = depth_pro.create_model_and_transforms(config=self.DEFAULT_MONODEPTH_CONFIG_DICT, device=self.device)
            self.model.eval()

        raw_image, _, f_px = depth_pro.load_rgb(raw_image_path)

        prediction = self.model.infer(self.transform(raw_image), f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        depth_dmap = depth.cpu().detach().numpy()
        return depth_dmap


class ScenePCD():
    def __init__(self, scene:scl.Scene):
        self.scene = scene
    
    def run(self, depth_map: np.ndarray, camera_name: str, color_image: np.ndarray = None):
        camera = self.scene.get_camera(camera_name)           
        h, w = depth_map.shape

        # Compute image pointcloud 
        colors = np.zeros((h, w, 3), dtype=np.float32)
        aux_xyz_3d_coords = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(0, h):
            # Read all pixels pos of this row
            cam_2d_coords_3xW = np.array(
                [np.linspace(0, w - 1, num=w), i * np.ones(w), np.ones(w)]
            )
            
            cam_2d_ray3d_3xW = camera.reproject_points2d(points2d_3xN=cam_2d_coords_3xW)
            xyz_3d_coords_3xW = cam_2d_ray3d_3xW * depth_map[i, :]
            
            aux_xyz_3d_coords[i, :, 0] = xyz_3d_coords_3xW[0, :]
            aux_xyz_3d_coords[i, :, 1] = xyz_3d_coords_3xW[1, :]
            aux_xyz_3d_coords[i, :, 2] = xyz_3d_coords_3xW[2, :]

            if color_image is not None:
                colors[i, :] = color_image[i, :] / 255.0 # Normalize color 

        # non_zero_depth_mask = aux_xyz_3d_coords[:, :, 2] != 0
        pcd_points = aux_xyz_3d_coords.reshape(-1, 3) # * 100
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        if color_image is not None:
            colors = colors.reshape(-1, 3)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd


class InstanceScenePCD():
    DEFAULT_MERGE_DICT = {
        'vehicle.car': [
            "vehicle.bus.bendy", 
            "vehicle.bus.rigid", 
            "vehicle.car", 
            "vehicle.construction", 
            "vehicle.emergency.ambulance", 
            "vehicle.emergency.police", 
            "vehicle.trailer", 
            "vehicle.truck"
        ],
        "vehicle.motorcycle":[
            "vehicle.bicycle",
            "vehicle.motorcycle"
        ]
    }

    def __init__(self, dbscan_samples:int = 50, dbscan_eps:int = 1, merge_semantic_labels:bool = False, label2id:dict = None):
        """
        INPUT
            - dbscan_samples:
            - dbscan_eps:
            - merge_semantic_labels: if set to true semantic labels are merged with DEFAULT_MERGE_DICT. 
                label2id is required.
        """
        self.dbscan_samples = dbscan_samples
        self.dbscan_eps = dbscan_eps
        
        self.merge_semantic_labels = merge_semantic_labels
        self.label2id = label2id
        if self.merge_semantic_labels:
            assert self.label2id is not None

    def _get_segmented_pcds(self, pcd: np.ndarray, pcd_colors: np.ndarray, seg_mask:np.ndarray, camera_name:str, id2label:dict = nuid2name, id2dynamic: dict = nuid2dynamic):
        # Merge labels
        if self.merge_semantic_labels is not None:
            for k, vals in self.DEFAULT_MERGE_DICT.items():
                to_id   = self.label2id[k]
                for v in vals:
                    from_id = self.label2id[v]
                    seg_mask[seg_mask == from_id] = to_id
        
        pcds = []
        labels = np.unique(seg_mask)
        # Compute segmented pcd
        for l in labels:
            mask = (seg_mask == l)
            points = pcd[mask]
            colors = pcd_colors[mask]
            
            data = {"pcd": points, "pcd_colors": colors, "label_id": l, "camera_name": camera_name}
            if id2label is not None and l in id2label:
                data['label'] = id2label[l]        
            if id2dynamic is not None and l in id2dynamic:
                data['dynamic'] = id2dynamic[l]
            pcds.append(data)

        return pcds
    
    def _get_cluster_colormap(self, labels):
        pallete = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
        color_map = {}
        for l in labels:
            if l == -1:
                color_map[l] = (0.5, 0.5, 0.5)  # grey color for noise points
            elif l < len(pallete):
                color_map[l] = pallete[l]
            else:
                color_map[l] = np.random.random(3)
        return color_map
        
    def run(self, pcd: o3d.geometry.PointCloud, semantic_mask:np.ndarray, camera_name:str, lims: tuple = (10, 5, 30)):
        """
        INPUT
            - pcd: open-3d pointcloud
            - semantic_mask: (H, W) semamtic mask
        OUTPUT:
            instance_pointclouds: 
                [{  'label': str, 
                    'label_id': int,
                    'camera_name': str,
                    'dynamic': bool, 
                    'pcd': np.ndarray, 
                    'pcd_colors': np.ndarray,
                    'instance_pcds': [{
                        'inst_id': int, 
                        'pcd': np.ndarray, 
                        'pcd_colors': np.ndarray
                    }],
                    'instance_3dboxes':[{
                        'inst_id': int, 
                        'center': (x_pos, y_pos, z_pos), 
                        'dimensions': (bbox_width, bbox_height, bbox_depth)  
                    }]
                }]
        """
        pcd_points      = np.asarray(pcd.points)
        semantic_mask   = semantic_mask[:, :, 0] if len(semantic_mask.shape) > 2 else semantic_mask

        # Filter pointcloud and segmented_mask
        mask = (abs(pcd_points[:, 0]) <= lims[0]) & (abs(pcd_points[:, 1]) <= lims[1]) & (abs(pcd_points[:, 2]) <= lims[2])
        semantic_mask   = semantic_mask.flatten()[mask]
        pcd_points      = np.asarray(pcd.points)[mask]
        pcd_colors      = np.asarray(pcd.colors)[mask]
        segmented_pointclouds = self._get_segmented_pcds(pcd_points, pcd_colors, semantic_mask, camera_name)


        # Compute instances for each segmented pcd class
        for seg_pcd in segmented_pointclouds:
            if not seg_pcd['dynamic']:
                continue
            
            # Compute Clusters
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_samples).fit(seg_pcd['pcd'])
            color_map = self._get_cluster_colormap(db.labels_)
            clusters = defaultdict(list)
            for label, point_xyz in zip(db.labels_, seg_pcd['pcd']):
                clusters[label].append(point_xyz)

            # Add cluster pointclouds and calc instance bbox
            seg_pcd['instance_pcds'] = []
            seg_pcd['instance_3dboxes'] = []
            for inst_id, pcd in clusters.items():
                inst_pcd = np.array(pcd)
                data = {'inst_id': inst_id, 
                        'pcd': inst_pcd, 
                        'pcd_colors': np.array([color_map[inst_id]] * len(pcd)).reshape(-1, 3)
                        }
                seg_pcd['instance_pcds'].append(data)

                # skip (this points are noise or background)
                if inst_id == -1:
                    continue 

                # Compute 3D cuboids
                x_pos = (np.max(inst_pcd[:, 0]) + np.min(inst_pcd[:, 0])) / 2
                y_pos = (np.max(inst_pcd[:, 1]) + np.min(inst_pcd[:, 1])) / 2
                z_pos = (np.max(inst_pcd[:, 2]) + np.min(inst_pcd[:, 2])) / 2
                bbox_width = np.abs(np.max(inst_pcd[:, 0]) - np.min(inst_pcd[:, 0]))
                bbox_height = np.abs(np.max(inst_pcd[:, 1]) - np.min(inst_pcd[:, 1]))
                bbox_depth = np.abs(np.max(inst_pcd[:, 2]) - np.min(inst_pcd[:, 2]))
                
                data = {'inst_id': inst_id, 
                        'center': (x_pos, y_pos, z_pos), 
                        'dimensions': (bbox_width, bbox_height, bbox_depth)
                        }
                seg_pcd['instance_3dboxes'].append(data)
        return segmented_pointclouds

class InstanceBEVDrawer():
    DEFAULT_DRAWING_LABELS = ['vehicle.car']

    def __init__(self, scene:scl.Scene, bev_parameters:draw.TopView.Params, drawing_semantic_labels:List[str]=None):
        self.scene = scene
        self.bev_parameters = bev_parameters
        self.drawing_semantic_labels = drawing_semantic_labels if drawing_semantic_labels is not None else self.DEFAULT_DRAWING_LABELS  
    
    def point2pixel(self, point: tuple[int, int]) -> tuple[int, int]:
        pixel = (
            int(round(point[0] * self.bev_parameters.scale_x + self.bev_parameters.offset_x)),
            int(round(point[1] * self.bev_parameters.scale_y + self.bev_parameters.offset_y)),
        )
        return pixel

    def run(self, bev_image:np.ndarray, instance_pcds:dict, frame_num:int):
        """
        INPUT:
            instance_pcds: 
                [{  'label': str, 
                    'label_id': int,
                    'camera_name': str,
                    'dynamic': bool, 
                    'pcd': np.ndarray, 
                    'pcd_colors': np.ndarray,
                    'instance_pcds': [{
                        'inst_id': int, 
                        'pcd': np.ndarray, 
                        'pcd_colors': np.ndarray
                    }],
                    'instance_3dboxes':[{
                        'inst_id': int, 
                        'center': (x_pos, y_pos, z_pos), 
                        'dimensions': (bbox_width, bbox_height, bbox_depth)  
                    }]
                }]
        """
        
        # Draw cuboids on BEV image
        for semantic_pcd in instance_pcds:
            # skip non dynamic objects
            if not semantic_pcd['dynamic']:
                continue 
            # skip non selected semantic classes
            if semantic_pcd['label'] not in self.drawing_semantic_labels:
                continue
            
            # Compute pseudo-pcd to bev_image ratio
            camera_name = semantic_pcd['camera_name']
            pcd         = semantic_pcd['pcd']
            ratio_3d2d = 2.1841948444444443
            ratio_2d3d = 1 # 0.4578346123943684
            print(f"ratio_2d3d: {ratio_2d3d} | ratio_3d2d: {ratio_3d2d}")

            # Draw instance 3d cuboids on top-view
            for inst_3dbox in semantic_pcd['instance_3dboxes']:
                if inst_3dbox['inst_id'] == -1:
                    continue # skip if there is an unlabeled 3dbox

                # Transform cuboid center to vehicle frame
                center_3x1 = np.array([inst_3dbox['center'] ]).T * ratio_2d3d
                center_4x1 = utils.add_homogeneous_row(center_3x1)
                center_transformed_4x1 = self.scene.transform_points3d_4xN(center_4x1, camera_name, "vehicle-iso8855", frame_num=frame_num) # + T

                center_bev_3x1 =  np.array([center_transformed_4x1[0, 0], center_transformed_4x1[1, 0], 1.0])
                center_bev_3x1 = self.bev_parameters.S.dot(center_bev_3x1.T).T
                center_bev_pixel = ( int(round(center_bev_3x1[0])), int(round(center_bev_3x1[1])))

                # Calc cuboid base vertices bev_image = on vehicle frame
                cx, cy, cz, _ = center_transformed_4x1[:, 0]
                w, h, d = inst_3dbox['dimensions']
                w *= ratio_2d3d
                h *= ratio_2d3d
                d *= ratio_2d3d
                vertices_Nx3 = np.array([
                    [cx - d/2, cy - w/2, cz - h/2],  # Vértice 0
                    [cx + d/2, cy - w/2, cz - h/2],  # Vértice 1
                    [cx + d/2, cy + w/2, cz - h/2],  # Vértice 2
                    [cx - d/2, cy + w/2, cz - h/2],  # Vértice 3
                ])

                # Project points to top-view pixels
                center_pixel = self.point2pixel((center_transformed_4x1[0, 0], center_transformed_4x1[1, 0]))
                cuboid_pixels = []
                for vert_3x1 in vertices_Nx3:
                    vx, vy, vz = vert_3x1[0], vert_3x1[1], vert_3x1[2]
                    cuboid_pixels.append(self.point2pixel((vx, vy)))
                
                # Draw cuboid base and center on top-view image
                thick = 2
                color = (0, 255, 0)

                cv2.circle(bev_image, center_pixel, 2, color, thick)
                cv2.circle(bev_image, center_bev_pixel, 1, (0, 255,255), thick)
                cv2.circle(bev_image, self.point2pixel((13, 11)), 1, (255, 255,255), thick)

                cv2.circle(bev_image, cuboid_pixels[0], 1, color, thick)
                cv2.circle(bev_image, cuboid_pixels[1], 1, color, thick)
                cv2.circle(bev_image, cuboid_pixels[2], 1, color, thick)
                cv2.circle(bev_image, cuboid_pixels[3], 1, color, thick)

                cv2.line(bev_image, cuboid_pixels[0], cuboid_pixels[1], color, 1)
                cv2.line(bev_image, cuboid_pixels[1], cuboid_pixels[2], color, 1)
                cv2.line(bev_image, cuboid_pixels[2], cuboid_pixels[3], color, 1)
                cv2.line(bev_image, cuboid_pixels[3], cuboid_pixels[0], color, 1)
                print()

        return bev_image


def main(scene_path:str, raw2segmodel_path, bev2segmodel_path, depth_pro_path):
    # Paths
    scene_openlabel_path = os.path.join(scene_path, "nuscenes_openlabel_complete_sequence.json")
    print(f"scene_openlabel_path: {scene_openlabel_path}")
    check_paths([scene_openlabel_path,])

    # Load OpenLABEL Scene
    vcd = core.OpenLABEL()
    vcd.load_from_file(scene_openlabel_path)
    scene = scl.Scene(vcd=vcd)
    camera_name = 'CAM_FRONT'
    

    # Open-CV windows
    cv2.namedWindow("DEBUG", cv2.WINDOW_NORMAL)


    # Create device for model inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create instances
    BMM = BEVMapManager(scene_path=scene_path, gen_flags={'all': True, 'pointcloud': False, 'instances': False})
    
    raw2seg_bev = Raw2Seg_BEV(raw2segmodel_path, None, device=device)
    raw_seg2bev = Raw_BEV2Seg(bev2segmodel_path, None, device=device)
    raw2seg_bev.set_openlabel(vcd)
    raw_seg2bev.set_openlabel(vcd)
    
    DE  = DepthEstimation(model_path=depth_pro_path, device=device)
    SP  = ScenePCD(scene=scene)
    ISP = InstanceScenePCD(model_debug=raw_seg2bev, label2id=raw_seg2bev.label2id)
    IBD = InstanceBEVDrawer(scene=scene, bev_parameters=raw_seg2bev.bev_parameters)

    # Scene frame iteration
    frame_keys = vcd.data['openlabel']['frames'].keys()
    for fk in tqdm(frame_keys, desc="frames"):
        frame = vcd.get_frame(frame_num=fk)
        frame_properties    = frame['frame_properties']
        raw_image_path      = frame_properties['streams'][camera_name]['stream_properties']['uri']
        raw_image_path      = os.path.join(scene_path, raw_image_path)
        # sample_token        = frame_properties['sample_token']

        # ##############################################################
        # Load input image #############################################
        raw_image = cv2.imread(raw_image_path)


        # ##############################################################
        # Generate semantic masks ######################################
        if BMM.gen_flags['all'] or BMM.gen_flags['semantic']: 
            raw_mask, bev_mask_sb  = raw2seg_bev.generate_bev_segmentation(raw_image,camera_name, frame_num=fk)
            bev_image, bev_mask_bs = raw_seg2bev.generate_bev_segmentation(raw_image, camera_name, frame_num=fk)
            # BMM.save_semantic_images(image_name=raw_image_path, images=[raw_mask, bev_mask_sb, bev_mask_bs])
        else:
            bev_image = raw_seg2bev.inverse_perspective_mapping(raw_image, camera_name=camera_name, frame_num=fk)
            raw_mask, bev_mask_sb, bev_mask_bs = BMM.load_semantic_images(image_name=raw_image_path)
        
        # cv2.imshow(window_bev_name, bev_image)
        # cv2.imshow(window_s_name, raw2seg_bev.mask2image(raw_mask))
        # cv2.imshow(window_sb_name,  raw2seg_bev.mask2image(bev_mask_sb))
        # cv2.imshow(window_bs_name,  raw_seg2bev.mask2image(bev_mask_bs))
        
        print(f"raw_mask.shape: {raw_mask.shape}")
        # ##############################################################
        # Depth estimation #############################################
        if BMM.gen_flags['all'] or BMM.gen_flags['depth']:
            depth_dmap = DE.run(raw_image_path)
            # BMM.save_depth_image(raw_image_path, depth_dmap)
        else:
            depth_dmap = BMM.load_depth_image(raw_image_path)

        # ##############################################################
        # Generate pointcloud  #########################################
        if BMM.gen_flags['all'] or BMM.gen_flags['pointcloud']:
            blended_image = get_blended_image(raw_image, raw_mask)
            pcd = SP.run(depth_dmap, camera_name, color_image=blended_image)
            # BMM.save_pointcloud(raw_image_path, pcd)
        else:
            pcd = BMM.load_pointcloud(raw_image_path)

        # ##############################################################
        # Generate panoptic pointcloud dict ############################
        if BMM.gen_flags['all'] or BMM.gen_flags['instances']:
            instance_pcds = ISP.run(pcd, raw_mask, camera_name, lims=(np.inf, np.inf, np.inf))
            BMM.save_instance_pcds(raw_image_path, instance_pcds)
        else:
            instance_pcds = BMM.load_instance_pcds(raw_image_path)

        # ##############################################################
        # Draw cuboids on BEV image ####################################
        bev_image_cuboids = IBD.run(bev_image, instance_pcds, frame_num=fk)
        cv2.imshow(window_bev_cuboid_name, bev_image_cuboids)


        # Check for a key press (if a key is pressed, it returns the ASCII code)
        cv2.waitKey(0)
        print()
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

    scene_path          = "./tmp/nuscenes_sequence" # args.scene_path
    raw2segmodel_path   = "models/segformer_nu_formatted/raw2seg_bev_mit-b0_v0.2"
    bev2segmodel_path   = "models/segformer_bev/raw2bevseg_mit-b0_v0.3"
    depth_pro_path      = "./models/ml_depth_pro/depth_pro.pt" 
    tmp_path            = "" # args.tmp_path
    

    main(scene_path=scene_path, raw2segmodel_path=raw2segmodel_path, bev2segmodel_path= bev2segmodel_path, depth_pro_path=depth_pro_path)
