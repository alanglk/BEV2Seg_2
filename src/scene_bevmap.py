from vcd import core, scl, draw, utils
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

def get_blended_image(image_a:np.ndarray, image_b:np.ndarray, alpha:float=0.5):
    """
    INPUT: raw_image is image_a and semantic mask colored is image_b 
    OUPUT: blended image
    """
    if image_a.shape != image_b.shape:
        raise ValueError("Las imágenes deben tener las mismas dimensiones y número de canales.")
    
    blended_image = cv2.addWeighted(image_a, alpha, image_b, 1 - alpha, 0)
    return blended_image

def get_pcds_of_semantic_label(instance_pcds:dict, semantic_labels:list = None):
    pcds = []
    for semantic_pcd in instance_pcds:
        # skip if semantic_label is set and is not equal to the object label
        if semantic_labels is not None and semantic_pcd['label'] not in semantic_labels:
            continue 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(semantic_pcd['pcd'])
        pcd.colors = o3d.utility.Vector3dVector(semantic_pcd['pcd_colors'])
        pcds.append(pcd)
    return pcds

def save_class_pcds(instance_pcds:dict, frame_num:int, semantic_labels = ["vehicle.car"]):
            pcds = get_pcds_of_semantic_label(instance_pcds, semantic_labels=["vehicle.car"])
            vehicle_pcd = np.asarray(pcds.pop(0).points)
            vehicle_pcd_path = os.path.join(scene_path, "debug", "vehicle_pcd", f"pointcloud_{frame_num+1}.png")
            o3d.io.write_point_cloud(filename=vehicle_pcd_path, pointcloud=vehicle_pcd, write_ascii=True)      

def intersection_factor(mask1, mask2):
    """
    Jaccard index based 
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0 # If there is no union 
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def merge_semantic_labels(semantic_mask, label2id, merge_dict:dict = None):
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
    if merge_dict is None:
        merge_dict = DEFAULT_MERGE_DICT

    # Merge labels
    for k, vals in merge_dict.items():
        to_id   = label2id[k]
        for v in vals:
            from_id = label2id[v]
            semantic_mask[semantic_mask == from_id] = to_id
    return semantic_mask


def AABB_intersect(A: dict, B:dict) -> bool:
    class AABB:
        def __init__(self, center: tuple, dims: tuple):
            self.minX = center[0] - dims[0]
            self.maxX = center[0] + dims[0]
            self.minY = center[1] - dims[1]
            self.maxY = center[1] + dims[1]
            self.minZ = center[2] - dims[2]
            self.maxZ = center[2] + dims[2]
    A = AABB(A['center'], A['dimensions'])
    B = AABB(B['center'], B['dimensions'])
    return A.minX <= B.maxX and A.maxX >= B.minX and A.minY <= B.maxY and A.maxY >= B.minY and A.minZ <= B.maxZ and A.maxZ >= B.minZ

def AABB_A_bigger_than_B(A: dict, B:dict) -> bool:
    A_dims = A['dimensions']
    B_dims = B['dimensions']
    return A_dims[0] * A_dims[1] * A_dims[2] >= B_dims[0] * B_dims[1] * B_dims[2]

def filter_instances(instance_pcds:dict, min_samples_per_instance:int = 150, max_distance:float = 15.0, max_height:float = 2.0):
            for semantic_pcd in instance_pcds:
                if not semantic_pcd['dynamic']:
                    continue # Skip if non dynamic
                
                # Remove noise pcds from list
                for i, aux in enumerate(semantic_pcd['instance_pcds']):
                    if aux['inst_id'] == -1:
                            semantic_pcd['instance_pcds'].pop(i)
                assert len(semantic_pcd['instance_pcds']) == len(semantic_pcd['instance_3dboxes'])

                # Remove far bboxes and pcds with few points
                added_AABBs = []
                removing_indices = []
                for i in range(len(semantic_pcd['instance_pcds'])):
                    num_samples = semantic_pcd['instance_pcds'][i]['pcd'].shape[0]
                    A = semantic_pcd['instance_3dboxes'][i] # Cuboid for instance pcd
                    height = abs(A['center'][1])
                    
                    # Do not add the instance if it has few samples or the y position is above threshold
                    if num_samples < min_samples_per_instance or height > max_height:
                        removing_indices.append(i)
                        continue
                    
                    # Do not add the instance if it is far away
                    distance = np.linalg.norm( A['center'] )
                    if distance > max_distance:
                        removing_indices.append(i)
                        continue

                    # Do not add the instance if it's cuboid A intersects with an already added instance B 
                    # Check if A is bigger than B. If its the case, remove B and add A
                    replace_index = None
                    for bbox_index in added_AABBs:
                        B = semantic_pcd['instance_3dboxes'][bbox_index]
                        if AABB_intersect(A, B):
                            if AABB_A_bigger_than_B(A, B):
                                replace_index = bbox_index # Replace B with A
                            replace_index = -1 # Dont add A
                            break

                    # The instance has an intersection
                    if replace_index is not None:
                        if replace_index == -1:
                            # Do not add the current instance
                            removing_indices.append(i)
                            continue
                        # Remove the B instance
                        removing_indices.append(replace_index)
                        added_AABBs.pop( added_AABBs.index(replace_index) )
                            
                    # Add the instance
                    added_AABBs.append(i)

                print(f"class: {semantic_pcd['label']} removing indices: {removing_indices}")
                semantic_pcd['instance_pcds']       = [valor for idx, valor in enumerate(semantic_pcd['instance_pcds'])     if idx not in removing_indices]
                semantic_pcd['instance_3dboxes']    = [valor for idx, valor in enumerate(semantic_pcd['instance_3dboxes'])  if idx not in removing_indices]
            
            return instance_pcds

def create_cuboid_edges(center, dims, color = (0.0, 1.0, 0.0)):
        # Desglosamos el centro y las dimensiones
        cx, cy, cz = center # x, y, z
        w, h, d = dims # width, height, depth
        
        # Definimos los vértices de un cuboide centrado en 'center' con dimensiones 'width', 'height', 'depth'
        vertices = np.array([
            [cx - w/2, cy - h/2, cz - d/2],  # Vértice 0
            [cx + w/2, cy - h/2, cz - d/2],  # Vértice 1
            [cx + w/2, cy + h/2, cz - d/2],  # Vértice 2
            [cx - w/2, cy + h/2, cz - d/2],  # Vértice 3
            [cx - w/2, cy - h/2, cz + d/2],  # Vértice 4
            [cx + w/2, cy - h/2, cz + d/2],  # Vértice 5
            [cx + w/2, cy + h/2, cz + d/2],  # Vértice 6
            [cx - w/2, cy + h/2, cz + d/2]   # Vértice 7
        ])

        # Definir las aristas del cuboide, donde cada par de índices representa una línea
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # Aristas de la cara inferior
            [4, 5], [5, 6], [6, 7], [7, 4],  # Aristas de la cara superior
            [0, 4], [1, 5], [2, 6], [3, 7]   # Aristas verticales
        ])

        # Crear un objeto LineSet para dibujar las aristas
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color(color)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(edges))

        cube = o3d.geometry.TriangleMesh.create_box(width=0.005, height=0.005, depth=0.005)
        cube.translate([cx, cy, cz])
        cube.paint_uniform_color([0,0,0])

        return line_set

def create_plane_at_y(y, size:int = 5):
    vertices = np.array([
        [-size, -y, -size],
        [size,  -y, -size],
        [-size, -y,  size],
        [size,  -y,  size]])
    faces = np.array([
        [0, 1, 2],
        [2, 1, 0],
        [3, 2, 1],
        [1, 2, 3]])
    
    # Crear la malla de triángulos
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(vertices)
    plane.triangles = o3d.utility.Vector3iVector(faces)
    plane.paint_uniform_color([0.5, 0.5, 0.5])
    
    return plane


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
    def __init__(self, 
                 dbscan_samples:int = 50, 
                 dbscan_eps:int = 1, 
                 dbscan_jobs:int = None,
                 min_samples:int = None):
        """
        INPUT
            - dbscan_samples:
            - dbscan_eps:
            - dbscan_jobs:
            - merge_semantic_labels: if set to true semantic labels are merged with DEFAULT_MERGE_DICT. 
                label2id is required.
        """
        self.dbscan_samples = dbscan_samples
        self.dbscan_eps     = dbscan_eps
        self.dbscan_jobs    = dbscan_jobs
        self.min_samples    = min_samples 

    def _get_segmented_pcds(self, pcd: np.ndarray, pcd_colors: np.ndarray, seg_mask:np.ndarray, camera_name:str, id2label:dict = nuid2name, id2dynamic: dict = nuid2dynamic):
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
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_samples, n_jobs=self.dbscan_jobs).fit(seg_pcd['pcd'])
            color_map = self._get_cluster_colormap(db.labels_)
            clusters = defaultdict(list)
            for label, point_xyz in zip(db.labels_, seg_pcd['pcd']):
                clusters[label].append(point_xyz)

            # Add cluster pointclouds and calc instance bbox
            seg_pcd['instance_pcds'] = []
            seg_pcd['instance_3dboxes'] = []
            for inst_id, pcd in clusters.items():
                if self.min_samples is not None and len(pcd) < self.min_samples:
                    continue # Skip cluster

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

class InstanceBEVMasks():
    DEFAULT_SELECTED_LABELS = ['vehicle.car']

    def __init__(self, scene:scl.Scene, bev_parameters:draw.TopView.Params, selected_semantic_labels:List[str]=None):
        """
            Get the instance Occupancy/Oclussion masks on the BEV domain.
        """
        self.scene = scene
        self.bev_parameters = bev_parameters
        self.selected_semantic_labels = selected_semantic_labels if selected_semantic_labels is not None else self.DEFAULT_SELECTED_LABELS  
    
    def point2pixel(self, point: tuple[int, int]) -> tuple[int, int]:
        pixel = (
            int(round(point[0] * self.bev_parameters.scale_x + self.bev_parameters.offset_x)),
            int(round(point[1] * self.bev_parameters.scale_y + self.bev_parameters.offset_y)),
        )
        return pixel

    def run(self,
            bev_mask: np.ndarray,
            instance_pcds:dict, 
            frame_num:int, 
            bev_image:np.ndarray = None,
            edge_color: tuple = (0, 255, 255), 
            vert_color:tuple = (0, 0, 255),
            base_color:tuple = (0, 255, 255),
            ):
        """
        INPUT:
            - bev_mask: (H, W, C) where C is going to be ignored
            - instance_pcds: dict of panoptic segmentation of the pointcloud
            - frame_num: frame number
            - bev_image: (H, W, C) if provided, the instance bboxes are going to be drawed on it
            - edge_color: if 
        OUTPUT:
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
                    }],
                    'instance_bev_mask':[{
                        'inst_id': int,
                        'occupancy_mask': (H, W) binary mask,
                        'oclussion_mask': (H, W) binary mask
                    }]
                }]
        """
        
        if len(bev_mask.shape) > 2:
            bev_mask = bev_mask[:, :, 0] # Get the first channel

        # Compute cuboids on BEV image
        for semantic_pcd in instance_pcds:
            # skip non dynamic objects
            if not semantic_pcd['dynamic']:
                continue 
            # skip non selected semantic classes
            if semantic_pcd['label'] not in self.selected_semantic_labels:
                continue
            
            # Find connected components on semantic mask
            semantic_mask = (bev_mask == semantic_pcd['label_id']).astype(np.uint8) 
            num_ccomps, semantic_ccomps = cv2.connectedComponents(semantic_mask, connectivity=8)
            intersection_factors = {}

            # Read 3d cuboids data on image frame
            camera = self.scene.get_camera(semantic_pcd['camera_name'], frame_num)
            semantic_pcd['instance_bev_mask'] = []
            for bbox_index, bbox in enumerate(semantic_pcd['instance_3dboxes']):
                inst_3dbox = create_cuboid_edges(bbox['center'], bbox['dimensions'])
                vertices_3xN = np.asarray(inst_3dbox.points).T

                # Get only base points and edges
                base_verts = [2, 3, 7, 6]
                base_edges = [ [2, 3], [3, 7], [7, 6], [6, 2] ]
                base_poly = []

                # Transform vertices to BEV
                vertices_4xN = utils.add_homogeneous_row(vertices_3xN)
                vertices_2d_3xN, idx = camera.project_points3d(vertices_4xN, remove_outside=False)
                vertices3d_4xN, idx_valid = self.scene.reproject_points2d_3xN_into_plane(vertices_2d_3xN, [0, 0, 1, 0], semantic_pcd['camera_name'], "vehicle-iso8855", frame_num=frame_num)
                _, N = vertices3d_4xN.shape # N = 8

                pixels = []
                for i in range(N):
                    if idx[i] and idx_valid[i]:
                        if np.isnan(vertices3d_4xN[0, i]) or np.isnan(vertices3d_4xN[1, i]):
                            continue
                        pixel = self.point2pixel((vertices3d_4xN[0, i], vertices3d_4xN[1, i]))
                        pixels.append(pixel)
                        # Draw points
                        if bev_image is not None and i in base_verts:
                            cv2.circle(bev_image, pixel, 1, vert_color, 2)
                
                base_poly = np.array([pixels[i] for i in base_verts]).reshape((-1, 1, 2))
                
                # Draw Edges
                if bev_image is not None:
                    for edge in base_edges:
                        cv2.line(bev_image, pixels[edge[0]], pixels[edge[1]], edge_color, 1)
                    
                    # Occupancy is drawn below
                    # mask = np.ones(bev_image.shape, dtype=np.uint8)
                    # mask = cv2.fillPoly(mask, [base_poly], base_color)
                    # bev_image = get_blended_image(bev_image, mask, alpha=0.9)

                # Compute instance mask and save it as occupancy mask
                instance_mask = np.zeros(bev_mask.shape, dtype=np.uint8)
                cv2.fillPoly(instance_mask, [base_poly], 1) # Binary mask
                semantic_pcd['instance_bev_mask'].append({
                    'inst_id': bbox['inst_id'], 
                    'occupancy_mask': instance_mask, 
                    'oclussion_mask': np.zeros((bev_mask.shape[0], bev_mask.shape[1]), dtype=np.uint8)
                    })
                assert len(semantic_pcd['instance_bev_mask']) == bbox_index +1 # Just to make sure

                # Compute intersection factor between the instance_mask and all the semantic components
                intersection_factors[bbox_index] = [None] # M0 is the background
                for c in range(1, num_ccomps):
                    ccomp_mask = np.zeros(semantic_ccomps.shape, dtype=np.uint8)
                    ccomp_mask[semantic_ccomps == c] = 1
                    factor = intersection_factor(instance_mask, ccomp_mask)
                    intersection_factors[bbox_index].append(factor)
            
            # Compute Oclusion masks based on the intersection factors
            for j in range(1, num_ccomps):
                max_val = 0
                max_i_index = -1
                # Iterate each cuboid and get the greatest i-cuboid/j-semantic_component factor
                for i in range(0, len(semantic_pcd['instance_bev_mask'])):
                    f = intersection_factors[i][j]
                    if f > max_val:
                        max_i_index = i 
                        max_val = f
                # If the max_val is not 0, add the j component mask to the i cuboid oclussion mask  

                if max_val > 0:
                    ccomp_mask = np.zeros(semantic_ccomps.shape, dtype=np.uint8)
                    ccomp_mask[semantic_ccomps == j] = 1
                    oc_mask = semantic_pcd['instance_bev_mask'][max_i_index]['oclussion_mask']
                    semantic_pcd['instance_bev_mask'][max_i_index]['oclussion_mask'] = oc_mask | ccomp_mask
                    # cv2.imshow("DEBUG", semantic_pcd['instance_bev_mask'][max_i_index]['oclussion_mask'] * 255)
                    # cv2.waitKey(0)
            
            # Draw Occupancy/Oclussion masks
            if bev_image is not None:
                h, w = semantic_pcd['instance_bev_mask'][0]['occupancy_mask'].shape
                render_mask = np.zeros((h, w, 3), dtype=np.uint8)
                for inst_bev_mask in semantic_pcd['instance_bev_mask']:
                    occupancy = (inst_bev_mask['occupancy_mask'][:, :, None] * np.asarray(base_color)).astype(np.uint8)
                    oclussion = (inst_bev_mask['oclussion_mask'][:, :, None] * np.asarray(base_color) * 0.5).astype(np.uint8)
                    render_mask = render_mask | occupancy | oclussion
                if np.max(render_mask) > 0:
                    bev_image = get_blended_image(bev_image, render_mask) 
                    cv2.imshow("DEBUG", bev_image)
                    cv2.waitKey(0)
                
        return instance_pcds

class InstanceRAWDrawer():
    DEFAULT_DRAWING_LABELS = ['vehicle.car']

    def __init__(self, scene:scl.Scene, drawing_semantic_labels:List[str]=None):
        self.scene = scene
        self.drawing_semantic_labels = drawing_semantic_labels if drawing_semantic_labels is not None else self.DEFAULT_DRAWING_LABELS  

    def run_on_image(self, raw_image:np.ndarray, instance_pcds:dict, frame_num:int, edge_color: tuple = (0, 255, 255), vert_color:tuple = (0, 0, 255)):
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
        
        # Draw cuboids on RAW image
        for semantic_pcd in instance_pcds:
            # skip non dynamic objects
            if not semantic_pcd['dynamic']:
                continue 
            # skip non selected semantic classes
            if semantic_pcd['label'] not in self.drawing_semantic_labels:
                continue
            
            camera = self.scene.get_camera(semantic_pcd['camera_name'], frame_num=frame_num)

            # Read 3d cuboids data on image frame
            for bbox in semantic_pcd['instance_3dboxes']:
                inst_3dbox = create_cuboid_edges(bbox['center'], bbox['dimensions'])
                
                vertices_3xN = np.asarray(inst_3dbox.points).T
                edgesNx2 = np.asarray(inst_3dbox.lines) # pairs

                # Draw Vertices
                vertices_4xN = utils.add_homogeneous_row(vertices_3xN)
                vertices_proj_4xN, idx_valid = camera.project_points3d(points3d_4xN=vertices_4xN, remove_outside=False)
                _, N = vertices_proj_4xN.shape # N = 8
                for i in range(0, N):
                    if idx_valid[i]:
                        if np.isnan(vertices_proj_4xN[0, i]) or np.isnan(vertices_proj_4xN[1, i]):
                            continue
                        center = ( utils.round(vertices_proj_4xN[0, i]), utils.round(vertices_proj_4xN[1, i]))
                        
                        # if not utils.is_inside_image(img_w, img_h, center[0], center[1]):
                        #     continue

                        cv2.circle(raw_image, (int(center[0]), int(center[1])), 1, vert_color, 2)

                for edge in edgesNx2:
                    p1 = (utils.round( vertices_proj_4xN[0, edge[0]] ), utils.round( vertices_proj_4xN[1, edge[0]] ))
                    p2 = (utils.round( vertices_proj_4xN[0, edge[1]] ), utils.round( vertices_proj_4xN[1, edge[1]] ))
                    cv2.line(raw_image, p1, p2, edge_color, 1)

        return raw_image

    def run_on_pointcloud(self, instance_pcds:dict, edge_color:tuple = (0.0, 1.0, 0.0)):
        selected_pcds = []
        all_bboxes = []
        
        # Draw cuboids on Pointcoud
        for semantic_pcd in instance_pcds:
            # skip non dynamic objects
            if not semantic_pcd['dynamic']:
                continue 
            # skip non selected semantic classes
            if semantic_pcd['label'] not in self.drawing_semantic_labels:
                continue
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(semantic_pcd['pcd'])
            pcd.colors = o3d.utility.Vector3dVector(semantic_pcd['pcd_colors'])
            selected_pcds.append(pcd)

            # Read 3d_bboxes
            for bbox_data in semantic_pcd['instance_3dboxes']:
                inst_3dbox = create_cuboid_edges(bbox_data['center'], bbox_data['dimensions'], color=edge_color)
                all_bboxes.append(inst_3dbox) 
                
        return selected_pcds, all_bboxes

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
    BMM = BEVMapManager(scene_path=scene_path, gen_flags={'all': False, 'pointcloud': False, 'instances': False})
    
    raw2seg_bev = Raw2Seg_BEV(raw2segmodel_path, None, device=device)
    raw_seg2bev = Raw_BEV2Seg(bev2segmodel_path, None, device=device)
    raw2seg_bev.set_openlabel(vcd)
    raw_seg2bev.set_openlabel(vcd)
    
    DE  = DepthEstimation(model_path=depth_pro_path, device=device)
    SP  = ScenePCD(scene=scene)
    ISP = InstanceScenePCD(dbscan_samples=15, dbscan_eps = 0.1, dbscan_jobs=None)
    IBM  = InstanceBEVMasks(scene=scene, bev_parameters=raw_seg2bev.bev_parameters)
    IRD = InstanceRAWDrawer(scene=scene)

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
        # raw_mask    = merge_semantic_labels(raw_mask,       raw_seg2bev.label2id)
        # bev_mask_bs = merge_semantic_labels(bev_mask_bs,    raw_seg2bev.label2id)
        bev_mask_sb = merge_semantic_labels(bev_mask_sb,    raw_seg2bev.label2id)

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
            pcd = BMM.load_pointcloud(raw_image_path)

       
        # ##############################################################
        # Generate panoptic pointcloud dict ############################
        print(f"# Generate panoptic pointcloud dict {'#'*28}")
        if BMM.gen_flags['all'] or BMM.gen_flags['instances']:
            instance_pcds = ISP.run(pcd, raw_mask, camera_name, lims=(np.inf, np.inf, np.inf))
            BMM.save_instance_pcds(raw_image_path, instance_pcds)
        else:
            instance_pcds = BMM.load_instance_pcds(raw_image_path)
        
        print(f"semantic labels in scene: {[semantic_pcd['label'] for semantic_pcd in instance_pcds]}")
        instance_pcds = filter_instances(instance_pcds, min_samples_per_instance=250, max_distance=50.0, max_height = 2.0)
        # save_class_pcds(instance_pcds, fk, semantic_labels=["vehicle.car"]) # Save class pcds

        # ##############################################################
        # Draw cuboids on BEV image ####################################
        # Transform cuboids to BEV, compute ConectedComponents of the bev_mask and
        # calc occupancy/occlusion masks of each instance
        print(f"# Draw cuboids on BEV image {'#'*36}")
        bev_blended = get_blended_image(bev_image, raw2seg_bev.mask2image(bev_mask_sb))
        instance_pcds = IBM.run(bev_mask_sb, instance_pcds, frame_num=fk, bev_image=None)
        
        # ##############################################################
        # Draw cuboids on RAW image ####################################
        print(f"# Draw cuboids on RAW image {'#'*36}")
        raw_blended = get_blended_image(raw_image, raw2seg_bev.mask2image(raw_mask))
        raw_image_cuboids = IRD.run_on_image(raw_blended, instance_pcds, frame_num=fk)
        pcd_semantic, pcd_cuboids = IRD.run_on_pointcloud(instance_pcds)
        bev_repoj_cuboids = raw2seg_bev.inverse_perspective_mapping(raw_image_cuboids, camera_name, fk) # Reproyectar cuboides en raw a bev

        # ##############################################################
        # Visualization ################################################
        all_geometries = pcd_semantic + pcd_cuboids + [create_plane_at_y(2.0)]
        cv2.imshow("DEBUG", bev_blended)
        cv2.waitKey(0)
        
        # debug_name = f"{os.path.splitext(os.path.basename(raw_image_path))[0]}.png"
        # cv2.imwrite(os.path.join(scene_path, "debug", "bev_cuboids", f"bev_cuboid_{fk+1}.png"), bev_image_cuboids)
        # cv2.imwrite(os.path.join(scene_path, "debug", "raw_cuboids", f"raw_cuboid_{fk+1}.png"), raw_image_cuboids)
        # cv2.imwrite(os.path.join(scene_path, "debug", "bev_reproj_cuboids", f"bev_reproj_cuboid_{fk+1}.png"), bev_repoj_cuboids)
        
    
        print()
        # Check for a key press (if a key is pressed, it returns the ASCII code)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit
            break
        # o3d.visualization.draw_geometries(all_geometries, 
        #                                   window_name="DEBUG", 
        #                                   zoom=0.8,
        #                                   lookat=[0, 0, 0],
        #                                   up=[0, -1, 0],
        #                                   front=[0, 0, 1])

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
