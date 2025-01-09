from depth_pro.depth_pro import DepthProConfig
import matplotlib.pyplot as plt
import numpy as np
import depth_pro
import torch
import open3d as o3d

from PIL import Image
from vcd import core, scl, draw, utils
import cv2
import os
import re
import pickle

from oldatasets.NuImages.nulabels import nuid2color, nuid2name, nuid2dynamic, nuname2label
from oldatasets.common.utils import target2image, display_images

from typing import List


from sklearn.cluster import DBSCAN
from collections import defaultdict


# FUNCTIONS
def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path doesnt exist: {path}")
    return True

# CLASSES
class DepthEstimation():
    def __init__(self, data_folder:str, model_path:str, device:torch.DeviceObjType, image_extension: str = ".png"):
        self.data_folder = data_folder
        self.input_image_extension = image_extension

        self.input_path = os.path.join(self.data_folder, "input", "images")
        self.output_path = os.path.join(self.data_folder, "output", "depth")

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

    def run_from_token(self, raw_image_token: str):
        """
        INPUT:
            raw_image: token of RGB input image
        """
        depth_out_path = os.path.join(self.output_path,f"{raw_image_token}.tiff")
        if os.path.exists(depth_out_path):
            return cv2.imread(depth_out_path, cv2.IMREAD_UNCHANGED)
        
        if self.model is None:
            print("[DepthEstimation] Loading model...")
            self.model, self.transform = depth_pro.create_model_and_transforms(config=self.DEFAULT_MONODEPTH_CONFIG_DICT, device=self.device)
            self.model.eval()

        raw_path = os.path.join(self.input_path, f"{raw_image_token}_raw{self.input_image_extension}")
        raw_image, _, f_px = depth_pro.load_rgb(raw_path)

        prediction = self.model.infer(self.transform(raw_image), f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        depth_dmap = depth.cpu().detach().numpy()

        # Convert the depth tensor to a NumPy array
        print(f"[DepthEstimation] Saving depth image in {depth_out_path}...")

        depth_image_pil = Image.fromarray(depth_dmap.astype(np.float32), mode='F')
        depth_image_pil.save(depth_out_path)

        return depth_dmap

class SemanticScenePCD():
    def __init__(self, data_folder:str, image_extension:str = '.png'):
        self.data_folder = data_folder
        self.image_extension = image_extension
        
        # INPUT PATHS
        self.openlabel_path = os.path.join(self.data_folder, "input", "openlabel")
        self.semantic_path  = os.path.join(self.data_folder, "input", "semantic")
        self.image_path     = os.path.join(self.data_folder, "input", "images")
        self.depth_path     = os.path.join(self.data_folder, "output", "depth")

        # OUTPUT PATHS
        self.output_pcd_path = os.path.join(self.data_folder, "output", "pointcloud", "raw")
        self.output_seg_path = os.path.join(self.data_folder, "output", "pointcloud", "semantic")

        check_paths([self.openlabel_path, self.semantic_path, self.image_path, self.depth_path, self.output_pcd_path, self.output_seg_path])
        

    def _get_pointcloud(self, depth_map: np.ndarray, camera: scl.Camera, color_image: np.ndarray = None) -> o3d.geometry.PointCloud:
        h, w = depth_map.shape

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
        pcd_points = aux_xyz_3d_coords.reshape(-1, 3) * 100
        

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        if color_image is not None:
            colors = colors.reshape(-1, 3)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def _get_segmented_pcds(self, pcd: np.ndarray, pcd_colors: np.ndarray, seg_mask:np.ndarray, camera_name:str, id2label:dict = nuid2name, id2dynamic: dict = nuid2dynamic):
        pcds = []
        labels = np.unique(seg_mask)
        
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

    def run(self, 
            image: np.ndarray, 
            open_label: core.OpenLABEL, 
            semantic_mask: np.ndarray, 
            depth_map: np.ndarray):
        pass

    def run_from_token(self, image_token,  lims: tuple = (10, 5, 30)):
        # Check if the semantic scene is already generated
        out_file_path = os.path.join(self.output_seg_path, f"{image_token}.plk")
        if os.path.exists(out_file_path):
            with open(out_file_path, "rb") as f:
                segmented_pointclouds = pickle.load(f)
        
                # If there are pre-computed pointclouds, return them. Else compute again
                if len(segmented_pointclouds) > 0:
                    return segmented_pointclouds
        
        # Hay que calcular los pointclouds segmentados
        camera_name = "CAM_FRONT"
        openlabel_path  = os.path.join(self.openlabel_path, f"{image_token}.json")
        image_path      = os.path.join(self.image_path, f"{image_token}_raw{self.image_extension}")
        semantic_path   = os.path.join(self.semantic_path, f"{image_token}_semantic{self.image_extension}")
        depth_path      = os.path.join(self.depth_path, f"{image_token}.tiff")  
        
        # Check all paths
        check_paths([openlabel_path, image_path, semantic_path, depth_path])

        # Load data
        vcd = core.OpenLABEL()
        vcd.load_from_file(openlabel_path)
        scene = scl.Scene(vcd)

        camera = scene.get_camera(camera_name)

        raw_image = cv2.imread(image_path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        depth_dmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        semantic_mask = cv2.imread(semantic_path)[:, :, 0]

        # For Visualization
        alpha = 0.5
        raw_image_float = raw_image.astype(np.float32)
        semantic_mask_rgb = target2image(semantic_mask, nuid2color)
        semantic_mask_rgb_float = semantic_mask_rgb.astype(np.float32)
        blended_image = cv2.addWeighted(raw_image_float, 1 - alpha, semantic_mask_rgb_float, alpha, 0)

        # Compute image pointcloud 
        raw_pcd_path = os.path.join(self.output_pcd_path, f"{image_token}_{camera_name}.pcd")
        try:
            check_paths([raw_pcd_path])
            pcd = o3d.io.read_point_cloud(raw_pcd_path)
        except Exception:
            pcd= self._get_pointcloud(depth_dmap, camera, color_image=blended_image)
            o3d.io.write_point_cloud(filename=raw_pcd_path, pointcloud=pcd, write_ascii=True)      
        pcd_points      = np.asarray(pcd.points)

        # Filter pointcloud and segmented_mask
        mask = (abs(pcd_points[:, 0]) <= lims[0]) & (abs(pcd_points[:, 1]) <= lims[1]) & (abs(pcd_points[:, 2]) <= lims[2])
        semantic_mask   = semantic_mask.flatten()[mask]
        pcd_points      = np.asarray(pcd.points)[mask]
        pcd_colors      = np.asarray(pcd.colors)[mask]

        # Compute segmented pointclouds
        segmented_pointclouds = self._get_segmented_pcds(pcd_points, pcd_colors, semantic_mask, camera_name)
        with open(out_file_path, "wb") as f:
            pickle.dump(segmented_pointclouds, f)

        return segmented_pointclouds

class InstanceScenePCD():
    def __init__(self, data_folder:str):
        self.data_folder = data_folder
        # INPUT_PATHS
        self.semantic_pcd_path = os.path.join(self.data_folder, "output", "pointcloud", "semantic")
        # OUTPUT PATHS
        self.output_inst_path = os.path.join(self.data_folder, "output", "pointcloud", "instances")
        check_paths([self.output_inst_path])
    
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
        
    def run(self, semantic_pointclouds: dict):
        """
        INPUT:
            semantic_pointclouds: [{'label': str, 'label_id': int, 'camera_name': str,'dynamic': bool, 'pcd': np.ndarray, 'pcd_colors': np.ndarray}]
        OUTPUT:
            semantic_pointclouds: 
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
                   
        for seg_pcd in semantic_pointclouds:
            if not seg_pcd['dynamic']:
                continue
            
            # Compute Clusters
            db = DBSCAN(eps = 1, min_samples=50).fit(seg_pcd['pcd'])
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
        return semantic_pointclouds
        
    def run_from_token(self, image_token:str):
        # Check if the instance scene is already generated
        inst_out_path = os.path.join(self.output_inst_path, f"{image_token}.plk")
        if os.path.exists(inst_out_path):
            with open(inst_out_path, "rb") as f:
                return pickle.load(f)
        # Load Input Data
        seg_in_path = os.path.join(self.semantic_pcd_path, f"{image_token}.plk")
        assert os.path.exists(seg_in_path)
        with open(seg_in_path, "rb") as f:
            semantic_pointclouds = pickle.load(f)
        
        instance_pointclouds = self.run(semantic_pointclouds)
        
        # Save instance_pointclouds data
        with open(inst_out_path, "wb") as f:
            pickle.dump(instance_pointclouds, f)
        return instance_pointclouds


class BEVDrawer():
    BEV_MAX_DISTANCE = 50
    BEV_WIDTH = 1024
    BEV_HEIGH = 1024

    def __init__(self, data_folder:str, image_extension:str = ".png"):
        self.data_folder = data_folder
        self.image_extension = image_extension

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

        # INPUT_PATHS
        self.image_path     = os.path.join(self.data_folder, "input", "images")
        self.semantic_path  = os.path.join(self.data_folder, "input", "semantic")
        self.openlabel_path = os.path.join(self.data_folder, "input", "openlabel")
        self.inst_pcd_path = os.path.join(self.data_folder, "output", "pointcloud", "instances")

        # OUTPUT PATHS
        self.output_inst_path = os.path.join(self.data_folder, "output", "pointcloud", "instances")
        check_paths([self.output_inst_path])
    
    def run(self, image_token, semantic_pointclouds: dict):
        """
        INPUT:
            semantic_pointclouds: 
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
        openlabel_path  = os.path.join(self.openlabel_path, f"{image_token}.json")
        image_path      = os.path.join(self.image_path, f"{image_token}_raw{self.image_extension}")
        semantic_path   = os.path.join(self.semantic_path, f"{image_token}_semantic{self.image_extension}")
        camera_name     = "CAM_FRONT" 

        vcd = core.OpenLABEL()
        vcd.load_from_file(openlabel_path)
        scene = scl.Scene(vcd)

        raw_image = cv2.imread(image_path)
        cv2.imshow("debug", raw_image)
        semantic_mask = cv2.imread(semantic_path)[:, :, 0]

        alpha = 0.5
        raw_image_float = raw_image.astype(np.float32)
        semantic_mask_rgb = target2image(semantic_mask, nuid2color)
        semantic_mask_rgb_float = semantic_mask_rgb.astype(np.float32)
        blended_image = cv2.addWeighted(raw_image_float, 1 - alpha, semantic_mask_rgb_float, alpha, 0)

        # Compute BEV image
        drawer = draw.TopView(scene=scene, coordinate_system="vehicle-iso8855", params=self.bev_parameters)
        drawer.add_images({camera_name: blended_image}, frame_num=0)
        map_x = drawer.images[camera_name]["mapX"]
        map_y = drawer.images[camera_name]["mapY"]
        bev = cv2.remap(blended_image, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        bev32 = np.array(bev, np.float32)
        if "weights" in drawer.images[camera_name]:
            cv2.multiply(drawer.images[camera_name]["weights"], bev32, bev32)
        bev_image = bev32.astype(np.uint8)
        cv2.imshow("debug bev", bev_image)

        # Draw cuboids on BEV
        for seg_pcd in semantic_pointclouds:
            if not seg_pcd['dynamic']:
                continue # skip non dynamic objects

            for inst_3dbox in seg_pcd['instance_3dboxes']:
                if inst_3dbox['inst_id'] == -1:
                    continue # skip if there is an unlabeled 3dbox
                
                cx, cy, cz = inst_3dbox['center']
                w, h, d = inst_3dbox['dimensions']
                cy +=h/2 # center on cuboid base
                cz *= -1
                center_3d = (cx, cy, cz)
                vertices_3d = np.array([
                    [cx - w/2, cy, cz - d/2],  # Vértice 0
                    [cx + w/2, cy, cz - d/2],  # Vértice 1
                    [cx + w/2, cy, cz + d/2],  # Vértice 2
                    [cx - w/2, cy, cz + d/2],  # Vértice 3
                ])

                center_pixel = drawer.point2pixel((center_3d[0], center_3d[2]))
                cuboid_pixels = []
                for vert_3d in vertices_3d:
                    vert_2d = (vert_3d[0], vert_3d[2])
                    cuboid_pixels.append(drawer.point2pixel(vert_2d))
                
                print(cuboid_pixels)

                thick = 2
                color = (0, 255, 0)

                cv2.circle(bev_image, center_pixel, 1, color, thick)

                cv2.circle(bev_image, cuboid_pixels[0], 1, color, thick)
                cv2.circle(bev_image, cuboid_pixels[1], 1, color, thick)
                cv2.circle(bev_image, cuboid_pixels[2], 1, color, thick)
                cv2.circle(bev_image, cuboid_pixels[3], 1, color, thick)

                cv2.line(bev_image, cuboid_pixels[0], cuboid_pixels[1], color, thick)
                cv2.line(bev_image, cuboid_pixels[1], cuboid_pixels[2], color, thick)
                cv2.line(bev_image, cuboid_pixels[2], cuboid_pixels[3], color, thick)
                cv2.line(bev_image, cuboid_pixels[3], cuboid_pixels[0], color, thick)

                

        return bev_image

# MAIN
def main(data_folder, model_path):
    image_token     = "60d367ec0c7e445d8f92fbc4a993c67e" 
    image_token     = "0a1fca1d93d04f60a4b12961a22310bb" 
    
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load image and compute depth map
    DE = DepthEstimation(data_folder=data_folder, model_path=model_path, device=device)
    depth_damp = DE.run_from_token(image_token)

    # Load image, semantic mask, computed depth map, computed pointcloud and return semantic pointclouds
    SSP = SemanticScenePCD(data_folder=data_folder)
    semantic_pointclouds = SSP.run_from_token(image_token, lims = (np.inf, np.inf, np.inf))
    
    ISP = InstanceScenePCD(data_folder=data_folder)
    instance_pointclouds = ISP.run_from_token(image_token)

    BEVD = BEVDrawer(data_folder=data_folder)
    bev_image = BEVD.run(image_token, instance_pointclouds)

    cv2.imshow("Semantic BEV Image", bev_image)
    cv2.waitKey(0)

    print("Final Instance Pointclouds")


# ENTRYPOINT
if __name__ == "__main__":
    DATA_FOLDER = "./data"
    MODEL_PATH  = "./models/ml_depth_pro/depth_pro.pt" 

    main(data_folder=DATA_FOLDER, 
         model_path=MODEL_PATH)