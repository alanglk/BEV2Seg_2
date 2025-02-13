from vcd import scl, draw, utils
### Add this to vcd.scl line 1641 for compatibility with non dynamic camera intrinsics
# intrinsic_types = ["intrinsics_pinhole" "intrinsics_fisheye", "intrinsics_cylindrical", "intrinsics_orthographic", "intrinsics_cubemap"]
# sp = vcd_frame["frame_properties"]["streams"][camera_name]["stream_properties"]
# for it in intrinsic_types:
#     # SO, there are dynamic intrinsics!
#     if it in sp:
#         dynamic_intrinsics = True  
#         break
#

from oldatasets.NuImages.nulabels import nuid2color, nuid2name, nuid2dynamic
from oldatasets.common.utils import target2image

from utils import create_cuboid_edges, intersection_factor, get_blended_image, get_pcds_of_semantic_label, get_pallete, filter_instances

from sklearn.cluster import DBSCAN
import open3d as o3d
import numpy as np
import cv2

from collections import defaultdict
from typing import List, Tuple


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
        
    def run(self, 
            pcd: o3d.geometry.PointCloud, 
            semantic_mask:np.ndarray, 
            camera_name:str, 
            lims: tuple = (10, 5, 30),
            min_samples_per_instance=250, 
            max_distance=50.0, 
            max_height = 2.0,
            verbose=False):
        """
        INPUT
            - pcd: open-3d pointcloud
            - semantic_mask: (H, W) semamtic mask
            - camera_name: cam_name from wich pcd is beign computed
            - lims: pre FILTER. (x, y, z) on camera_name frame absolute lims to filter out the raw pcd
            - min_samples_per_instance: post FILTER. Minimun number of points to consider an instance
            - max_distance: post FILTER. Max distance of the centroid from the camera_name origin
            - max_height: post FILTER. Maximun height of an instance
            - verbose: wether to print out info or not
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
        instance_pcds = self._get_segmented_pcds(pcd_points, pcd_colors, semantic_mask, camera_name)


        # Compute instances for each segmented pcd class
        for seg_pcd in instance_pcds:
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
        
        # Apply filter to the detected objects
        if verbose:
            print(f"[InstanceScenePCD] [PostFilter] semantic labels in scene: {[semantic_pcd['label'] for semantic_pcd in instance_pcds]}")
        instance_pcds = filter_instances(instance_pcds, 
                                         min_samples_per_instance=min_samples_per_instance, 
                                         max_distance=max_distance, 
                                         max_height=max_height, 
                                         verbose=verbose)

        return instance_pcds

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
            ) -> Tuple[dict, np.ndarray]:
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
            bev_image with bboxes and occupancy/oclusion masks if provided. Else none.
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
                    # cv2.imshow("DEBUG", bev_image)
                    # cv2.waitKey(0)
                return instance_pcds, bev_image
        return instance_pcds, None

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
    


class OdometryStitching():
    def __init__(self, 
                 scene: scl.Scene,
                 first_frame_data:dict = None,
                 frame_length:int = 0, 
                 pcd_semantic_labels: List = None,
                 cuboid_semantic_labels: List = None,
                 dest_coordinate_system:str = "odom"):
        """
        Create accumulated data using the scene odometry
        """
        self.scene = scene
        self.pcd_semantic_labels = pcd_semantic_labels
        self.cuboid_semantic_labels = cuboid_semantic_labels
        self.cs_dst = dest_coordinate_system


        self.initial_translation_4x1 = np.zeros((4, 1))
        if first_frame_data is not None:
            frame_transforms = first_frame_data['frame_properties']['transforms']
            frame_odometry = frame_transforms['vehicle-iso8855_to_odom']['odometry_xyzypr']
            self.initial_translation_4x1 = utils.add_homogeneous_row(np.asarray(frame_odometry[:3]).reshape((3, 1)))

        self.accum_pcd_points = np.empty((0, 3))
        self.accum_pcd_colors = np.empty((0, 3))
        self.accum_cuboids = []
        
        # Dont use color pallete if frame_length is 0
        self.frame_color_pallete = get_pallete(frame_length).astype(np.float64) / 255.0 if frame_length > 0 else None

    def add_frame_pcd(self, instance_pcds:dict, camera_name:str, frame_num:int, use_frame_color = True) -> o3d.geometry.PointCloud:
        frame_pcds = get_pcds_of_semantic_label(instance_pcds, semantic_labels=self.pcd_semantic_labels)
        transform_4x4, _ = self.scene.get_transform(cs_src=camera_name, cs_dst=self.cs_dst, frame_num=frame_num)

        for fpcd in frame_pcds:
            points_4xN = utils.add_homogeneous_row( np.asarray(fpcd.points).T )
            points_transformed_4xN = transform_4x4 @ points_4xN - self.initial_translation_4x1
            points_transformed_Nx3 = points_transformed_4xN[:-1, :].T

            if use_frame_color and self.frame_color_pallete is not None:
                frame_colors = np.repeat(self.frame_color_pallete[frame_num].reshape((3, -1)), points_transformed_Nx3.shape[0], axis=1).T
            else:
                frame_colors = np.asarray(fpcd.colors)
            
            self.accum_pcd_points = np.vstack((self.accum_pcd_points, points_transformed_Nx3))
            self.accum_pcd_colors = np.vstack((self.accum_pcd_colors, frame_colors))
        
        accum_pcd = o3d.geometry.PointCloud()
        accum_pcd.points = o3d.utility.Vector3dVector(self.accum_pcd_points)
        accum_pcd.colors = o3d.utility.Vector3dVector(self.accum_pcd_colors)
        return accum_pcd
    
    def add_frame_cuboids(self, instance_pcds:dict, camera_name:str, frame_num:int, use_frame_color = True) -> List[o3d.geometry.LineSet]:
        transform_4x4, _ = self.scene.get_transform(cs_src=camera_name, cs_dst=self.cs_dst, frame_num=frame_num)

        frame_color = (0, 1, 0)
        if use_frame_color and self.frame_color_pallete is not None:
            frame_color = tuple(self.frame_color_pallete[frame_num])
        
        # Accumulated cuboids
        for semantic_pcd in instance_pcds:
            # skip non dynamic objects
            if not semantic_pcd['dynamic']:
                continue 
            
            # skip non selected semantic classes
            if self.cuboid_semantic_labels is not None and semantic_pcd['label'] not in self.cuboid_semantic_labels:
                continue
            
            for bbox in semantic_pcd['instance_3dboxes']:
                inst_3dbox = create_cuboid_edges(bbox['center'], bbox['dimensions'], color=frame_color, transform_4x4=transform_4x4, initial_traslation_4x1=self.initial_translation_4x1)
                self.accum_cuboids.append(inst_3dbox)
        
        return self.accum_cuboids