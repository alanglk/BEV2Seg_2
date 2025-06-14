from vcd import core, scl, types, draw
from vcd import utils as vcd_utils

from scipy.optimize import linear_sum_assignment
import numpy as np

from my_utils import get_pallete, DEFAULT_MERGE_DICT, parse_mtl, parse_obj_with_materials
from oldatasets.Occ2.occ2labels import occ2name2id, occ2id2color
from oldatasets.common.utils import target2image
from oldatasets.common.bev_vcd import Dataset2BEV
from src.bev2seg_2 import BEV2SEG_2_Interface

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import seaborn as sns

import open3d as o3d
import textwrap
import threading
import time
import cv2

# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering
from three_d_metrics.open3d_addon import * # 3d metrics

from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay
from scipy.ndimage import label as connected_components_label
import rasterio
from rasterio import features

from collections import deque
import copy

from tqdm import tqdm
from typing import TypedDict, Dict, List, Tuple, Optional, Any, Literal
import argparse
import pickle
import json
import sys
import os


WINDOW_SIZE_FRAMES = 5
# TURNING_THRESHOLD =  np.deg2rad(5.0)
TURNING_THRESHOLD =  np.deg2rad(90.0)
RELATIVE_ROTATION_THRESHOLD = np.deg2rad(15.0)

def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path doesnt exist: {path}")


class ObjInfo(Dict):
    # Dict[core.UID,types.ObjectData]
    # Es mejor dejar que sea dinámico
    pass
class FramePresence(TypedDict):
    str: List[str] # Semantic-type - List of objects UIDs
class AnnotationInfo(TypedDict):
    objects: Dict[str, ObjInfo] # Semantic-type - ObjInfo
    frame_presence: Dict[int, FramePresence] # frame num - FramePresence
class TurningDataType(TypedDict):
    turning_flag: bool
    pos: Tuple[float, float, float]
    ypr: Tuple[float, float, float]

def get_gt_dt_inf(all_objects:ObjInfo, selected_types:List[str]=None, ignoring_names:List[str]=None, filter_out:bool = False) -> Tuple[AnnotationInfo, AnnotationInfo]:
    """
    Return the information stored aboud the selected types of ground_truth and detection annotations. 
    If `filter_out` is set, the not selected types will not be returned. 

    AnnotationInfo format:
    ```
    {
        'objects': {
            'vehicle.car': {
                '0': {vcd},
                '1': {},
                '2': {}
            }
        },
        'frame_presence': {
            0: {'vehicle.car': ['0', '1']},
            1: {'vehicle.car': ['1']},
            2: {}
        }
    }
    
    ```
    """
    gt_objs = {'objects':{}, 'frame_presence': None} 
    dt_objs = {'objects':{}, 'frame_presence': None}

    # Distinguis between GT and Annotations
    for uid, obj in all_objects.items():
        tp = obj['type'] # str
        assert isinstance(tp, str)
        tps = tp.split('/')
        if 'annotated' in tps:
            # Custom annotations
            tp = tps[1] # NuLabel type  
            if tp not in dt_objs['objects']:
                dt_objs['objects'][tp] = {}
            dt_objs['objects'][tp].update({uid:obj})
        else:
            # Scene Ground_truth
            tp = '.'.join(tps).lower() # Convert type to NuLabel format
            if tp not in gt_objs['objects']:
                gt_objs['objects'][tp] = {}
            gt_objs['objects'][tp].update({uid:obj})
    
    # Check if the openlabel has groundtruth and annotations
    if len(gt_objs['objects'].keys()) == 0:
        raise Exception(f"There is no ground truth on the openLABEL")
    if len(dt_objs['objects'].keys()) == 0:
        raise Exception(f"There are no detections (custom annotations) on the openLABEL")
    
    # Drop out the non selected objects   
    def filter_out(obj_dict:Dict[str, ObjInfo], selected_types:List[str]) -> Dict[str, ObjInfo]:
        dropping_keys = []
        for k in obj_dict.keys():
            if k not in selected_types:
                dropping_keys.append(k)
        for k in dropping_keys:
            obj_dict.pop(k)
        return obj_dict
    def ignore_names(obj_dict:Dict[str, ObjInfo], ignoring_names:List[str]) -> Dict[str, ObjInfo]:
        dropping_keys = []
        for tp, obj in obj_dict.items():
            for uid, obj_data in obj.items():
                if obj_data['name'] in ignoring_names:
                    dropping_keys.append((tp, uid))
        for tp, uid in dropping_keys:
            obj_dict[tp].pop(uid)
        return obj_dict

    # Lanes and driveable area (intersections...)
    gt_lanes = {}
    if 'road_segment' in gt_objs['objects']:
        gt_lanes.update( gt_objs['objects']['road_segment'].copy() ) 
    if 'lane' in gt_objs['objects']:
        gt_lanes.update( gt_objs['objects']['lane'].copy() ) 
    
    # Filter out non selected objects from gt
    if filter_out:
        gt_objs['objects'] = filter_out(gt_objs['objects'], selected_types)     if selected_types is not None else gt_objs['objects']
        dt_objs['objects'] = filter_out(dt_objs['objects'], selected_types)     if selected_types is not None else dt_objs['objects']
        gt_objs['objects'] = ignore_names(gt_objs['objects'], ignoring_names)   if ignoring_names is not None else gt_objs['objects']
        dt_objs['objects'] = ignore_names(dt_objs['objects'], ignoring_names)   if ignoring_names is not None else dt_objs['objects']


    # Compute presence of objects in frames
    def get_frame_presence(obj_dict:ObjInfo) -> FramePresence:
        """
        Return the frame_presence dict:
        ```
        'frame_presence':{
            0:{ 'vehicle.car': [0, 1] },
            1:{ 'vehicle.car': [1] },
            2:{ 'vehicle.car': [] },
            3:{}
        }
        ```
        """
        frame_presence = {}
        for tp in obj_dict.keys():
            objs = obj_dict[tp]
            for uid, obj in objs.items():
                assert 'frame_intervals' in obj
                for interval in obj['frame_intervals']:
                    start   = interval['frame_start']
                    end     = interval['frame_end']
                    for fk in range(start, end+1):
                        if not fk in frame_presence:
                            frame_presence[fk] = {}
                        if not tp in frame_presence[fk]:
                            frame_presence[fk][tp] = []
                        frame_presence[fk][tp].append(uid)
        return frame_presence
    gt_objs['frame_presence'] = get_frame_presence(gt_objs['objects'])
    dt_objs['frame_presence'] = get_frame_presence(dt_objs['objects'])

    # Assert same number of continuous frames
    def get_frames_with_no_objs(obj_dict:AnnotationInfo, max_frame:int) -> List[int]:
        frames_with_no_objs  = []
        for fk in range(max_frame+1):
            if not fk in obj_dict['frame_presence']:
                obj_dict['frame_presence'][fk] = {}
                frames_with_no_objs.append(fk)
            elif len(list(obj_dict['frame_presence'][fk].keys())) == 0:
                frames_with_no_objs.append(fk)
            else:
                num_objs = 0
                for _, uids in obj_dict['frame_presence'][fk].items():
                    num_objs += len(uids)
                if num_objs == 0:
                    frames_with_no_objs.append(fk)
        return frames_with_no_objs

    max_frame_gt = list(gt_objs['frame_presence'].keys())[-1]
    max_frame_dt = list(dt_objs['frame_presence'].keys())[-1]
    max_frame = max(max_frame_gt, max_frame_dt)
    gt_frames_with_no_objs = get_frames_with_no_objs(gt_objs, max_frame)
    dt_frames_with_no_objs = get_frames_with_no_objs(dt_objs, max_frame)

    if len(gt_frames_with_no_objs):
        print(f"[GT]    Frames with no entries: {gt_frames_with_no_objs}")
    if len(dt_frames_with_no_objs):
        print(f"[DT]   Frames with no entries: {dt_frames_with_no_objs}")
    return gt_objs, dt_objs, gt_lanes

def merge_semantic_labels(objs_inf:AnnotationInfo, merge_dict:dict = DEFAULT_MERGE_DICT):
    """Return AnnotationInfo with merged semantic types
    """
    # Merge object semantic
    dropping_types = []
    for tp in objs_inf['objects']:
        if tp in merge_dict:
            continue # super-type
        
        for merge_tp in merge_dict:
            if tp in merge_dict[merge_tp]:
                if merge_tp not in objs_inf['objects']:
                    objs_inf['objects'][merge_tp] = {}
                objs_inf['objects'][merge_tp].update(objs_inf['objects'][tp])
                dropping_types.append(tp)

    # Merge frame pressence based on the already merged types
    for frame in objs_inf['frame_presence']:
        for tp in objs_inf['frame_presence'][frame]:
            if tp in dropping_types:
                merge_tp = None
                for m_tp in merge_dict:
                    if tp in merge_dict[m_tp]:
                        merge_tp = m_tp
                if merge_tp is None:
                    continue
                
                if merge_tp not in objs_inf['frame_presence'][frame]:
                    objs_inf['frame_presence'][frame][merge_tp] = []
                objs_inf['frame_presence'][frame][merge_tp] += objs_inf['frame_presence'][frame][tp]
    
    # Drop merged types from objects
    for tp in dropping_types:
        objs_inf['objects'].pop(tp)

    # Drop merged types from frame_pressence
    for frame in objs_inf['frame_presence']:
        for tp in dropping_types:
            if tp in objs_inf['frame_presence'][frame]:
                objs_inf['frame_presence'][frame].pop(tp)
    return objs_inf

def get_types_in_obj_inf(objs_inf:AnnotationInfo) -> list:
    distinct_types = {}
    for tp in objs_inf['objects']:
        if tp not in distinct_types:
            distinct_types.update({tp:0})
    
    for frame in objs_inf['frame_presence']:
        for tp in objs_inf['frame_presence'][frame]:
            assert tp in distinct_types
            distinct_types[tp] += 1

    return list(distinct_types.items())

def get_camera_fov_polygon(scene:scl.Scene, camera_depth:float, fov_coord_sys:str, frame_num) -> np.ndarray:
    """Get the FOV area of a camera
    """
    # TODO: Documentar esto!!
    # Aquí cambiaste alguna mierda en el get_camera
    camera = scene.get_camera(camera_name=fov_coord_sys, frame_num=frame_num)

    h, w = camera.height, camera.width
    d = camera_depth
    K_3x3 = camera.K_3x3
    fx = K_3x3[0, 0]
    alpha = np.arctan((w/2)/fx)
    delta_x = np.tan(alpha) * d
    fov_poly_3d = np.array([[0, 0, 0], [delta_x, 0, d], [-delta_x, 0, d]])

    N, _ = fov_poly_3d.shape # (N, 3) where N is the number of points
    fov_poly_4xN = vcd_utils.add_homogeneous_row(fov_poly_3d.T)
    
    # Transform to the odom coordinate system
    fov_poly_4xN_t = scene.transform_points3d_4xN(fov_poly_4xN, cs_src=fov_coord_sys, cs_dst='odom', frame_num=frame_num)
    fov_poly_3d_t = fov_poly_4xN_t[:3].T  # X, Y, Z
    return fov_poly_3d_t[:, :2]

def bbox_intersects_with_rays(rays_points:np.ndarray, intersections:np.ndarray, bbox_footprint:Polygon) -> np.ndarray:
    does_it_intersects = False
    _, num_rays, num_steps = rays_points.shape

    for j in range(num_rays):
        current_bbox_hits_k = []
        for k in range(num_steps):
            if intersections[j, k] == 2.0:
                if len(current_bbox_hits_k) > 0: 
                    for occl_k in range(current_bbox_hits_k[-1] + 1, num_steps):
                        if intersections[j, occl_k] == 0.0: # Solo si está libre
                            intersections[j, occl_k] = 2.0
                        elif intersections[j, occl_k] == 1.0: # Si era una intersección anterior, ahora es oclusión
                             intersections[j, occl_k] = 2.0
                        elif intersections[j, occl_k] == 2.0: # Si ya estaba ocluida, paramos
                            break
                break # Este rayo ya está ocluido, no hay necesidad de seguir para esta bbox

            point_2d = Point(rays_points[0, j, k], rays_points[1, j, k])
            if bbox_footprint.contains(point_2d):
                does_it_intersects = True 
                if intersections[j, k] == 0.0:
                    intersections[j, k] = 1.0
                current_bbox_hits_k.append(k) 
            else: 
                if current_bbox_hits_k:
                    # Marcar como ocluido desde el punto siguiente al último hit de *esta* bbox.
                    for occl_k in range(current_bbox_hits_k[-1] + 1, num_steps):
                        if intersections[j, occl_k] == 0.0: # Solo si está libre
                            intersections[j, occl_k] = 2.0
                        elif intersections[j, occl_k] == 1.0: # Si era una intersección anterior, ahora es oclusión
                            intersections[j, occl_k] = 2.0
                        elif intersections[j, occl_k] == 2.0: # Si ya estaba ocluida, paramos
                            break # No hay necesidad de seguir, el rayo ya está bloqueado
                    break # Salir del bucle 'k' y pasar al siguiente rayo 'j'

        # Lógica de oclusión final para el rayo si el bucle 'k' terminó sin un 'break'
        if current_bbox_hits_k: # Si esta bbox tuvo al menos un hit en este rayo
            for occl_k in range(current_bbox_hits_k[-1] + 1, num_steps):
                if intersections[j, occl_k] == 0.0:
                    intersections[j, occl_k] = 2.0
                elif intersections[j, occl_k] == 1.0:
                    intersections[j, occl_k] = 2.0
                elif intersections[j, occl_k] == 2.0:
                    break
    
    # for j in range(num_rays):
    #     for k in range(num_steps):
    #         point_2d = Point(rays_points[0, j, k], rays_points[1, j, k])
    #         if bbox_footprint.contains(point_2d):
    #             intersections[j, k] = 1.0 
    #             does_it_intersects = True
    #     ks = np.where(intersections[j] == 1.0)[0]
    #     if len(ks) > 0 and len(intersections[j, ks[-1]:]) > 1: 
    #         intersections[j, ks[-1]:] = 2.0 # Occlusion
    
    # Debug plotting
    # plt.tight_layout()
    # fx, fy = bbox_footprint.exterior.xy
    # plt.plot(fx, fy, color="orange")
    # plt.scatter(rays_points[0, :, :], rays_points[1, :, :], s=0.5, color="blue", label="ray point")
    # js, ks = np.where(intersections == 1.0)
    # plt.scatter(rays_points[0, js, ks], rays_points[1, js, ks], s=0.5, color="red", label="intersection")
    # js, ks = np.where(intersections == 2.0)
    # plt.scatter(rays_points[0, js, ks], rays_points[1, js, ks], s=0.5, color="black", label="occlusion")
    # plt.legend()
    # plt.show()
    return intersections, does_it_intersects

def find_border_indices(js_cluster:np.ndarray, ks_cluster:np.ndarray) -> Tuple[List[int], List[int]]:
    visited_idx = set()
    js_border_min = js_cluster.min()
    js_border_max = js_cluster.max()
    js_border_left, js_border_right, js_border_top, js_border_bottom = [], [], [], []
    ks_border_left, ks_border_right, ks_border_top, ks_border_bottom = [], [], [], []
    for i in range(len(js_cluster)):
        if i in visited_idx:
            continue
        js_ = js_cluster[i]
        idx = np.where(js_cluster == js_)[0]
        if js_ == js_border_min:
            ks_all = ks_cluster[idx].tolist()
            js_border_right += [js_ for _ in range(len(ks_all))]
            ks_border_right += ks_all
        elif js_ == js_border_max:
            ks_all = ks_cluster[idx].tolist()
            js_border_left += [js_ for _ in range(len(ks_all))]
            ks_border_left += ks_all
        else:
            ks_min, ks_max = ks_cluster[idx].min(), ks_cluster[idx].max()
            js_border_bottom.append(js_)
            js_border_top.append(js_)
            ks_border_bottom.append(ks_min)
            ks_border_top.append(ks_max)
        visited_idx.update(idx.tolist())
    
    # Add border indices in counter-clock-wise order
    js_border = js_border_right         + js_border_bottom + js_border_left + js_border_top[::-1]
    ks_border = ks_border_right[::-1]   + ks_border_bottom + ks_border_left + ks_border_top[::-1]
    return js_border, ks_border

def create_multipolygon(rays_points:np.ndarray, 
                        js, ks, 
                        poly_type:Literal["visible", "occuped", "occluded"], 
                        frame_num:int,
                        gen_method:Literal["clustering", "connected_comp"] = "connected_comp",
                        use_clustering:bool = False,
                        clustering_eps:float=0.5, 
                        clustering_jobs:int=2, 
                        gt_occ_bev_masks_path:str="trash") -> MultiPolygon:
    # rays_points[(x, y, z), num_ray, num_step]
    xy_points = rays_points[:2, js, ks].T 

    if len(xy_points) == 0:
        return MultiPolygon()

    grouped_points = {} # Diccionario que almacena los índices de los puntos de cada agrupación
    if gen_method == "clustering":
        cluster_path = os.path.join(gt_occ_bev_masks_path, f"{poly_type}_{frame_num}.pkl")
        if os.path.exists(cluster_path):
            with open(cluster_path, "rb") as f:
                db = pickle.load(f)
        else:
            db = DBSCAN(eps=clustering_eps, n_jobs=clustering_jobs).fit(xy_points)
            with open(cluster_path, "wb") as f:
                pickle.dump(db, f)

        # Find connected components
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_noise_ = list(db.labels_).count(-1)
        original_indices = list(zip(js, ks))  # Emparejar los índices js y ks con las etiquetas
        print(f"n_clusters: {n_clusters_} | n_noise: {n_noise_}")

        # Crear un diccionario para almacenar los índices de los puntos que pertenecen a cada cluster
        for idx, label in zip(original_indices, db.labels_):
            if label == -1:  # Ignorar ruido (-1)
                continue
            if label not in grouped_points:
                grouped_points[label] = []
            grouped_points[label].append(idx)
    
    elif gen_method == "connected_comp":
        # --- START OF CONNECTED COMPONENTS IMPLEMENTATION ---
        # 1. Determine the overall grid size from js and ks to create a 2D mask
        #    We need the maximum j and k values to define the dimensions of our grid.
        #    js and ks are 1D arrays of indices.
        max_j = np.max(js)
        max_k = np.max(ks)

        # Create an empty 2D boolean mask.
        # Dimensions should be (max_j + 1, max_k + 1) because indices are 0-based.
        # Initialize with False
        mask_2d = np.zeros((max_j + 1, max_k + 1), dtype=bool)

        # 2. Populate the mask: set True at the (j, k) positions where rays_points match the poly_type
        #    The `js` and `ks` arrays already represent the points for the current `poly_type`
        #    (filtered implicitly before calling this function).
        mask_2d[js, ks] = True

        # 3. Use scipy.ndimage.label to find connected components
        #    The 'structure' argument defines connectivity (e.g., 8-connectivity for diagonal and cardinal)
        #    [[1,1,1], [1,1,1], [1,1,1]] for 8-connectivity (including center itself, which is fine)
        #    [[0,1,0], [1,1,1], [0,1,0]] for 4-connectivity (cardinal directions only)
        #    Let's use 8-connectivity for a more robust "connected" definition.
        labeled_array, num_features = connected_components_label(mask_2d, structure=np.ones((3,3)))

        print(f"Found {num_features} {poly_type} connected components in frame {frame_num}")

        # 4. Group the original (j, k) indices based on their component label
        #    Iterate through the original js, ks which were already filtered for poly_type.
        for i in range(len(js)):
            j_idx = js[i]
            k_idx = ks[i]
            label = labeled_array[j_idx, k_idx] # Get the label for this (j, k) point

            if label == 0: # Label 0 typically means background/unlabeled, or isolated noise in some contexts
                continue   # Should not happen if js,ks only contain points that are True in mask_2d

            if label not in grouped_points:
                grouped_points[label] = []
            grouped_points[label].append((j_idx, k_idx))
    
    else:
        raise Exception(f"Unknown gen_method: {gen_method}")
    # Crear un poligono por cada grupo de puntos
    polygons = []
    for _, indices in grouped_points.items():
        js_group, ks_group = zip(*indices)
        js_group, ks_group = np.array(js_group), np.array(ks_group)
        
        # Find border indices
        js_border, ks_border = find_border_indices(js_group, ks_group)
        border_pts = np.array([rays_points[:2, j, k] for j, k in zip(js_border, ks_border)]) # Para mantener el orden
        poly = Polygon(border_pts)

        # Debug plotting
        # plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.5, color="black", label="Original")
        # plt.scatter(border_pts[:, 0], border_pts[:, 1], c=np.linspace(0, 1, border_pts.shape[0]), cmap="Blues", label="Border")
        # plt.colorbar(label="Border Order")
        # plt.legend()
        # plt.show()
        
        if poly.is_valid:
            polygons.append(poly)
    return MultiPolygon(polygons)
def plot_ray_polys(multipolygon: MultiPolygon, color: str, ax: plt.Axes = None):
    if ax is None:
        ax = plt.gca() # Get current axes if not provided
    for poly in multipolygon.geoms:
        fx, fy = poly.exterior.xy
        ax.plot(fx, fy, color=color, linewidth=1, zorder=0) # Plot boundary line
        ax.fill(fx, fy, color=color, alpha=0.3, zorder=0) # Fill area
def rasterize_shapely(geometry:Polygon, bev_params:draw.TopView.Params):
    """
    Rasteriza una geometría de Shapely (Polygon o MultiPolygon) en un array numpy.
    Args:
        geometry (Polygon or MultiPolygon): La geometría a rasterizar.
        params: Los parámetros del TopView
    Returns:
        numpy.ndarray: bool np.uint8 array
    """
    min_x, max_x = bev_params.range_x
    min_y, max_y = bev_params.range_y
    width       = bev_params.topview_size[0]
    height      = bev_params.topview_size[1]

    # Crear una transformación afín para mapear las coordenadas del ráster al espacio real
    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, width, height)

    # Rasterizar la geometría
    raster = features.rasterize(
        [(geometry, 1)],  # Lista de tuplas (geometría, valor a asignar)
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Valor para los píxeles fuera de la geometría
        dtype=rasterio.uint8  # Tipo de datos para el ráster
    )

    return raster.astype(bool)  # Convertir a booleano (True si hay geometría)
def get_occlusion_polys(objs_data:dict, 
                        lane_data:dict, 
                        scene:scl.Scene, 
                        fov_coord_sys:str, 
                        frame_num:int, 
                        ray_max_distance:float = 50.0, 
                        bev_coord_sys:str='vehicle-iso8855', 
                        h_sp:float=0.01, 
                        gt_occ_bev_masks_path:str=None
                        ) -> List[np.ndarray]:
    """Get the occlusion of 3d objects by projecting rays in the top-view
    """
    camera = scene.get_camera(camera_name=fov_coord_sys, frame_num=frame_num)

    d = ray_max_distance
    K_3x3 = camera.K_3x3
    fx = K_3x3[0, 0]
    alpha = np.arctan((camera.width/2)/fx) + 0.1
    
    # Launch rays
    num_rays    = 100
    #h_sp       = 0.01    # Longitud de paso
    steps       = int(d / h_sp)
    
    rays_orig   = np.array([[0], [0], [0]]).reshape(3, 1, 1)        # Shape (3, 1, 1)
    rays_angles = np.linspace(-alpha, alpha, num_rays)              # Shape (num_rays,)
    rays_dirs   = np.stack([np.sin(rays_angles), 
                            np.zeros_like(rays_angles),
                            np.cos(rays_angles)], axis=0)           # Shape (3, num_rays)
    rays_dirs   = rays_dirs[:, :, np.newaxis]                       # Shape (3, num_rays, 1)
    rays_steps  = np.linspace(0.0, d, steps)                        # Shape (steps,)
    rays_steps  = rays_steps.reshape(1, 1, -1)                      # Shape (1, 1, steps)
    rays_points = rays_orig + rays_dirs * rays_steps                # Shape (3, num_rays, steps)
    rays_3xN    = rays_points.reshape(3, -1)                        # Shape (3, num_rays * steps)
    
    # Transform rays points to odom frame 
    rays_4xN        = vcd_utils.add_homogeneous_row(rays_3xN)
    transform_4x4   = scene.get_transform(fov_coord_sys, bev_coord_sys, frame_num)[0]
    rays_trans_4xN  = transform_4x4 @ rays_4xN
    rays_trans_3xN  = rays_trans_4xN[:3]
    rays_points     = rays_trans_3xN.flatten().reshape((3, num_rays, steps)) # ((x,y,z), ray index, step index)
    intersections   = np.zeros((num_rays, steps)) # (ray index, step index)

    # Point intersection?
    intersection_footprints = []
    for i in range(len(objs_data)):
        bbox = objs_data[i]['val']
        if objs_data[i]['coordinate_system'] != bev_coord_sys:
            bbox = scene.transform_cuboid(bbox, cs_src=objs_data[i]['coordinate_system'], cs_dst=bev_coord_sys, frame_num=frame_num)
        obx     = get_oriented_bbox_from_vals(bbox)
        aux     = np.asarray(obx.get_box_points())
        assert len(aux) == 8, f"Invalid cuboid input. It has not 8 points: {aux}"
        aux     = aux[[0, 1, 2, 7], :2] # 0, 1, 2, 7 -> Footprint indices
        aux     = np.array([aux[0], aux[1], aux[3], aux[2]])
        bbox_footprint = Polygon(aux)
        intersections, does_it_intersects = bbox_intersects_with_rays(rays_points, intersections, bbox_footprint)
        if does_it_intersects:
            intersection_footprints.append(bbox_footprint)
    intersection_footprints = MultiPolygon(intersection_footprints)

    # Build a Graph -> Identify connected components -> Create Multipolygons
    js, ks = np.where(intersections == 0.0) # visible
    visible_points  = rays_points[:, js, ks]
    visible_polys   = create_multipolygon(rays_points, js, ks, poly_type="visible", frame_num=frame_num, gen_method='connected_comp', gt_occ_bev_masks_path=gt_occ_bev_masks_path)

    js, ks = np.where(intersections == 1.0) # occuped
    occuped_points  = rays_points[:, js, ks]
    occuped_polys   = create_multipolygon(rays_points, js, ks, poly_type="occuped", frame_num=frame_num, gen_method='connected_comp', gt_occ_bev_masks_path=gt_occ_bev_masks_path)

    js, ks = np.where(intersections == 2.0) # occluded
    occluded_points = rays_points[:, js, ks]
    occluded_polys  = create_multipolygon(rays_points, js, ks, poly_type="occluded", frame_num=frame_num, gen_method='connected_comp', gt_occ_bev_masks_path=gt_occ_bev_masks_path)
    
    # Rasterize using the BEV params
    bev_height, bev_width = BEV2SEG_2_Interface.BEV_HEIGH, BEV2SEG_2_Interface.BEV_WIDTH 
    bev_aspect_ratio = bev_width / bev_height
    bev_x_range = (-1.0, BEV2SEG_2_Interface.BEV_MAX_DISTANCE)
    bev_y_range = (-((bev_x_range[1] - bev_x_range[0]) / bev_aspect_ratio) / 2,
                    ((bev_x_range[1] - bev_x_range[0]) / bev_aspect_ratio) / 2)
    bev_params = draw.TopView.Params(
        topview_size        =   (bev_width, bev_height),
        background_color    =   0,
        range_x             =   bev_x_range,
        range_y             =   bev_y_range,
        step_x              =   1.0,
        step_y              =   1.0
        )
    
    bev_mask = np.ones((bev_height, bev_width)) * occ2name2id['background']

    # Rasterize driveable area
    lane_T_4x4, _ = scene.get_transform(cs_src='odom', cs_dst=bev_coord_sys, frame_num=frame_num)
    lane_multipoly = []
    for k in lane_data.keys():
        if not 'object_data' in lane_data[k]:
            continue # Basura que se ha colado supongo
        # Solo hay un poly3d
        assert len(lane_data[k]['object_data']['poly3d']) == 1 
        poly3d_data = lane_data[k]['object_data']['poly3d'][0]
        assert 'closed' in poly3d_data and 'val' in poly3d_data and 'name' in poly3d_data
        assert poly3d_data['name'] == 'polygon' 
        if not poly3d_data['closed'] == True:
            continue # No es un polígono cerrado  
        
        poly3d_vals = np.array(poly3d_data['val'])
        N = poly3d_vals.size // 3 # 3 axis as it is a 3D poly
        poly3d_vals_3xN = poly3d_vals.reshape((N, 3)).T
        poly3d_vals_4xN = vcd_utils.add_homogeneous_row(poly3d_vals_3xN)
        assert np.sum(poly3d_vals_4xN[2, :]) == 0.0,  "All lane poly Zs are 0"
        
        poly3d_vals_4xN = lane_T_4x4 @ poly3d_vals_4xN # Transform to the bev coor sys
        poly3d_vals     = poly3d_vals_4xN[:2, :].T # Just debugging
        lane_poly       = Polygon(poly3d_vals)
        lane_multipoly.append(lane_poly)
        if visible_polys.intersects(lane_poly):
            mask = rasterize_shapely(lane_poly, bev_params=bev_params)
            bev_mask[mask == True] = occ2name2id['driveable']

    # Rasterize other polys
    mask = rasterize_shapely(occluded_polys, bev_params=bev_params)
    bev_mask[mask == True] = occ2name2id['occluded']
    # mask = rasterize_shapely(occuped_polys, bev_params=bev_params)
    mask = rasterize_shapely(intersection_footprints, bev_params=bev_params)
    bev_mask[(mask == True) & (bev_mask != occ2name2id['occluded'])] = occ2name2id['occuped']

    # Apply BEV visible mask
    visible_mask = np.ones((camera.height, camera.width, 3), dtype=np.uint8) * 255
    bev_visible_mask = Dataset2BEV(fov_coord_sys, scene, BEV2SEG_2_Interface.BEV_MAX_DISTANCE, BEV2SEG_2_Interface.BEV_WIDTH, BEV2SEG_2_Interface.BEV_HEIGH)._img2bev(visible_mask, frame_num)[:, :, 0]
    bev_visible_mask = np.asarray(bev_visible_mask, dtype=bool)
    assert len(np.unique(bev_visible_mask)) > 1
    bev_mask = bev_mask * bev_visible_mask

    # Colored mask
    bev_mask_colored = target2image(bev_mask, colormap=occ2id2color)
    
    # For saving masks
    gt_vec_mask_path = os.path.join(gt_occ_bev_masks_path, f"gt_vec_mask_{frame_num}.png")
    gt_ras_mask_path = os.path.join(gt_occ_bev_masks_path, f"gt_occ_mask_{frame_num}.png")
    gt_ras_mask_colored_path = os.path.join(gt_occ_bev_masks_path, f"gt_occ_mask_colored_{frame_num}.png")
    
    # Debug vec
    plt.ioff()
    figure = plt.figure()
    plt.tight_layout()
    plt.scatter(visible_points[0],  visible_points[1],  s=0.5, color="blue", label="visible")
    plt.scatter(occuped_points[0],  occuped_points[1],  s=0.5, color="red", label="intersection")
    plt.scatter(occluded_points[0], occluded_points[1], s=0.5, color="black", label="occlusion")
    # plot_ray_polys(MultiPolygon(lane_multipoly), "#ffc065", label="driveable")
    plot_ray_polys(visible_polys, "blue")
    plot_ray_polys(occuped_polys, "red")
    plot_ray_polys(occluded_polys, "black")
    # plt.legend()
    plt.savefig(gt_vec_mask_path)
    plt.close(figure)
    # plt.show()

    # Debug mask
    cv2.imwrite(gt_ras_mask_path, bev_mask)
    cv2.imwrite(gt_ras_mask_colored_path, bev_mask_colored)
    
    return bev_mask, bev_mask_colored

def plot_path(xs, ys, plot_colors, marker:str='x', zorder_base:int = 1, alpha:float=1.0, ax:Axes = None):
    if ax is None:
        ax = plt.gca()
    marker_zorder = zorder_base
    line_zorder = zorder_base +1
    for i in range(len(xs) - 1):
        # Plot the line segment from current point to next point
        ax.plot(xs[i:i+2], ys[i:i+2], color=plot_colors[i], linewidth=2, zorder=marker_zorder, alpha=alpha)
        if marker is not None:
            # Plot the marker at the current point
            ax.plot(xs[i], ys[i], marker=marker, color=plot_colors[i], markersize=8, zorder=line_zorder, alpha=alpha) 
    if len(xs) > 0 and marker is not None:
        # Ensure the last point also has a marker, as the loop only goes up to the second-to-last
        ax.plot(xs[-1], ys[-1], marker=marker, color=plot_colors[-1], markersize=8, zorder=marker_zorder, alpha=alpha)

def plot_object_path(object_uid, 
                     frame_keys:List[Any], 
                     vcd:core.OpenLABEL, 
                     stationary:bool = False, 
                     color:str="#7700A0",
                     coord_sys:str='odom', 
                     ax:Axes=None):
    assert len(frame_keys) > 0
    if ax is None:
        ax = plt.gca()
    
    if stationary:
        bbox = vcd.get_object_data(object_uid, 'bbox3D', frame_num=frame_keys[0])
        obx = get_oriented_bbox_from_vals(bbox['val'])
        # center = obx.center
        aux = np.asarray(obx.get_box_points())
        aux = aux[np.where(aux <= 1.0)[0]][:, :2]
        aux = aux[:, ::-1]
        aux = np.array([aux[0], aux[1], aux[3], aux[2]])
        poly = Polygon(aux)
        plot_ray_polys(MultiPolygon([poly]), color=color, ax=ax)
        return
    
    positions, colors = [], []
    for fk in frame_keys:
        bbox = vcd.get_object_data(object_uid, 'bbox3D', frame_num=fk)
        obx = get_oriented_bbox_from_vals(bbox['val'])
        center = obx.center
        positions.append(center)
        colors.append(color)
    positions = np.array(positions)
    xs, ys = positions[:, 1], positions[:, 0]
    plot_path(xs, ys, colors, zorder_base=2, alpha=0.7, marker=None, ax=ax)


def get_turning_data(vcd:core.OpenLABEL, 
                     lane_data:dict,
                     WINDOW_SIZE_FRAMES:int=5,
                     TURNING_THRESHOLD:float = np.deg2rad(5.0),
                     map_color:str       = "#FF9800",
                     straight_color:str  = "#00A697",
                     turning_color:str   = "#C00071"
                     ) -> Dict[int, TurningDataType]:
        """
        Number of frames to look back (e.g., 5 frames = 0.5 seconds at 10Hz)
        Yaw change threshold over the window -> e.g., 10 degrees change over 0.5 seconds (5 frames)
        """

        # Get lanes that intersects with vehicle's trajectory
        lane_multipoly = []
        for k in lane_data.keys():
            if not 'object_data' in lane_data[k]:
                continue # Basura que se ha colado supongo
            # Solo hay un poly3d
            assert len(lane_data[k]['object_data']['poly3d']) == 1 
            poly3d_data = lane_data[k]['object_data']['poly3d'][0]
            assert 'closed' in poly3d_data and 'val' in poly3d_data and 'name' in poly3d_data
            assert poly3d_data['name'] == 'polygon' 
            if not poly3d_data['closed'] == True:
                continue # No es un polígono cerrado  
            
            poly3d_vals     = np.array(poly3d_data['val'])
            N               = poly3d_vals.size // 3 # 3 axis as it is a 3D poly
            poly3d_vals_Nx3 = poly3d_vals.reshape((N, 3))
            lane_vals       = poly3d_vals_Nx3[:, :2]
            lane_poly       = Polygon(lane_vals)
            lane_multipoly.append([lane_poly, False]) # lane_poly, Intersects with vehicle
        
        # Distinguis turning and normal areas
        turning_dict:Dict[int, TurningDataType]     = {}
        prev_yaw: Optional[float] = None # Stores the yaw from the previous frame (in radians)
        recent_yaw_diffs: deque[float] = deque(maxlen=WINDOW_SIZE_FRAMES)

        is_any_turn = False
        frame_keys = list(vcd.data['openlabel']['frames'].keys())
        for fk in frame_keys:
            frame_properties = vcd.get_frame(frame_num=fk)['frame_properties']
            frame_odometry  = frame_properties['transforms']['vehicle-iso8855_to_odom']['odometry_xyzypr']
            pos             =  frame_odometry[:3]
            ypr             = frame_odometry[3:]
            current_yaw = ypr[0]

            # Check which lanes instersects with vehicle trajectory
            for i in range(len(lane_multipoly)):
                if lane_multipoly[i][0].intersects(Point(pos[0], pos[1], pos[2])):
                    lane_multipoly[i][1] = True

            is_turning = False
            if prev_yaw is not None:
                yaw_diff = current_yaw - prev_yaw
                # Normalize angle difference to be within -pi to pi radians
                # This handles wrap-around from approx +pi to -pi correctly.
                # math.fmod(x, y) returns x % y, but handles negative x correctly.
                # The result of (x + pi) % (2*pi) - pi will be in (-pi, pi]
                normalized_yaw_diff = np.fmod(yaw_diff + np.pi, 2 * np.pi) - np.pi
                recent_yaw_diffs.append(normalized_yaw_diff)

                # Only start detecting turns once we have enough data in the window
                if len(recent_yaw_diffs) == WINDOW_SIZE_FRAMES:
                    average_yaw_rate = sum(recent_yaw_diffs) / WINDOW_SIZE_FRAMES
                    if abs(average_yaw_rate) > TURNING_THRESHOLD:
                        is_turning = True

            is_any_turn = is_any_turn or is_turning
            turning_dict[fk] = TurningDataType(turning_flag=is_turning, pos=list(pos), ypr=list(ypr))
            prev_yaw = current_yaw
        
        global _debug_plt
        if not  _debug_plt:
            return turning_dict

        # Plotting!!!!
        plot_positions:List[Tuple[float, float]]    = []
        plot_turning_color: List[str] = []
        for k, v in turning_dict.items():
            plot_positions.append((v['pos'][0], v['pos'][1]))
            if v['turning_flag']:
                plot_turning_color.append(turning_color)
            else:
                plot_turning_color.append(straight_color)


        # Get only lanes that intersects with the vehicle trajectory
        lane_multipoly = [lane for lane, intersects_with_vehicle in lane_multipoly if intersects_with_vehicle]
        lane_multipoly = MultiPolygon(lane_multipoly)

        # Change xs for ys
        rotated_lane_polys = []
        for poly in lane_multipoly.geoms:
            rotated_coords = [(y, x) for x, y in poly.exterior.coords]
            rotated_lane_polys.append(Polygon(rotated_coords))
        lane_multipoly = MultiPolygon(rotated_lane_polys)
        
        # Plot the map with vehicle and objects info
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        plot_ray_polys(lane_multipoly, map_color, ax=ax)
        
        plot_positions = np.array(plot_positions)
        xs, ys = plot_positions[:, 1], plot_positions[:, 0]
        plot_path(xs, ys, plot_turning_color, ax=ax)

        # UID 41 -> Parked truck
        parked_truck_color = "#7700A0"
        plot_object_path('41', frame_keys, vcd, color=parked_truck_color, stationary=True, ax=ax)
        parked_truck_patch = Patch(color=parked_truck_color, alpha=0.3, label='Parked Truck')


        # UID 154 -> Followed Van
        van_color = "#FF0064"
        plot_object_path('154', frame_keys, vcd, color=van_color, stationary=False, ax=ax)
        van_line  = plt.Line2D([0], [0], color=van_color, linewidth=2, linestyle='-')

        # UID 129 -> Parked Excavator
        parked_excavator_color = "#00A032"
        plot_object_path('129', frame_keys, vcd, color=parked_excavator_color, stationary=True, ax=ax)
        parked_excavator_patch = Patch(color=parked_excavator_color, alpha=0.3, label='Parked Excavator')

        ax.invert_xaxis()
        ax.set_aspect('equal', adjustable='box')
        legend_objs = []
        
        map_patch = Patch(color=map_color, alpha=0.3, label='Lane') # Use the same alpha as in fill
        straight_line   = plt.Line2D([0], [0], color=straight_color, linewidth=2, linestyle='-', marker='x')
        
        legend_objs = [map_patch, straight_line]
        legend_labels = ['Lane', 'Ego-Vehicle']

        if is_any_turn:
            turning_line    = plt.Line2D([0], [0], color=turning_color, linewidth=2, linestyle='-', marker='x')
            legend_objs.append(turning_line)
            legend_labels.append('Turning')

        legend_objs += [parked_truck_patch, van_line, parked_excavator_patch]
        legend_labels += ['Parked Truck', 'Followed Van', 'Parked Excavator']

        ax.legend(legend_objs, legend_labels, loc='best')
        
        plt.show()

        return turning_dict

def get_relative_rotation_from_ego(objs_data:dict, objs_uids:list, vcd:core.OpenLABEL, frame_num:int, rotation_treshold:float) -> dict:
    obj_rel_rotation = {}
    for i in range(len(objs_data)):
        assert i in range(len(objs_uids))

        frame_properties = vcd.get_frame(frame_num=frame_num)['frame_properties']
        frame_odometry  = frame_properties['transforms']['vehicle-iso8855_to_odom']['odometry_xyzypr']
        v_pos             = frame_odometry[:3]
        v_ypr             = frame_odometry[3:]

        assert objs_data[i]['coordinate_system'] == 'odom', f"annotated object coord frame {objs_data[i]['coordinate_system']} is not 'odom'"
        bbox = objs_data[i]['val'] # [ x, y, z, rx, ry, rz, sx, sy, sz ]
        
        b_uid = objs_uids[i]
        b_pos = [ bbox[0], bbox[1], bbox[2] ]
        b_rot = [ bbox[3], bbox[4], bbox[5] ]
        b_siz = [ bbox[6], bbox[7], bbox[8] ]
        
        yaw_diff = np.abs(v_ypr[0] - b_rot[2]) # yaw - rz
        rotated = yaw_diff > rotation_treshold
        obj_rel_rotation[b_uid] = {'pos': b_pos, 'rot': b_rot, 'size': b_siz, 'yaw_diff':yaw_diff, 'rotated': rotated}
        
    return obj_rel_rotation

def rotate_3d(points, angles):
    """ Rotate a set of 3D points (Nx3) by angles (rx, ry, rz) (radians) around the origin """
    rx, ry, rz = angles
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx  # Apply rotations in X → Y → Z order
    return np.dot(points, R.T)
def bbox_3d_corners(center, size, rotation):
    """ Compute the 8 corner points of the 3D bounding box """
    x, y, z = center
    sx, sy, sz = size
    # Unrotated corner points (relative to center)
    corners = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy, sz],  [sx, -sy, sz],  [sx, sy, sz],  [-sx, sy, sz]
    ])

    rotated_corners = rotate_3d(corners, rotation) # Rotate corners
    rotated_corners += np.array([x, y, z]) # Translate to final position
    return rotated_corners
def project_to_xy(corners):
    """ Project 3D points onto the XY plane """
    return [(x, y) for x, y, z in corners]
def check_intersection_with_fov_poly(polygon:Polygon, bbox:List[float]) -> bool:
    """ Check if the 2D projection of the 3D bounding box intersects with the polygon """
    center = bbox[:3]
    rotation = bbox[3:6]
    size = bbox[6:9]
    corners_3d = bbox_3d_corners(center, size, rotation)  # Get rotated 3D bounding box corners
    footprint_2d = Polygon(project_to_xy(corners_3d)).convex_hull  # Project to XY plane

    global _debug, _debug_fov_intersection
    if _debug and _debug_plt:
        pass
        # px, py = polygon.exterior.xy
        # fx, fy = footprint_2d.exterior.xy
        # plt.plot(px, py, color="blue")
        # plt.plot(fx, fy, color="orange")
        # plt.show()
    return polygon.intersects(footprint_2d) # Check intersection
def get_obj_indices_in_fov(objs_data:List[dict], fov_poly:np.ndarray, camera_name:str, scene:scl.Scene, frame_num:int) -> List[int]:
    """Return the list of indices that intersects with fov_poly on the XY plane
    """
    fov_poly = Polygon(fov_poly)
    in_indices = []
    for i, obj_data in enumerate(objs_data):
        # Transform bbox to camera frame
        bbox_t = obj_data['val']
        if obj_data['coordinate_system'] != 'odom':
            bbox_t = scene.transform_cuboid(obj_data['val'], obj_data['coordinate_system'], camera_name, frame_num)
        if check_intersection_with_fov_poly(fov_poly, bbox_t):
            in_indices.append(i)
    return in_indices


def difference_in_each_dimension(bbox1: Tuple[float], bbox2: Tuple[float]) -> Tuple[float]:
    """
    bboxes on the same coords system.
    bbox1: [ x, y, z, rx, ry, rz, sx, sy, sz ]
    bbox2: [ x, y, z, rx, ry, rz, sx, sy, sz ]
    """
    x1, y1, z1, rx1, ry1, rz1, sx1, sy1, sz1 = bbox1
    x2, y2, z2, rx2, ry2, rz2, sx2, sy2, sz2 = bbox2
    dx, dy, dz = abs(sx1 - sx2), abs(sy1 - sy2), abs(sz1 - sz2)
    return dx, dy, dz

def compute_cost_between_bboxes(bbox1: Tuple[float], bbox2: Tuple[float], dist_type:str = 'v2v ') -> float:
    """
    bboxes on the same coords system.
    bbox1: [ x, y, z, rx, ry, rz, sx, sy, sz ]
    bbox2: [ x, y, z, rx, ry, rz, sx, sy, sz ]
    """
    dist = 0.0

    if dist_type == 'centroids':
        x1, y1, z1, rx1, ry1, rz1, sx1, sy1, sz1 = bbox1
        x2, y2, z2, rx2, ry2, rz2, sx2, sy2, sz2 = bbox2
        dist_2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2) 
        dist = np.sqrt(dist_2)
    elif dist_type == 'v2v':
        bbox1 = get_oriented_bbox_from_vals(bbox1)
        bbox2 = get_oriented_bbox_from_vals(bbox2)
        dist = bbox1.v2v(bbox2)      # Volume-Volume Distance
    else:
        raise Exception(f"Undefined distance type {dist_type}")
    return dist

def get_cost_matrix(gt_bboxes:List[dict], dt_bboxes:List[dict], scene:scl.Scene, frame_num:int, dist_type:str = 'v2v') -> np.ndarray:
    """
    Compute cost matrix with bboxes on 'odom' coordinate system
    {
        "name": "box3D",
        "coordinate_system": "odom",
        "val": [ x, y, z, rx, ry, rz, sx, sy, sz ]
    }
    """
    n = len(gt_bboxes)
    m = len(dt_bboxes)
    cost_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            bbox1, bbox2 = gt_bboxes[i]['val'], dt_bboxes[j]['val']
            if gt_bboxes[i]['coordinate_system'] != 'odom':
                bbox1 = scene.transform_cuboid(bbox1, cs_src=gt_bboxes[i]['coordinate_system'], cs_dst='odom', frame_num=frame_num)
            if dt_bboxes[j]['coordinate_system'] != 'odom':
                bbox2 = scene.transform_cuboid(bbox2, cs_src=dt_bboxes[j]['coordinate_system'], cs_dst='odom', frame_num=frame_num)

            cost_matrix[i][j] = compute_cost_between_bboxes(bbox1, bbox2, dist_type=dist_type)

    return cost_matrix

def assign_detections_to_ground_truth(cost_matrix:np.ndarray, 
                                      max_association_distance:float=3.0, 
                                      scene_name:str=None, 
                                      frame_num:int=None) -> List[int]:
    """
    Cost matrix (N, M) where N is the number of GT elements and M is the
    number of detected elements (custom annotations).
    Returns the modified cost_matrix and the list of assignments
    """

    # Asignaciones predefinidas
    if scene_name is not None and frame_num is not None:
        if frame_num == 26:
            padded_cost_matrix = cost_matrix
            return padded_cost_matrix, [(0, 0)]

    if cost_matrix.size == 0:
        return cost_matrix, [] # There are no elements

    # Define a large value to represent invalid associations
    inf_value = np.max(cost_matrix) + 1

    # Identify detections (columns) where all values exceed max_association_distance
    invalid_detections = np.all(cost_matrix > max_association_distance, axis=0)

    # Remove those invalid detections from the cost matrix
    valid_detections_mask = ~invalid_detections
    filtered_cost_matrix = cost_matrix[:, valid_detections_mask]

    # If all detections are invalid, return an empty assignment
    if filtered_cost_matrix.size == 0:
        return cost_matrix, []

    # Pad to square matrix for LSAP
    n_filtered, m_filtered = filtered_cost_matrix.shape
    max_dim = max(n_filtered, m_filtered)
    padded_cost_matrix = np.full((max_dim, max_dim), inf_value)
    padded_cost_matrix[:n_filtered, :m_filtered] = filtered_cost_matrix

    # Solve LSAP
    row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)

    # Convert indices back to original detection indices
    valid_assignments = [
        (r, np.flatnonzero(valid_detections_mask)[c])
        for r, c in zip(row_ind, col_ind) if r < n_filtered and c < m_filtered
    ]

    # Check if any of the assignments exceeds the max_association_distance
    dropping_assignments = []
    for idx, (i, j) in enumerate(valid_assignments):
        if cost_matrix[i, j] > max_association_distance:
            dropping_assignments.append(idx)
    for idx in dropping_assignments:
        valid_assignments.pop(idx)

    return padded_cost_matrix, valid_assignments



def assign_detections_to_ground_truth_gemini(cost_matrix:np.ndarray, 
                                      max_association_distance:float=3.0,
                                      scene_name:str=None, 
                                      frame_num:int=None) -> List[int]:
    """
    Cost matrix (N, M) where N is the number of GT elements and M is the
    number of detected elements (custom annotations).
    Returns the modified cost_matrix and the list of assignments
    """

    if cost_matrix.size == 0:
        return cost_matrix, []

    if np.any(np.isnan(cost_matrix)):
        raise ValueError("La matriz de costes contiene valores NaN.")

    # --- 1. Filtrar Ground Truths y Detecciones Inválidas ---
    # Un GT/Detección es inválido si TODOS sus posibles emparejamientos tienen un coste > max_association_distance.

    # Si max_association_distance es infinito positivo, no se realiza filtrado por umbral.
    if np.isinf(max_association_distance) and max_association_distance > 0:
        valid_ground_truths_mask    = np.ones(cost_matrix.shape[0], dtype=bool)
        valid_detections_mask       = np.ones(cost_matrix.shape[1], dtype=bool)
    else:
        # Identificar ground truths (filas) y detecciones (columnas) inválidos
        invalid_detections_mask     = np.all(cost_matrix > max_association_distance, axis=0)
        invalid_ground_truths_mask  = np.all(cost_matrix > max_association_distance, axis=1)
        valid_detections_mask       = ~invalid_detections_mask
        valid_ground_truths_mask    = ~invalid_ground_truths_mask

    # Obtener los índices originales de los GTs y Detecciones que se conservan
    original_gt_indices     = np.arange(cost_matrix.shape[0])
    original_det_indices    = np.arange(cost_matrix.shape[1])
    retained_original_gt_indices    = original_gt_indices[valid_ground_truths_mask]
    retained_original_det_indices   = original_det_indices[valid_detections_mask]

    filtered_cost_matrix = cost_matrix[np.ix_(valid_ground_truths_mask, valid_detections_mask)]

    # Si no quedan elementos después del filtrado, devolver la matriz original y sin asignaciones
    if filtered_cost_matrix.size == 0:
        return cost_matrix, []

    # --- 2. Preparar Matriz para LSAP ---
    n_retained_gt, m_retained_det = filtered_cost_matrix.shape
    max_dim = max(n_retained_gt, m_retained_det)

    inf_value = np.max(cost_matrix) + 1.0
    padded_cost_matrix_for_lsap = np.full((max_dim, max_dim), inf_value)
    padded_cost_matrix_for_lsap[:n_retained_gt, :m_retained_det] = filtered_cost_matrix
    row_ind_lsap, col_ind_lsap = linear_sum_assignment(padded_cost_matrix_for_lsap)

    # --- 3. Procesar Asignaciones y Mapear Índices de Vuelta ---
    assignments = []
    for r_idx_in_padded, c_idx_in_padded in zip(row_ind_lsap, col_ind_lsap):
        if (r_idx_in_padded < n_retained_gt) and (c_idx_in_padded < m_retained_det):
            
            # Mapear los índices (de la matriz filtrada/rellenada) a los índices originales
            original_gt_idx = retained_original_gt_indices[r_idx_in_padded]
            original_det_idx = retained_original_det_indices[c_idx_in_padded]

            if cost_matrix[original_gt_idx, original_det_idx] <= max_association_distance:
                assignments.append((original_gt_idx, original_det_idx))
    
    return padded_cost_matrix_for_lsap, assignments


def get_oriented_bbox_from_vals(vals:List[float]) -> OrientedBoundingBox:
    """Return OrientedBoundingBox from three_d_metrics"""
    x, y, z, rx, ry, rz, sx, sy, sz = vals
    center  = [x, y, z] 
    size    = [sx, sy, sz]
    rotation = o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])  
    return OrientedBoundingBox(center, rotation, size)

from matplotlib.widgets import Slider
def compute_3d_detection_metrics(gt_bboxes:List[dict], 
                                 dt_bboxes:List[dict], 
                                 assignments:List[Tuple[int]],
                                 cost_matrix:np.ndarray,
                                 scene:scl.Scene,
                                 frame_num:int, 
                                 gt_uids:List[str]=None, 
                                 dt_uids:List[str]=None
                                 ):
    """Compute 3D detection metics: 
    
    - Precission
    - Recall
    - IoU_v (volumetric)
    - v2v (volume2volume distance)]
    
    Parameters
    --------------
    gt_bboxes, dt_bboxes: 
    ```
    {
        "name": "box3D",
        "coordinate_system": "odom",
        "val": [ x, y, z, rx, ry, rz, sx, sy, sz ]
    }
    ```
    cost_matrix: v2v associated distances
    """   
    global _debug, _debug_plt, _plt_init, _plt_figure, _plt_axes 
    global _plt_3d_gt_bboxes, _plt_3d_dt_bboxes, _plt_3d_labels
    if _debug and _debug_plt:
        if not _plt_init:
            _debug_init_plt(frame_num)
        _debug_clear_plt(frame_num)
        _plt_3d_gt_bboxes    = []
        _plt_3d_dt_bboxes    = []
        _plt_3d_labels       = []
    
    tp = 0 # Detections that are associated
    fp = 0 # Detections that arent associated
    fn = 0 # Groundtruth not associated
    
    dds     = [] # Differences in dimensions
    deds    = [] # Differnces in each dimensions
    v2vs    = [] # Volume-Volume distances
    vious   = [] # Volumetric IoUs
    bbds    = [] # Bounding Boxes Disparities

    n = len(gt_bboxes)
    m = len(dt_bboxes)

    # Find False Negatives
    for i in range(n):
        associated = False
        for a in assignments:
            if i == a[0] or i == a[1]:
                associated = True
        fn += 1 if not associated else 0

    # Find False and True Positives
    for j in range(m):
        associated = False
        for a in assignments:
            if i == a[0] or i == a[1]:
                associated = True
        fp += 1 if not associated else 0
        tp += 1 if associated else 0


    for (i, j) in assignments:
        bbox1, bbox2 = gt_bboxes[i]['val'], dt_bboxes[j]['val']
        if gt_bboxes[i]['coordinate_system'] != 'odom':
            bbox1 = scene.transform_cuboid(bbox1, cs_src=gt_bboxes[i]['coordinate_system'], cs_dst='odom', frame_num=frame_num)
        if dt_bboxes[j]['coordinate_system'] != 'odom':
            bbox2 = scene.transform_cuboid(bbox2, cs_src=dt_bboxes[j]['coordinate_system'], cs_dst='odom', frame_num=frame_num)
        # centroid_distance = compute_cost_between_bboxes(bbox1, bbox2, dist_type='centroids')
        ded = difference_in_each_dimension(bbox1, bbox2)
        
        bbox1 = get_oriented_bbox_from_vals(bbox1)
        bbox2 = get_oriented_bbox_from_vals(bbox2)

        dd  = bbox1.dd(bbox2)       # Difference in Dimensions
        v2v = bbox1.v2v(bbox2)      # Volume-Volume Distance
        iou = bbox1.IoU_v(bbox2)    # Volumetric IoU
        bbd = bbox1.bbd(bbox2)      # Bounding Box Disparity
        
        # ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d', proj_type = 'ortho', elev=30, azim=-80)
        # bbd = bbox1.bbd(bbox2, ax=ax)      # Bounding Box Disparity
        # plt.tight_layout()
        # plt.legend()
        # plt.show()
        
        dds.append(dd)
        deds.append(ded)
        v2vs.append(v2v)
        vious.append(iou)
        bbds.append(bbd)

        if _debug and _debug_plt:
            label = f"{i, j}"
            if gt_uids is not None and dt_uids is not None:
                label = f"{gt_uids[i], dt_uids[j]}"
            _plt_3d_gt_bboxes.append(bbox1)
            _plt_3d_dt_bboxes.append(bbox2)
            _plt_3d_labels.append(label)
            print(f"[3D-Metrics]    {label} DD      Metric: {dd}")
            print(f"[3D-Metrics]    {label} DED     Metric: {ded}")
            print(f"[3D-Metrics]    {label} IoU_v   Metric: {iou}")
            print(f"[3D-Metrics]    {label} V2V     Metric: {v2v}")


    if _debug_plt and len(assignments) > 0:
        if len(assignments) > 1:
            _debug_update_plt_slider(0)
            # El slider no funciona
            # ax_slider = _plt_axes[4]
            # slider = Slider(ax_slider, "Assignment", 0, len(assignments)-1, valinit=0, valstep=1)
            # slider.on_changed(_debug_update_plt_slider)
        else:
            _debug_update_plt_slider(0)

    return tp, fp, fn, dds, deds, v2vs, vious, bbds
# ===========================================================================================
#                                      DEBUG FUNCTIONS                                      =
# =========================================================================================== 
_debug = False
_debug_plt = False
_debug_3d = False

_renderer = None

_plt_init = False
_plt_axes = None
_plt_figure = None 
_plt_curr_frame = -1
_plt_3d_gt_bboxes    = None
_plt_3d_dt_bboxes    = None
_plt_3d_labels       = None

_loading_thread = None
_stop_loading_event = None

def _loading_animation(stop_event, msg:str):
    """'Loading...' animation in another thread"""
    chars = ['|', '/', '-', '\\']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{msg} {chars[i % len(chars)]} ")
        sys.stdout.flush()
        time.sleep(0.2)
        i += 1
    sys.stdout.write(f"\r{msg}... Done! ✅\n") # Final msg
def init_loading_print(msg:str = "Loading"):
    global _stop_loading_event, _loading_thread  # Declarar las variables globales
    _stop_loading_event = threading.Event()  # Crear el evento
    _loading_thread = threading.Thread(target=_loading_animation, args=(_stop_loading_event, msg))
    _loading_thread.start()
def finish_loading_print():
    global _stop_loading_event, _loading_thread  
    if _stop_loading_event:
        _stop_loading_event.set()  
    if _loading_thread:
        _loading_thread.join()  

def _debug_init_plt(frame_num:int):
    global _plt_init, _plt_curr_frame, _plt_figure, _plt_axes
    if not _plt_init: 
        _plt_figure = plt.figure(figsize=(10, 8)) 
        _debug_clear_plt(frame_num)
        _plt_curr_frame = frame_num
        _plt_init = True
        plt.ion()
        plt.show()

def _debug_clear_plt(frame_num:int):
    global _plt_init, _plt_curr_frame, _plt_figure, _plt_axes 
    if _plt_curr_frame != frame_num:
        _plt_curr_frame = frame_num
        _plt_figure.clf()
        _plt_axes = []
        # Add new axes
        ax1 = _plt_figure.add_subplot(2, 2, 1)
        _plt_axes.append(ax1)
        ax2 = _plt_figure.add_subplot(2, 2, 2)
        ax2.axis('off')
        _plt_axes.append(ax2)
        ax3 = _plt_figure.add_subplot(2, 2, 3, projection='3d', proj_type = 'ortho', elev=30, azim=-80)
        _plt_axes.append(ax3)
        ax4 = _plt_figure.add_subplot(2, 2, 4, projection='3d', proj_type = 'ortho', elev=30, azim=-80)
        _plt_axes.append(ax4)
        ax_slider = _plt_figure.add_axes([0.2, 0.02, 0.6, 0.03])
        _plt_axes.append(ax_slider)
        
def _debug_show_plt():
    global _plt_init
    if not _plt_init:
        raise Exception("Cannot update plt without initializing it")
    plt.draw()
    plt.pause(0.001)

def _debug_update_plt_slider(val):
    global _plt_init, _plt_figure, _plt_axes 
    global _plt_3d_gt_bboxes, _plt_3d_dt_bboxes, _plt_3d_labels
    if not _plt_init:
        return
    ax3, ax4 = _plt_axes[2], _plt_axes[3]

    if _plt_3d_labels is None or len(_plt_3d_labels) == 0:
        return
    idx = int(val)
    bbox1, bbox2 = _plt_3d_gt_bboxes[idx], _plt_3d_dt_bboxes[idx]
    plot_bb(bbox1.getT(),default_colors[2]/255, ax=ax3)
    plot_bb(bbox2.getT(),default_colors[8]/255, ax=ax3)
    plot_bb(bbox1.getT(),default_colors[2]/255, ax=ax4)
    plot_bb(bbox2.getT(),default_colors[8]/255, ax=ax4)
    label = _plt_3d_labels[idx]
    ax3.set_title(f"{label} IoU_v")
    ax4.set_title(f"{label} V2V Distance")

    _plt_figure.canvas.draw_idle()

def debug_show_cost_matrix(cost_matrix: np.ndarray, assignments:List[Tuple[int]], gt_labels: List[str], dt_labels: List[str], semantic_type:str, frame_num:int):
    global _debug_plt, _plt_init, _plt_figure, _plt_axes 
    if not _debug_plt:
        return
    if not _plt_init:
       _debug_init_plt(frame_num)
    _debug_clear_plt(frame_num)

    ax1 = _plt_axes[0]
    wrapped_gt_labels = ["\n".join(textwrap.wrap(label, width=15)) for label in gt_labels]
    sns.heatmap(
        cost_matrix,
        annot=True,  # show values
        fmt=".2f",  # value format
        cmap="coolwarm",  # color pallete
        linewidths=0.5,  # lines between cells
        xticklabels=dt_labels,
        yticklabels=wrapped_gt_labels,
        cbar_kws={'label': 'Cost'},
        annot_kws={"size": 8},
        ax=ax1
    )
    ax1.set_xticklabels(dt_labels, fontsize=8)
    ax1.set_yticklabels(wrapped_gt_labels, fontsize=8, rotation=0, ha="right")

    # Mark assingments
    for (i, j) in assignments:
        ax1.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
    ax1.set_ylabel("Ground Truth (gt)")
    ax1.set_xlabel("Detections (dt)")
    ax1.set_title(f"Cost Matrix [frame: {frame_num}]")
    

def debug_load_accum_pcd(vcd:core.OpenLABEL, base_path:str):
    """Return the scene accumulated pointcloud on 'odom' frame"""
    lidar_stream = vcd.get_stream("LIDAR_TOP")
    pcd_path = os.path.join(base_path, lidar_stream['uri'])
    check_paths([pcd_path])
    return o3d.io.read_point_cloud(pcd_path) # on 'odom' frame

class Debug3DRenderer:
    def __init__(self, 
                 vcd:core.OpenLABEL, 
                 scene:scl.Scene, 
                 base_path:str, 
                 frame_num:int,
                 voxel_size:float=0.2, 
                 background_color:Tuple[float]=[0.185, 0.185, 0.185],
                 ego_model_path:str=None,
                 load_pcd:bool = True):
        """
        Init the renderer by loading the accumulated pointcloud
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_action_callback(ord(' '), self._space_callback) # Space to next frame
        self.vis.create_window("3D Debug Renderer")
        self.vis.get_render_option().background_color = background_color

        # Save references to vcd data and scene data
        self.vcd = vcd
        self.scene = scene

        # Load Accumulated PCD and add to the visualizer
        self.load_pcd = load_pcd
        if self.load_pcd:
            init_loading_print("Loading pointcloud")
            accum_pcd = debug_load_accum_pcd(self.vcd, base_path)
            accum_pcd = accum_pcd.voxel_down_sample(voxel_size)
            finish_loading_print()

            self.accum_pcd = accum_pcd
            self.vis.add_geometry(accum_pcd)

        # Save reference to the camera controller
        self.ctr = self.vis.get_view_control()

        # Create the ego_vehicle representation
        center, rotation, size = self._get_ego_frame_data(frame_num)
        self.ego_vehicle_bbox = o3d.geometry.OrientedBoundingBox(center, rotation, size)
        self.ego_bbox_color = [0.0, 1.0, 0.0]
        
        self.ego_mesh = None 
        self.ego_model_path = None # ego_model_path
        if self.ego_model_path is not None:
            self.ego_mesh = self._get_ego_mesh(self.ego_model_path)
            
            max_bounds = self.ego_mesh.get_max_bound()
            min_bounds = self.ego_mesh.get_min_bound()
            self.ego_sizes = np.abs(max_bounds - min_bounds)

            # Aling ego_mesh with odom frame  
            r_3x3 = o3d.geometry.get_rotation_matrix_from_xyz([0.0, -np.pi/2, -np.pi/2])
            self.ego_mesh.rotate(r_3x3) 
            # self.ego_mesh.scale()

        # Set the current frame indicator for rendering the same semantic types of the same frame
        self.current_frame = frame_num
        self.space_pressed = False
        
    def _space_callback(self, vis, key, action):
        print(f"[Debug3DRenderer] Key: {key}, Action: {action}")
        self.space_pressed = key == 1 # 1 key pressed 0 key released

    # def _get_line_material(self) -> rendering.MaterialRecord:
    #     mat = rendering.MaterialRecord()
    #     mat.shader = "defaultLit"
    #     return mat
    # def _get_pcd_material(self) -> rendering.MaterialRecord:
    #     mat = rendering.MaterialRecord()
    #     mat.shader = "defaultUnlit"
    #     return mat

    def _get_ego_frame_data(self, frame_num:int):
        """Return the ego vehicle data in odom frame:
             ego_center_3x1, rotation_3x3, size
        """
        frame_properties = self.vcd.get_frame(frame_num=frame_num)['frame_properties']
        frame_odometry = frame_properties['transforms']['vehicle-iso8855_to_odom']['odometry_xyzypr']
        t_vec =  frame_odometry[:3]
        ypr = frame_odometry[3:]
        r_3x3 = vcd_utils.euler2R(ypr)
        transform_4x4 = vcd_utils.create_pose(R=r_3x3, C=np.array([t_vec]).reshape(3, 1))

        ego_center_3x1 = np.zeros((3, 1))
        ego_center_4x1 = vcd_utils.add_homogeneous_row(ego_center_3x1)
        ego_center_3x1 = (transform_4x4 @ ego_center_4x1)[:3].ravel()

        return ego_center_3x1, r_3x3, [2.0, 2.0, 2.0]

    def _get_ego_mesh(self, ego_model_path) -> o3d.geometry.TriangleMesh | None:
        if ego_model_path is None:
            return None
        
        check_paths([ego_model_path])
        
        files = os.listdir(ego_model_path)
        mtl_files = []
        obj_files = []
        for f in files:
            if f.endswith(".mtl"):
                mtl_files.append(f)
            if f.endswith(".obj"):
                obj_files.append(f)
        assert len(mtl_files) == len(obj_files)

        if len(mtl_files) == 0:
            print(f"[Debug3DRenderer] No supported 3d mesh files were encountered for Ego Vehicle in {ego_model_path}")
            return None
        if len(mtl_files) > 1:
            print(f"[Debug3DRenderer] Multiple mtl and obj files in {ego_model_path}")
            return None
        
        # Parse mtl file and obj file
        mtl_path = os.path.join(ego_model_path, mtl_files[0])
        obj_path = os.path.join(ego_model_path, obj_files[0])            
        materials = parse_mtl(mtl_path)
        vertices, faces, face_materials = parse_obj_with_materials(obj_path)
        
        # Assign colors per vertex (by averaging face colors affecting each vertex)
        # Map from vertices to the colors of the faces they belong to
        vertex_colors = np.ones((len(vertices), 3))  # Default white
        vertex_color_map = {i: [] for i in range(len(vertices))}

        # Assign colors based on face materials
        for face, material in zip(faces, face_materials):
            color = materials.get(material, {"Kd": [1.0, 1.0, 1.0]})["Kd"]
            for v in face:
                vertex_color_map[v].append(color)

        # Average colors for each vertex
        for v, colors in vertex_color_map.items():
            if colors:
                vertex_colors[v] = np.mean(colors, axis=0)

        # Create Open3D mesh
        ego_mesh = o3d.geometry.TriangleMesh()
        ego_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        ego_mesh.triangles = o3d.utility.Vector3iVector(faces)
        ego_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        ego_mesh.compute_vertex_normals()
        return ego_mesh

    def _get_fov_mesh(self, 
                      fov_poly:np.ndarray, 
                      color:np.ndarray = np.array([1.0, 0.0, 0.0])
                      ) -> o3d.geometry.LineSet:
        if np.max(color) > 1.0:
            color = color.astype(float) / 255.0

        # to 3D coords (Z=0)
        N, _ = fov_poly.shape  # (N, 2), points in XY
        fov_poly_3d = np.hstack([fov_poly, np.zeros((N, 1))])  # (X, Y) -> (X, Y, Z=0)

        if N < 3:
            raise ValueError("[Debug3DRenderer] ERROR: Less than 3 points to create a fov mesh")

        # Delaunay triangulation to create a surface in XY
        edges = []
        if N == 3:
            edges = [(0, 1), (1, 2), (2, 0)]
        else:
            tri = Delaunay(fov_poly)
            edges = set()
            for triangle in tri.simplices:
                for i in range(3):  
                    edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
                    edges.add(edge)
            edges = list(edges)
            if not edges:
                edges = [(i, (i + 1) % N) for i in range(N)]

        # Crear LineSet
        fov_lines = o3d.geometry.LineSet()
        fov_lines.points = o3d.utility.Vector3dVector(fov_poly_3d)
        fov_lines.lines = o3d.utility.Vector2iVector(edges)
        fov_lines.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
        return fov_lines

    def _get_text_mesh(self,
                       text:str, 
                       position:np.ndarray,
                       size:float=0.1, 
                       color:np.ndarray=np.array([1, 0, 0])
                       ) ->  o3d.geometry.TriangleMesh:
        if np.max(color) > 1.0:
            color = color.astype(float) / 255.0
        text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.3).to_legacy()
        text_mesh.scale(size, center=(0, 0, 0))
        text_mesh.paint_uniform_color(color)  # Text color
        text_mesh.translate(position) # Translate to point
        
        # Invertir el orden de los índices de los triángulos
        triangles = np.asarray(text_mesh.triangles)
        triangles = triangles[:, [2, 1, 0]]  
        text_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        text_mesh.compute_vertex_normals()

        return text_mesh

    def _get_arrow_mesh(self, 
                   A:np.ndarray, 
                   B:np.ndarray, 
                   cone_height:float=0.2, 
                   cylinder_radius:float=0.05,
                   color:Tuple[float]=(1, 0, 0)
                   ) ->  o3d.geometry.TriangleMesh:
        arrow_direction = B - A 
        arrow_length = np.linalg.norm(arrow_direction)
        arrow_direction /= arrow_length             # Normalize
        z_axis = np.array([0, 0, 1])    # Reference arrow vector in o3d

        # Get arrow rotation matrix
        arrow_dot   = np.dot(z_axis, arrow_direction)
        arrow_cross = np.cross(z_axis, arrow_direction)
        arrow_angle = np.arccos(np.clip(arrow_dot, -1.0, 1.0))
        axis_height = np.linalg.norm(arrow_cross)
        if axis_height < 1e-6:
            rotation_matrix = np.eye(3) # Identity matrix (no rotation)
        else:
            arrow_cross /= axis_height # Normalize
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(arrow_cross * arrow_angle)
            
        # Create arrow
        arrow_cylinder_height = arrow_length - cone_height
        arrow_cylinder_height = arrow_cylinder_height if arrow_cylinder_height > 0 else 0.1
        arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cylinder_radius, cone_radius=cylinder_radius * 1.5,
                cylinder_height=arrow_cylinder_height,
                cone_height=cone_height)
        arrow.rotate(rotation_matrix, center=(0, 0, 0))  # Apply rotation matrix
        arrow.translate(A)  # A as Origin
        arrow.paint_uniform_color(color)  # Red
        return arrow


    def _get_bboxes3d(self, 
                     uids: List[str], 
                     frame_num:int, 
                     color:np.ndarray = np.array([135, 206, 250])
                     ) -> List[o3d.geometry.LineSet]:
        if np.max(color) > 1.0:
            color = color.astype(float) / 255.0
        geometries = []
        for uid in uids:
            bbox = self.vcd.get_object_data(uid,'bbox3D', frame_num=frame_num)
            if "val" not in bbox:
                print(f"[Debug3DRenderer] Missing bbox3D values for object with UID {uid}")
            
            obx = get_oriented_bbox_from_vals(bbox['val'])
            center = obx.center
            obx = o3d.geometry.LineSet.create_from_oriented_bounding_box(obx)
            obx.paint_uniform_color(color) # GT Color

            # Add uids text labels to bboxes
            text = self._get_text_mesh(str(uid), center, color=color)
            
            # Add geometries
            geometries.append(obx)
            geometries.append(text)
        return geometries

    def _get_associated_arrows(self, 
                               gt_uids:List[str], 
                               dt_uids:List[str], 
                               assignments:List[Tuple[int]], 
                               frame_num:int, 
                               arrow_cone_height:float=0.2, 
                               arrow_cylinder_radius:float=0.05,
                               arrow_color:np.ndarray = np.array([255, 0, 0])
                               ) -> List[o3d.geometry.LineSet]:
        
        if np.max(arrow_color) > 1.0:
            arrow_color = arrow_color.astype(float) / 255.0
        geometries = []
        for index, (i, j) in enumerate(assignments):
            obj_uid_1 = gt_uids[i]
            obj_uid_2 = dt_uids[j]

            bbox1 = self.vcd.get_object_data(obj_uid_1,'bbox3D', frame_num=frame_num) # GT
            bbox2 = self.vcd.get_object_data(obj_uid_2,'bbox3D', frame_num=frame_num) # DT
            if "val" not in bbox1:
                print(f"[Debug3DRenderer] Missing bbox3D values for bbox1 with UID {obj_uid_1}")
            if "val" not in bbox2:
                print(f"[Debug3DRenderer] Missing bbox3D values for bbox2 with UID {obj_uid_2}")

            x1, y1, z1, rx1, ry1, rz1, sx1, sy1, sz1 = bbox1['val']
            x2, y2, z2, rx2, ry2, rz2, sx2, sy2, sz2 = bbox2['val']            
            # Compute arrow between GT and DT
            arrow = self._get_arrow_mesh(np.array([x1, y1, z1]), np.array([x2, y2, z2]))

            # Add geometries
            geometries.append(arrow)
        return geometries
    
    def _update_camera_ego(self, frame_num):
        """Update camera and ego position"""
        ego_center, r_3x3, _ = self._get_ego_frame_data(frame_num)
        self.ego_vehicle_bbox.center = ego_center
        self.ego_vehicle_bbox.R = r_3x3
        ego_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(self.ego_vehicle_bbox)
        ego_set.paint_uniform_color(self.ego_bbox_color)
        
        ego_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=ego_center)
        ego_axis.rotate(r_3x3)

        if self.ego_mesh is not None:
            ego_mesh = copy.deepcopy(self.ego_mesh)
            ego_mesh.rotate(r_3x3)
            ego_mesh_center = ego_center 
            ego_mesh_center[1] -= -self.ego_sizes[1] / 2
            ego_mesh.translate(ego_mesh_center)
            self.vis.add_geometry(ego_mesh)

        self.vis.add_geometry(ego_set)
        self.vis.add_geometry(ego_axis)

        # Configurar la cámara para mirar hacia abajo (mirada top-down)
        self.ctr.set_zoom(0.2)  
        self.ctr.set_front([0, 0, 1])       # Top-view
        self.ctr.set_lookat(ego_center)     # Apuntar al centro del vehículo (ego)
        self.ctr.set_up([0, -1, 0])         # Alinear eje Y como el 'arriba' de la cámara

    def update(self, 
               gt_uids:List[str], 
               dt_uids:List[str], 
               assignments:List[Tuple[int]], 
               semantic_type:List[str], 
               frame_num:int,
               fov_poly:np.ndarray=None,
               gt_in_indices:List[int]=None, 
               dt_in_indices:List[int]=None):

        """
        Update scene with frame cuboids associations and move the camera setting ego_vehicle position at the window center

        Args:
            gt_uids: Ground Truth objects uids of semantic_type detected on current frame
            dt_uids: Detected objects uids of semantic_type detected on current frame
            assignments: associations between gt and dt
            frame_num (int): current frame
            fov_poly: polygon of visible area
            gt_in_indices: indices of visible objects in GT
            dt_in_indices: indices of visible objects in detections
        """

        # Remove prev geometries except the pointcloud if we have changed the
        # frame index
        if self.current_frame != frame_num:
            self.current_frame = frame_num
            self.space_pressed = False 
            self.vis.clear_geometries()
            if self.load_pcd:
                self.vis.add_geometry(self.accum_pcd)

        # De momento semantic_types no lo utilizo porque solo trabajo con vehículos.
        # la idea es que se pueda diferenciar las clases semánticas por color
        all_geometries = []
        gt_bboxes = []
        dt_bboxes = []
        arrows = []
        
        # FOV -> Light Butter Yellow
        if fov_poly is not None:
            all_geometries += [self._get_fov_mesh(fov_poly, np.array([250, 230, 160]))] 

        # In and Out FOV GT and Detectios + Association Arrows
        if gt_in_indices is not None and dt_in_indices is not None: 
            # Ground Truth -> Light Blue
            gt_in_uids      = [gt_uids[i] for i in gt_in_indices]
            gt_out_uids     = [gt_uids[i] for i in range(len(gt_uids)) if i not in gt_in_indices]
            gt_bboxes_in    = self._get_bboxes3d(gt_in_uids,    frame_num, np.array([75, 156, 220]))    
            gt_bboxes_out   = self._get_bboxes3d(gt_out_uids,   frame_num, np.array([19, 64, 99]))
            gt_bboxes += gt_bboxes_in + gt_bboxes_out
            
            # Detections -> Pale Green
            dt_in_uids      = [dt_uids[i] for i in dt_in_indices]
            dt_out_uids     = [dt_uids[i] for i in range(len(dt_uids)) if i not in dt_in_indices]
            dt_bboxes_in    = self._get_bboxes3d(dt_in_uids,    frame_num, np.array([102, 201, 102]))    
            dt_bboxes_out   = self._get_bboxes3d(dt_out_uids,   frame_num, np.array([33, 93, 33]))
            dt_bboxes += dt_bboxes_in + dt_bboxes_out

            # Associations
            arrows = self._get_associated_arrows(gt_in_uids, dt_in_uids, assignments, frame_num, arrow_color=np.array([240, 230, 140])) # Khaki
        else:
            gt_bboxes = self._get_bboxes3d(gt_uids, frame_num, np.array([75, 156, 220]))  # GT -> Light Blue
            dt_bboxes = self._get_bboxes3d(dt_uids, frame_num, np.array([102, 201, 102])) # DT -> Pale Green
        all_geometries += gt_bboxes + dt_bboxes + arrows

        for g in all_geometries:
            self.vis.add_geometry(g)

        # Upate camera and ego
        self._update_camera_ego(frame_num)

        # Refrescar visualización
        print("[Debug3DRenderer] Press Space to continue to next frame")
        while not self.space_pressed:
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        """Cierra la ventana de visualización."""
        self.vis.destroy_window()


# ===========================================================================================
#                                            MAIN                                           =
# =========================================================================================== 
"""
data = {
    'scene_name_eval0':{
        'openlabel_path': str,
        'model_config':{
            'scene':{
                'scene_path':...,
                'camera_name': 'CAM_FRONT'
            },
            'semantic':{
                'raw2segmodel_path': ...,
                'bev2segmodel_path': ...,
                'merge_semantic_labels_flag': True,
                'merge_dict': DEFAULT_MERGE_DICT,
            },
            'depth_estimation':{
                'depth_pro_path':...
            },
            'scene_pcd':{
            },
            'instance_scene_pcd':{
                'dbscan_samples': 15,
                'dbscan_eps': 0.1,
                'dbscan_jobs': None,
                'lims': (np.inf, np.inf, np.inf),
                'min_samples_per_instance': 250,
                'max_distance': 50.0,
                'max_height': 2.0
            }
        },
        'eval_config':{
            'camera_depth': 15.0,
            'max_association_distance': 7.0,
            'semantic_types': ['vehicle.car'],
            'ignoring_names': ['ego_vehicle']
        },
        'frames': {
            0: {
                'semantic_type':{
                    'num_gt_objs': 0,
                    'num_dt_objs': 0,
                    
                    'metrics':{
                        'tp': 0,
                        'fp': 0,
                        'fn': 0,
                        'IoU_v': [...],
                        'v2v_dist': [...],
                        'bbd': [...]
                    }
                }
            }
        }
        
    }
}
"""
def main(
        scene_path:str,
        openlabel_path:str,
        save_data_path:str,
        semantic_types: List[str],
        ignoring_names: List[str],
        gt_occ_bev_masks_path:str=None,
        camera_depth:float = BEV2SEG_2_Interface.BEV_MAX_DISTANCE,
        max_association_distance:float = 3.0,
        association_dist_type:str = 'v2v',
        save_openlabel_path:str = None,
        debug:bool=False,
        debug_ego_model_path:str=None
        ):

    

    # Check wheter the file exists
    check_paths([scene_path, openlabel_path])
    if save_openlabel_path is not None:
        check_paths([save_openlabel_path])
        save_openlabel_path = os.path.abspath(save_openlabel_path)
    scene_name      = os.path.basename(scene_path)
    save_data_path  = os.path.abspath(save_data_path)
    
    if gt_occ_bev_masks_path is not None:
        check_paths([gt_occ_bev_masks_path])

    # Debug init
    global _renderer, _debug, _debug_3d, _debug_plt
    _debug = debug
    if _debug:
        _debug_3d = True
        _debug_plt = True
        print(f"[DEBUG] Debugging evaluation of 3d intances -> _debug_3d: {_debug_3d}, _debug_plt: {_debug_plt}")

    # Load OpenLABEL
    vcd = core.OpenLABEL()
    vcd.load_from_file(openlabel_path)
    scene = scl.Scene(vcd)

    # Get model config from metadata
    metadata = vcd.get_metadata()
    model_config = None
    if 'model_config' not in metadata:
        raise Exception("OpenLABEL file provided has not metadata for evaluation")
    model_config = metadata['model_config']

    # Evaluation params
    assert scene_name == metadata['scene_name'], f"scene folder name: {scene_name} does not match with metadata scene name: {metadata['scene_name']}"
    eval_name                   = metadata['scene_name']
    camera_name                 = model_config['scene']['camera_name']
    merge_semantic_labels_flag  = model_config['semantic']['merge_semantic_labels_flag'] # True
    merge_dict                  = model_config['semantic']['merge_dict'] # DEFAULT_MERGE_DICT
    eval_config = {
        'camera_depth': camera_depth,
        'max_association_distance': max_association_distance,
        'association_dist_type': association_dist_type,
        'semantic_types': semantic_types,
        'ignoring_names': ignoring_names
    }
    
    # Create data dict for saving evaluation data
    data = {}
    if os.path.exists(save_data_path) and not _debug:
        print(f"Loading evaluation data from: {save_data_path}")
        with open(save_data_path, "rb") as f:
            data = pickle.load(f)

        if eval_name in data:
            print(f"Scene {eval_name} is already evaluated with this params:")
            print("============= model_config =============") 
            print(f"{json.dumps(model_config, indent=4)}\n")
            print("============= eval_config ==============") 
            print(f"{json.dumps(eval_config, indent=4)}\n")
            
            res = input(f"Do you want to evaluate it again? [Y/n]")
            if res.lower() != 'y':
                print("Finish!! :D")
                return
            else:
                new_name = input(f"Input new evaluation name (prev_name -> {eval_name}): ")
                while new_name in data:
                    new_name = input(f"{new_name} is already registered. Input another name: ")
                eval_name = new_name
    print(f"Evaluating scene and saving as {eval_name} in {save_data_path}")
    data[eval_name] = {}
    data[eval_name]['openlabel_path']   = openlabel_path
    data[eval_name]['model_config']     = model_config
    data[eval_name]['eval_config']      = eval_config
    data[eval_name]['frames'] = {}

    # Selected types = semantic_types + merge_dict_sub_types   
    selected_types = []
    for tp in semantic_types:
        if not merge_semantic_labels_flag or merge_dict is None:
            selected_types.append(tp)
            continue
        if tp not in merge_dict:
            selected_types.append(tp)
            continue
        for sub_tp in merge_dict[tp]:
            selected_types.append(sub_tp)

    # Get the selected ground_truth and annotated objects 
    all_objs_inf = vcd.get_objects()
    gt_objs_inf, dt_objs_inf, gt_lanes = get_gt_dt_inf(all_objs_inf, selected_types=selected_types, ignoring_names=ignoring_names, filter_out=True)
    
    # If merge labels was applied to the object detection, it has to be considered
    # for the groundtruth
    if merge_semantic_labels_flag:
        gt_objs_inf = merge_semantic_labels(gt_objs_inf) # Merge semantic labes as was made to custom detections
        dt_objs_inf = merge_semantic_labels(dt_objs_inf) # It does nothing (in theory)


    # Get Turning or not Information dict
    turning_data = get_turning_data(vcd, lane_data=gt_lanes, WINDOW_SIZE_FRAMES=WINDOW_SIZE_FRAMES, TURNING_THRESHOLD=TURNING_THRESHOLD)
    print(f"Turning flags: {[v['turning_flag'] for _, v in turning_data.items()]}")

    # Main loop
    frame_keys = vcd.data['openlabel']['frames'].keys()
    for fk in tqdm(frame_keys, desc="frames"):
        gt_uids_in_frame = gt_objs_inf['frame_presence'][fk] # Ground_truth of frame
        dt_uids_in_frame = dt_objs_inf['frame_presence'][fk] # Detections of frame
        data[eval_name]['frames'][fk] = {}

        for tp in semantic_types:
            if tp not in gt_uids_in_frame or tp not in dt_uids_in_frame:
                continue

            gt_uids = gt_uids_in_frame[tp]
            dt_uids = dt_uids_in_frame[tp]
            
            gt_objs_data = [ vcd.get_object_data(uid=uid, data_name='bbox3D', frame_num=fk) for uid in gt_uids]
            dt_objs_data = [ vcd.get_object_data(uid=uid, data_name='bbox3D', frame_num=fk) for uid in dt_uids]
            
            if len(gt_objs_data) == 0 or len(dt_objs_data) == 0:
                continue # Skip
            


            fov_poly = get_camera_fov_polygon(scene, camera_depth, camera_name, fk) # (O, A, B )in odom cs
            gt_in_indices = get_obj_indices_in_fov(gt_objs_data, fov_poly, camera_name, scene, fk)
            dt_in_indices = get_obj_indices_in_fov(dt_objs_data, fov_poly, camera_name, scene, fk)

            gt_in_uids = [gt_uids[i] for i in gt_in_indices]
            dt_in_uids = [dt_uids[i] for i in dt_in_indices]
            gt_in_objs_data = [gt_objs_data[i] for i in gt_in_indices]
            dt_in_objs_data = [dt_objs_data[i] for i in dt_in_indices]

            if gt_occ_bev_masks_path is not None:
                bev_mask, bev_mask_colored = get_occlusion_polys(gt_in_objs_data, gt_lanes, scene=scene, fov_coord_sys=camera_name, frame_num=fk, ray_max_distance=40.0, h_sp=0.01, gt_occ_bev_masks_path=gt_occ_bev_masks_path)
            
            objects_relative_rotation = get_relative_rotation_from_ego(gt_in_objs_data, gt_in_uids, vcd=vcd, rotation_treshold=RELATIVE_ROTATION_THRESHOLD, frame_num=fk)

            cost_matrix = get_cost_matrix(gt_in_objs_data, dt_in_objs_data, scene=scene, frame_num=fk, dist_type=association_dist_type)
            padded_cost_matrix, assignments = assign_detections_to_ground_truth(cost_matrix, max_association_distance=max_association_distance, scene_name=scene_name, frame_num=fk)

            _tp, _fp, _fn, _dd, _ded, _IoU_v, _v2v_dist, _bbd = compute_3d_detection_metrics(gt_in_objs_data, dt_in_objs_data, assignments, cost_matrix, scene=scene, frame_num=fk, gt_uids=gt_in_uids, dt_uids=dt_in_uids) 
            data[eval_name]['frames'][fk][tp] = { 'gt_uids': gt_in_uids, 'dt_uids': dt_in_uids, 'assignments':assignments, 'metrics':{ 'tp': _tp, 'fp': _fp, 'fn': _fn, 'dd':_dd, 'ded':_ded, 'IoU_v': _IoU_v, 'v2v_dist': _v2v_dist, 'bbd': _bbd }, 'gt_rel_rotation': objects_relative_rotation, 'vehicle_turning_data':turning_data[fk] }

            if _debug:
                
                if _debug_plt:
                    debug_show_cost_matrix(padded_cost_matrix, assignments, gt_in_uids, dt_in_uids, tp, fk)
                    _debug_show_plt()

                # Render 3d scene
                if _debug_3d:
                    if _renderer is None:
                        _renderer = Debug3DRenderer(vcd, 
                                                    scene, 
                                                    base_path=scene_path, 
                                                    frame_num=fk, 
                                                    ego_model_path=debug_ego_model_path, 
                                                    load_pcd=False)
                    _renderer.update(gt_uids, 
                                    dt_uids, 
                                    assignments, 
                                    semantic_type=tp, 
                                    frame_num=fk, 
                                    fov_poly=fov_poly, 
                                    gt_in_indices=gt_in_indices, 
                                    dt_in_indices=dt_in_indices)


            # ADD RELATIONS TO VISUALIZE IN WEBLABEL
            if save_openlabel_path is not None:
                for i, j in assignments:
                    gt_obj_uid = gt_uids[i]
                    dt_obj_uid = dt_uids[j]

                    vcd.add_relation_object_object(
                        "Association", 
                        semantic_type=f"Association/{tp}",
                        object_uid_1=gt_obj_uid,
                        object_uid_2=dt_obj_uid,
                        relation_uid=None,
                        ont_uid=None,
                        frame_value=fk,
                        set_mode=core.SetMode.replace)
    
    if _renderer is not None:
        _renderer.close()
    if save_openlabel_path is not None:
        vcd.add_metadata_properties({'eval_config': eval_config})
        vcd.save(save_openlabel_path)
    
    # Save evaluation data
    # if not _debug:
    #     print(f"Saving data in {save_data_path}...")
    #     with open(save_data_path, "wb") as f:
    #         pickle.dump(data, f)
    

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Script for evaluating 3D detections.")
    parser.add_argument('scene_path', type=str, help="Path to the scene. This path must have the 'generated' folder and the 'scene' folder with the 'detections_openlabel.json' file that has the ground truth and annotated detections")
    parser.add_argument('save_path', type=str, help="Output .pkl path")
    
    parser.add_argument('--gen_bev_masks', action='store_true', help="If passed ground truth occupancy-occlusion-driveable masks will be generated in scene folder")
    parser.add_argument('--save_openlabel_path', type=str, default=None, help="If set, the associated OpenLABEL will be saved")
    parser.add_argument('--semantic_types', nargs="+", default=["vehicle.car"], help="List of semantic types to consider")
    parser.add_argument('--ignoring_names', nargs="+", default=["ego_vehicle"], help="List of object names to ignore")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()
    
    check_paths([args.scene_path])

    openlabel_path = os.path.join(args.scene_path, 'scene', 'detections_openlabel.json')
    gt_occ_bev_masks_path = None
    if args.gen_bev_masks:
        gt_occ_bev_masks_path = os.path.join(args.scene_path, 'generated', 'gt_occ_bev_mask')
        if not os.path.exists(gt_occ_bev_masks_path):
            print(f"Folder '{gt_occ_bev_masks_path}' doesn't exist. Creating it now...")
            os.makedirs(gt_occ_bev_masks_path, exist_ok=True)
    
    main(scene_path=args.scene_path,
         openlabel_path=openlabel_path, 
         save_data_path=args.save_path,
         gt_occ_bev_masks_path=gt_occ_bev_masks_path,
         semantic_types=args.semantic_types,
         ignoring_names=args.ignoring_names,
         camera_depth=BEV2SEG_2_Interface.BEV_MAX_DISTANCE,
         max_association_distance=3.0,
         association_dist_type='v2v',
         save_openlabel_path=None, 
         debug=args.debug, 
         debug_ego_model_path=None)
    exit()



    OPENLABEL_PATH      = "./tmp/my_scene/scene/detections_openlabel.json"
    GT_OCC_MASKS_PAHT   = "./tmp/my_scene/generated/gt_occ_bev_mask"
    SAVE_DATA_PATH      = "./data/pipeline_3d_evaluations.pkl"              
    SAVE_OPENLABEL_PATH = None # "./tmp/my_scene/nuscenes_sequence/associated_openlabel.json"
    DEBUG_EGO_MODEL     = "./assets/carlota_3d"
    # DEBUG_EGO_MODEL = "./assets/lowpoly_car_3d"

    semantic_types = [ "vehicle.car" ]
    ignoring_names = [ "ego_vehicle" ]

    main(openlabel_path=OPENLABEL_PATH, 
         save_data_path=SAVE_DATA_PATH,
         gt_occ_bev_masks_path=GT_OCC_MASKS_PAHT,
         semantic_types=semantic_types,
         ignoring_names=ignoring_names,
         camera_depth=15.0,
         max_association_distance=3.0,
         association_dist_type='v2v',
         save_openlabel_path=SAVE_OPENLABEL_PATH, 
         debug=False, 
         debug_ego_model_path=DEBUG_EGO_MODEL)