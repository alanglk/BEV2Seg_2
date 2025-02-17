from vcd import core, scl, types, utils
from scipy.optimize import linear_sum_assignment
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
import textwrap
import threading
import time

from utils import get_pallete

from tqdm import tqdm
from typing import TypedDict, Dict, List, Tuple
import argparse
import sys
import os


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
    
    # Check if the selected types are present in the ground_truth and annotations
    for tp in selected_types:
        if tp not in gt_objs['objects']:
            raise Exception(f"GT has no type {tp}")
        if tp not in dt_objs['objects']:
            raise Exception(f"DT (custom annotations) has no type {tp}")

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
    return gt_objs, dt_objs


def compute_cost_between_bboxes(bbox1: Tuple[float], bbox2: Tuple[float]) -> float:
    """
    bboxes on the same coords system.
    bbox1: [ x, y, z, rx, ry, rz, sx, sy, sz ]
    bbox2: [ x, y, z, rx, ry, rz, sx, sy, sz ]
    """
    x1, y1, z1, rx1, ry1, rz1, sx1, sy1, sz1 = bbox1
    x2, y2, z2, rx2, ry2, rz2, sx2, sy2, sz2 = bbox2

    dist_2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2) 
    dist = np.sqrt(dist_2)
    return dist

def get_cost_matrix(gt_bboxes:List[dict], dt_bboxes:List[dict], scene:scl.Scene, frame_num:int) -> np.ndarray:
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

            cost_matrix[i][j] = compute_cost_between_bboxes(bbox1, bbox2)

    return cost_matrix

def assign_detections_to_ground_truth(cost_matrix:np.ndarray) -> List[int]:
    """
    Cost matrix (N, M) where N is the number of GT elements and M is the
    number of detected elements (custom annotations). Returns the list of assignments
    """
    n, m = cost_matrix.shape
    
    # If it is not a square matrix, fill missing values with infinity
    max_dim = max(n, m)
    padded_cost_matrix = np.full((max_dim, max_dim), np.max(cost_matrix)+1)  # Valor alto
    padded_cost_matrix[:n, :m] = cost_matrix  # Insertamos la matriz original

    # Solve LSAP
    row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)

    # Filtrar asignaciones inválidas (aquellas que caen en la parte añadida)
    valid_assignments = [(r, c) for r, c in zip(row_ind, col_ind) if r < n and c < m]

    assert len(valid_assignments) == min(n, m)
    return valid_assignments


# ===========================================================================================
#                                      DEBUG FUNCTIONS                                      =
# =========================================================================================== 
_stop_loading_event = None
_loading_thread = None
_renderer = None

def _loading_animation(stop_event):
    """'Loading...' animation in another thread"""
    chars = ['|', '/', '-', '\\']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rLoading {chars[i % len(chars)]} ")
        sys.stdout.flush()
        time.sleep(0.2)
        i += 1
    sys.stdout.write("\rLoading... Done! ✅\n") # Final msg
def init_loading_print():
    global _stop_loading_event, _loading_thread  # Declarar las variables globales
    _stop_loading_event = threading.Event()  # Crear el evento
    _loading_thread = threading.Thread(target=_loading_animation, args=(_stop_loading_event,))
    _loading_thread.start()
def finish_loading_print():
    global _stop_loading_event, _loading_thread  
    if _stop_loading_event:
        _stop_loading_event.set()  
    if _loading_thread:
        _loading_thread.join()  

def debug_show_cost_matrix(cost_matrix: np.ndarray, assignments:List[Tuple[int]], gt_labels: List[str], dt_labels: List[str], semantic_type:str, frame_num:int):
    plt.figure(figsize=(10, 8))
    wrapped_gt_labels = ["\n".join(textwrap.wrap(label, width=15)) for label in gt_labels]
    
    ax = sns.heatmap(
        cost_matrix,
        annot=True,  # show values
        fmt=".2f",  # value format
        cmap="coolwarm",  # color pallete
        linewidths=0.5,  # lines between cells
        xticklabels=dt_labels,
        yticklabels=wrapped_gt_labels,
        cbar_kws={'label': 'Cost'},
        annot_kws={"size": 8}
    )
    ax.set_xticklabels(dt_labels, fontsize=8)
    ax.set_yticklabels(wrapped_gt_labels, fontsize=8, rotation=0, ha="right")

    # Mark assingments
    for (i, j) in assignments:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
    plt.ylabel("Ground Truth (gt)")
    plt.xlabel("Detections (dt)")
    plt.title(f"Cost Matrix for type: {semantic_type} in frame {frame_num}")
    plt.show()

def debug_load_accum_pcd(vcd:core.OpenLABEL, base_path:str):
    """Return the scene accumulated pointcloud on 'odom' frame"""
    lidar_stream = vcd.get_stream("LIDAR_TOP")
    pcd_path = os.path.join(base_path, lidar_stream['uri'])
    check_paths([pcd_path])
    return o3d.io.read_point_cloud(pcd_path) # on 'odom' frame

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class Debug3DRenderer:
    def __init__(self, 
                 vcd:core.OpenLABEL, 
                 scene:scl.Scene, 
                 base_path:str, 
                 frame_num:int,
                 voxel_size:float=0.2, 
                 background_color:Tuple[float]=[0.185, 0.185, 0.185]):
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
        init_loading_print()
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
        self.ego_color = [0.0, 1.0, 0.0]

        # Set the current frame indicator for rendering the same semantic types of the same frame
        self.current_frame = frame_num
        self.space_pressed = False
        
    def _space_callback(self, vis, key, action):
        print(f"[Debug3DRenderer] Key: {key}, Action: {action}")
        self.space_pressed = True

    def _get_line_material(self) -> rendering.MaterialRecord:
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        return mat
    def _get_pcd_material(self) -> rendering.MaterialRecord:
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        return mat

    def _get_ego_frame_data(self, frame_num:int):
        """Return the ego vehicle data in odom frame:
             ego_center_3x1, rotation_3x3, size
        """
        frame_properties = self.vcd.get_frame(frame_num=frame_num)['frame_properties']
        frame_odometry = frame_properties['transforms']['vehicle-iso8855_to_odom']['odometry_xyzypr']
        t_vec =  frame_odometry[:3]
        ypr = frame_odometry[3:]
        r_3x3 = utils.euler2R(ypr)
        transform_4x4 = utils.create_pose(R=r_3x3, C=np.array([t_vec]).reshape(3, 1))

        ego_center_3x1 = np.zeros((3, 1))
        ego_center_4x1 = utils.add_homogeneous_row(ego_center_3x1)
        ego_center_3x1 = (transform_4x4 @ ego_center_4x1)[:3].ravel()

        return ego_center_3x1, r_3x3, [2.0, 2.0, 2.0]

    def _get_text_mesh(self,
                       text:str, 
                       position:np.ndarray,
                       size:float=0.1, 
                       color:Tuple[float]=(1, 0, 0)
                       ) ->  o3d.geometry.TriangleMesh:
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

    def _get_associated_geometries(self, 
                                   gt_uids:List[str], 
                                   dt_uids:List[str], 
                                   assignments:List[Tuple[int]], 
                                   frame_num:int, 
                                   arrow_cone_height:float=0.2, 
                                   arrow_cylinder_radius:float=0.05,
                                   arrow_color:Tuple[float]=(1, 0, 0)
                                   ) -> List[o3d.geometry.LineSet]:
        
        colors = get_pallete(len(assignments)) / 255.0
        geometries = []
        for index, (i, j) in enumerate(assignments):
            obj_uid_1 = gt_uids[i]
            obj_uid_2 = dt_uids[j]
            colors[index]

            # TODO: box3D -> bbox3D
            bbox1 = self.vcd.get_object_data(obj_uid_1,'bbox3D', frame_num=frame_num) # GT
            bbox2 = self.vcd.get_object_data(obj_uid_2,'box3D', frame_num=frame_num) # DT
            if "val" not in bbox1:
                print(f"[Debug3DRenderer] Missing bbox3D values for bbox1 with UID {obj_uid_1}")
            if "val" not in bbox2:
                print(f"[Debug3DRenderer] Missing bbox3D values for bbox2 with UID {obj_uid_2}")

            x1, y1, z1, rx1, ry1, rz1, sx1, sy1, sz1 = bbox1['val']
            x2, y2, z2, rx2, ry2, rz2, sx2, sy2, sz2 = bbox2['val']
            
            center1 = [x1, y1, z1] 
            center2 = [x2, y2, z2]

            size1 = [sx1, sy1, sz1]
            size2 = [sx2, sy2, sz2]
            
            rotation1 = o3d.geometry.get_rotation_matrix_from_xyz([rx1, ry1, rz1])  
            rotation2 = o3d.geometry.get_rotation_matrix_from_xyz([rx2, ry2, rz2])  
            
            obx1 = o3d.geometry.OrientedBoundingBox(center1, rotation1, size1)
            obx2 = o3d.geometry.OrientedBoundingBox(center2, rotation2, size2)
            
            obx1 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obx1)
            obx2 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obx2)
            obx1.paint_uniform_color(colors[index]) # GT Color
            obx2.paint_uniform_color(colors[index]) # DT Color

            # Add uids text labels to bboxes
            text1 = self._get_text_mesh(str(obj_uid_1), np.array([x1, y1, z1]))
            text2 = self._get_text_mesh(str(obj_uid_2), np.array([x2, y2, z2]))
            
            # Compute arrow between GT and DT
            arrow = self._get_arrow_mesh(np.array([x1, y1, z1]), np.array([x2, y2, z2]))

            # Add geometries
            geometries.append(obx1)
            geometries.append(obx2)
            geometries.append(text1)
            geometries.append(text2)
            geometries.append(arrow)
        return geometries
    
    def _update_camera_ego(self, frame_num):
        """Update camera and ego position"""
        ego_center, r_3x3, _ = self._get_ego_frame_data(frame_num)
        self.ego_vehicle_bbox.center = ego_center
        self.ego_vehicle_bbox.R = r_3x3
        ego_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(self.ego_vehicle_bbox)
        ego_set.paint_uniform_color(self.ego_color)
        self.vis.add_geometry(ego_set)

        # Configurar la cámara para mirar hacia abajo (mirada top-down)
        self.ctr.set_zoom(0.2)  
        self.ctr.set_front([0, 0, 1])       # Top-view
        self.ctr.set_lookat(ego_center)     # Apuntar al centro del vehículo (ego)
        self.ctr.set_up([0, -1, 0])         # Alinear eje Y como el 'arriba' de la cámara

    def update(self, gt_uids:List[str], dt_uids:List[str], assignments:List[Tuple[int]], semantic_type:List[str], frame_num:int):

        """
        Update scene with frame cuboids associations and move the camera setting ego_vehicle position at the window center

        Args:
            gt_uids: Ground Truth objects uids of semantic_type detected on current frame
            dt_uids: Detected objects uids of semantic_type detected on current frame
            assignments: associations between gt and dt
            frame_num (int): current frame
        """

        # Remove prev geometries except the pointcloud if we have changed the
        # frame index
        if self.current_frame != frame_num:
            self.vis.clear_geometries()
            self.vis.add_geometry(self.accum_pcd)
            self.current_frame = frame_num
            self.space_pressed = False

        # De momento semantic_types no lo utilizo porque solo trabajo con vehículos.
        # la idea es diferenciar las clases semánticas por color
        geometries = self._get_associated_geometries(gt_uids, dt_uids, assignments, frame_num)
        for g in geometries:
            self.vis.add_geometry(g)

        # Upate camera and ego
        self._update_camera_ego(frame_num)

        # Refrescar visualización
        print("[Debug3DRenderer] Press Space to continue to next frame")
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.space_pressed:
                break

    def close(self):
        """Cierra la ventana de visualización."""
        self.vis.destroy_window()


# ===========================================================================================
#                                            MAIN                                           =
# =========================================================================================== 
def main(
        openlabel_path:str,
        semantic_types: List[str],
        ignoring_names: List[str],
        save_path:str = None,
        debug:bool=False
        ):
    # Check wheter the file exists
    check_paths([openlabel_path])
    scene_path = os.path.abspath(os.path.dirname(openlabel_path))

    # Load OpenLABEL
    vcd = core.OpenLABEL()
    vcd.load_from_file(openlabel_path)
    scene = scl.Scene(vcd)

    # Get the selected ground_truth and annotated objects 
    all_objs_inf = vcd.get_objects()
    gt_objs_inf, dt_objs_inf = get_gt_dt_inf(all_objs_inf, selected_types=semantic_types, ignoring_names=ignoring_names, filter_out=True)

    global _renderer
    frame_keys = vcd.data['openlabel']['frames'].keys()
    for fk in tqdm(frame_keys, desc="frames"):
        gt_uids_in_frame = gt_objs_inf['frame_presence'][fk] # Ground_truth of frame
        dt_uids_in_frame = dt_objs_inf['frame_presence'][fk] # Detections of frame
        
        for tp in semantic_types:
            if tp not in gt_uids_in_frame or tp not in dt_uids_in_frame:
                continue

            gt_uids = gt_uids_in_frame[tp]
            dt_uids = dt_uids_in_frame[tp]
            
            # cambiar box3d a bbox3d
            gt_objs_data = [ vcd.get_object_data(uid=uid, data_name='bbox3D', frame_num=fk) for uid in gt_uids]
            dt_objs_data = [ vcd.get_object_data(uid=uid, data_name='box3D', frame_num=fk) for uid in dt_uids]
            
            if len(gt_objs_data) == 0 or len(dt_objs_data) == 0:
                continue # Skip

            cost_matrix = get_cost_matrix(gt_objs_data, dt_objs_data, scene=scene, frame_num=fk)
            assignments = assign_detections_to_ground_truth(cost_matrix)
            print(cost_matrix)
            print(assignments)
            
            if debug:
                gt_labels = [gt_objs_inf['objects'][tp][uid]['name'] for uid in gt_uids]
                dt_labels = [dt_objs_inf['objects'][tp][uid]['name'] for uid in dt_uids]
                # debug_show_cost_matrix(cost_matrix, assignments, gt_labels, dt_labels, tp, fk)
                
                # Render scene
                if _renderer is None:
                    _renderer = Debug3DRenderer(vcd, scene, base_path=scene_path, frame_num=fk)
                _renderer.update(gt_uids, dt_uids, assignments, semantic_type=tp, frame_num=fk)
            

            # ADD RELATIONS TO VISUALIZE IN WEBLABEL
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

            # print(f"frame: {fk}, semantic-type: {tp} first GT object data:")
            # uid_0 = gt_uids[0]
            # name_0 = gt_objs_inf['objects'][tp][uid_0]['name']
            # print(f"uid_0: {uid_0} name_0: {name_0}")
    
    if _renderer is not None:
        _renderer.close()
    vcd.save(save_path)

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Script for evaluating 3D detections.")
    # parser.add_argument('openlabel_path', type=str, help="Path to the openlabel with the ground truth and annotated detections")

    parser.add_argument('--semantic_types', nargs="+", default=["vehicle.car"], help="List of semantic types to consider")
    parser.add_argument('--ignoring_names', nargs="+", default=["ego_vehicle"], help="List of object names to ignore")
    parser.add_argument('--save_path', type=str, default=None, help="If set, the associated openlabel will be saved")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    # Check provided paths
    if args.save_path is not None:
        check_paths([args.openlabel_path, args.save_path])
    elif 'openlabel_path' in args._get_args():
        check_paths([args.openlabel_path])

    OPENLABEL_PATH = "./tmp/my_scene/nuscenes_sequence/annotated_openlabel.json"
    SAVING_PATH = "./tmp/my_scene/nuscenes_sequence/associated_openlabel.json"
    semantic_types = [ "vehicle.car" ]
    ignoring_names = [ "ego_vehicle" ]

    main(openlabel_path=OPENLABEL_PATH, semantic_types=semantic_types,ignoring_names=ignoring_names, save_path=SAVING_PATH, debug=True)