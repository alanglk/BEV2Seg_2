import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from vcd import core, utils, scl, draw
import numpy as np
import copy

from abc import ABC, abstractmethod

import cv2

from typing import Literal, List, Tuple
from my_utils import check_paths, parse_mtl, parse_obj_with_materials, create_cuboid_edges
from bevmap_manager import BEVMapManager
from vehicle_gridmap import VehicleGridMap
from bev2seg_2 import Raw2Seg_BEV

import os

class Settings:
    # Available Shaders
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"
    LINE = "unlitLine"

    def __init__(self):
        self.bg_color = gui.Color(0.185, 0.185, 0.185)
        self.scroll_color = gui.Color(1.0, 0.05, 0.05)
        self.use_ego_model = False
        
        self.working_dir = os.getcwd()

        self.sample_texture = o3d.geometry.Image(cv2.cvtColor(cv2.imread("./assets/checker-map.png"), cv2.COLOR_BGR2RGB))

        self.dt_bbox_color = np.array([102, 201, 102]) / 255.0
        self.dt_bbox_label_color = gui.Color(
            self.dt_bbox_color[0], 
            self.dt_bbox_color[1], 
            self.dt_bbox_color[2])
        
        self.ego_dims = (2.0, 2.0, 2.0) # sx, sy, sz in meters
        self.ego_bbox_color = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.ego_path, self.last_ego_path = None, None
        self.ego_bbox = None
        self.ego_mesh = None
        self.ego_mesh_init = False
        
        self.max_frame = 0
        self.current_frame = 0
        self.camera_height = 45.0
        self.material_line_width = 3


        # Semantic selection
        self.scene_semantics = []
        self.selected_pcd_semantics = []
        self.selected_bbox_semantics = []

        # Occupancy (oy) / Occlusion (on)
        # self.occ_bg_color   = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.oy_color       = np.array([255, 0, 255], dtype=np.uint8) # np.array([1.0, 0.0, 1.0], dtype=np.float64)
        self.on_color       = np.array([255, 255, 0], dtype=np.uint8) # np.array([1.0, 1.0, 0.0], dtype=np.float64)
        self.occ_inst_color = np.array([71, 0, 216]) / 255.0
        self.occ_inst_color = gui.Color(
                                self.occ_inst_color[0], 
                                self.occ_inst_color[1], 
                                self.occ_inst_color[2])


        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord(),
            Settings.LINE:rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        
        self._materials[Settings.LINE].shader = Settings.LINE
        self._materials[Settings.LINE].line_width = self.material_line_width
        
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Default material
        # self.apply_material = True # Clear by AppWindow after updating the scene material
        self.material = self._materials[Settings.LIT]
    
    def set_material(self, name):
        """
        name: Settings.UNLIT, Settings.LIT, Settings.NORMALS, Settings.DEPTH
        """
        assert name in [Settings.UNLIT, Settings.LIT, Settings.NORMALS, Settings.DEPTH]
        self.material = self._materials[name]
        # self.apply_material = True
        
    def load_ego_mesh(self):
        if self.ego_path is None or self.ego_path == self.last_ego_path:
            return
        print("[Settings] Loading EGO model...")
        check_paths([self.ego_path])
        self.last_ego_path = self.ego_path
        self.ego_mesh_init = False
        
        files = os.listdir(self.ego_path)
        mtl_files = []
        obj_files = []
        for f in files:
            if f.endswith(".mtl"):
                mtl_files.append(f)
            if f.endswith(".obj"):
                obj_files.append(f)
        assert len(mtl_files) == len(obj_files)

        if len(mtl_files) == 0:
            print(f"[Settings] No supported 3d mesh files were encountered for Ego Vehicle in {self.ego_path}")
            return None
        if len(mtl_files) > 1:
            print(f"[Settings] Multiple mtl and obj files in {self.ego_path}")
            return None
        
        # Parse mtl file and obj file
        mtl_path = os.path.join(self.ego_path, mtl_files[0])
        obj_path = os.path.join(self.ego_path, obj_files[0])            
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
        self.ego_mesh = o3d.geometry.TriangleMesh()
        self.ego_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.ego_mesh.triangles = o3d.utility.Vector3iVector(faces)
        self.ego_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        self.ego_mesh.compute_vertex_normals()
        
class AppWindow(ABC):
    def __init__(self, 
                 width:int, 
                 height:int, 
                 window_name:str="Open3d AppWindow",
                 ego_path:str = None):
        # ---- AppWindow main settings ------------
        self.width, self.height = width, height
        self.settings = Settings()

        # ---- Create the window ------------------
        self.window = gui.Application.instance.create_window( window_name, width, height )
        w = self.window  # to make the code more concise

        # ---- 3D widget --------------------------
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # ---- Settings panel ---------------------
        # Use font_size for sizing gui elements 
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert( 0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        
        # Frame displayer
        self.frame_label = gui.Label(f"Frame: {self.settings.current_frame}")
        self._settings_panel.add_child(self.frame_label)
        
        # Scene Setting Controller
        scene_ctrls = gui.CollapsableVert("Scene Controls", 0.25 * em, gui.Margins(em, 0, 0, 0))
        
        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        scene_ctrls.add_child(grid)

        self._use_ego_model = gui.Checkbox("Enable EGO model")
        self._use_ego_model.set_on_checked(self._on_use_ego_model)
        self._ego_button = gui.Button("Load EGO model")
        self._ego_button.set_on_clicked(self._on_ego_file_button)
        scene_ctrls.add_fixed(separation_height)
        scene_ctrls.add_child(self._use_ego_model)
        scene_ctrls.add_child(self._ego_button)

        self._settings_panel.add_child(scene_ctrls)

        # Semantic Settings Controller
        semantic_ctrls = gui.CollapsableVert("Semantic Controls", 0.25 * em, gui.Margins(em, 0, 0, 0))
        
        collapsable = gui.CollapsableVert("Pointcloud Labels", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self.pcd_scroller = gui.ScrollableVert(0.25 * em, gui.Margins(em, 0, em, 0))
        self.pcd_scroller.background_color = self.settings.scroll_color
        collapsable.add_child(self.pcd_scroller)
        semantic_ctrls.add_child(collapsable)

        collapsable = gui.CollapsableVert("BBox Labels", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self.bbox_scroller = gui.ScrollableVert(0.25 * em, gui.Margins(em, 0, em, 0))
        self.bbox_scroller.background_color = self.settings.scroll_color
        collapsable.add_child(self.bbox_scroller)
        semantic_ctrls.add_child(collapsable)

        self._settings_panel.add_child(semantic_ctrls)



        # ---- Set Default Settings Values --------
        # In our case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout callback.
        w.set_on_layout(self._on_layout)
        w.set_on_key(self._on_key)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        self._apply_settings()
    
    @abstractmethod
    def update(self, fk:int):
        raise NotImplementedError("Not implemented yet")

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._bg_color.color_value = self.settings.bg_color
        
        if self.settings.use_ego_model and self.settings.ego_path is not None:
            self.settings.load_ego_mesh() # Load model if it is a not loaded model
        self.frame_label.text = f"Frame: {self.settings.current_frame}"

        # Update scene
        try:
            self.update(self.settings.current_frame)
        except:
            pass # Not initialized yet

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
    def _on_key(self, event):
        if event.is_repeat:
            return  # Ignorar repeticiones de teclas
        if event.type == gui.KeyEvent.Type.UP:
            return # Ignore key up events

        if event.key == gui.KeyName.LEFT:
            fk = self.settings.current_frame - 1 # Retroceder un frame
            self.settings.current_frame = fk if fk >= 0 else 0
        elif event.key == gui.KeyName.RIGHT:
            fk = self.settings.current_frame + 1 # Avanzar un frame
            max_fk = self.settings.max_frame
            self.settings.current_frame = fk if fk <= max_fk else max_fk
        self._apply_settings()
    def _on_bg_color(self, new_color: gui.Color):
        self.settings.bg_color = new_color
        self._apply_settings()
    def _on_use_ego_model(self, use):
        self.settings.use_ego_model = use
        self._apply_settings()
    def _on_ego_file_button(self):
        # Open a FileDialog to select Ego Model
        file_dialog = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Load Ego Model", self.window.theme)
        file_dialog.set_path(".") # Current dir
        file_dialog.set_on_done(self._on_ego_path_done)
        file_dialog.set_on_cancel(self._on_ego_path_cancel)
        self.window.show_dialog(file_dialog)
    def _on_ego_path_cancel(self):
        os.chdir(self.settings.working_dir)
        self.window.close_dialog()
    def _on_ego_path_done(self, selected_path):
        if selected_path is not None:
            print(f"[AppWindow] New Ego model path: {selected_path}")
            self.settings.ego_path = selected_path
            self._apply_settings()
        os.chdir(self.settings.working_dir)
        self.window.close_dialog()
    def _add_semantic_selector(self, semantic_label:str):
        """sel_type:Literal["pcd", "bbox"]"""
        em = self.window.theme.font_size

        chk = gui.Checkbox(semantic_label)
        chk.set_on_checked(lambda check: self._update_semantic_selection(semantic_label, "pcd", check))
        self.pcd_scroller.add_child(chk)

        chk = gui.Checkbox(semantic_label)
        chk.set_on_checked(lambda check: self._update_semantic_selection(semantic_label, "bbox", check))
        self.bbox_scroller.add_child(chk)


    def _update_semantic_selection(self, semantic_label:str, sel_type:Literal["pcd", "bbox"], check:bool):
        def upd_sem_sel_type(array:list, check:bool) -> list:
            if check:
                array.append(semantic_label)
                return array
            idx = array.index(semantic_label)
            if idx != -1:
                array.pop(idx)
            return array
        if sel_type == "pcd":
            upd_sem_sel_type(self.settings.selected_pcd_semantics, check)
        elif sel_type == "bbox":
            upd_sem_sel_type(self.settings.selected_bbox_semantics, check)
        
        self._apply_settings()


class DebugBEVMap(AppWindow):
    def __init__(self, 
                 width:int, 
                 height:int,
                 vcd:core.OpenLABEL,
                 BMM:BEVMapManager,
                 camera_name:str='CAM_FRONT',
                 window_name:str="BEVMap Debugger", 
                 ego_path:str=None,
                 bev_params:draw.TopView.Params=None):
        super().__init__(width, height, window_name, ego_path)
        self.vcd = vcd
        self.vcd_scene = scl.Scene(vcd)
        self.camera_name = camera_name
        self.BMM = BMM

        frame_keys = self.vcd.data['openlabel']['frames'].keys()
        self.settings.max_frame = list(frame_keys)[-1]
        self.settings.current_frame = 0

        self.frame_3dlabels = []
        self.frame_acc_pcd = {}     # { 0:{"vehicle.car": o3d.geometry.pcd } }
        
        self.vehicle_gridmap = VehicleGridMap(vcd=self.vcd,
                                              scene=self.vcd_scene, 
                                              bev_params=bev_params, 
                                              grid_frame='vehicle-iso8855',
                                              grid_x_range=(-5.0, 30.0),
                                              grid_y_range=(-15.0, 15.0),
                                              px_size=0.25, 
                                              py_size=0.25, 
                                              z_height=0.1)

        # Create the ego_vehicle representation
        # In settings.ego*
        self.update(self.settings.current_frame)

    def _get_frame_data(self, fk:int):
        # Load the frame data from file
        assert fk >= 0 and fk <= self.settings.max_frame
        frame = self.vcd.get_frame(frame_num=fk)
        frame_properties    = frame['frame_properties']
        raw_image_path      = frame_properties['streams'][self.camera_name]['stream_properties']['uri']
        return self.BMM.load_occ_bev_masks(raw_image_path)

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

        return ego_center_3x1, r_3x3, self.settings.ego_dims
    def _get_ego_representation(self, frame_num:int):
        # If ego representation not initialized
        ego_bbox = self.settings.ego_bbox

        if ego_bbox is None:
            center, rotation, size = self._get_ego_frame_data(frame_num)
            ego_bbox_center = np.array([center[0]- center[0] / 2, center[1] - center[1] / 2, center[2]])
            ego_bbox = o3d.geometry.OrientedBoundingBox(ego_bbox_center, rotation, size)
            ego_mesh = self.settings.ego_mesh
            
            if ego_mesh is not None and not self.settings.ego_mesh_init:
                self.settings.ego_mesh_init = True
                # Initial rotation of ego_mesh    
                # max_bounds = ego_mesh.get_max_bound()
                # min_bounds = ego_mesh.get_min_bound()
                # self.settings.ego_dims = np.abs(max_bounds - min_bounds)
                # Aling ego_mesh with odom frame  
                # r_3x3 = o3d.geometry.get_rotation_matrix_from_xyz([0.0, -np.pi/2, -np.pi/2])
                # ego_mesh.rotate(r_3x3)
                ego_mesh.rotate(rotation)
        
        return ego_bbox, ego_mesh     
    def _update_camera_ego(self, frame_num):
        """Update camera and ego position"""
        line_mat = self.settings._materials[Settings.LINE]
        
        ego_bbox, ego_mesh      = self._get_ego_representation(frame_num)
        ego_center, r_3x3, ego_size    = self._get_ego_frame_data(frame_num)
        #ego_bbox_center = np.array([ego_center[0]- ego_size[0] / 2, ego_center[1] - ego_size[1] / 2, ego_center[2]])
        ego_bbox_center = np.array([ego_center[0], ego_center[1], ego_center[2]])
        ego_bbox.center = ego_bbox_center
        ego_bbox.R = r_3x3
        ego_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(ego_bbox)
        ego_set.paint_uniform_color(self.settings.ego_bbox_color)
        self._scene.scene.add_geometry('ego_set', ego_set, line_mat)


        ego_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=ego_center)
        ego_axis.rotate(r_3x3)
        self._scene.scene.add_geometry('ego_axis', ego_axis, self.settings.material)

        if self.settings.use_ego_model and self.settings.ego_mesh is not None:
            ego_mesh = copy.deepcopy(self.settings.ego_mesh)
            ego_mesh.rotate(r_3x3)
            ego_mesh_center = ego_center 
            ego_mesh_center[1] -= -self.settings.ego_dims[1] / 2
            ego_mesh.translate(ego_mesh_center)
            self._scene.scene.add_geometry('ego_mesh', ego_mesh, self.settings.material)

        # Configurar la cámara para mirar hacia abajo (mirada top-down)
        # bounds = self._scene.scene.bounding_box
        # self._scene.setup_camera(60, bounds, bounds.get_center())
        
        cam_eye = np.array([ego_center[0], ego_center[1], ego_center[2] + self.settings.camera_height]).reshape((3, 1))
        self._scene.look_at(ego_center, cam_eye, [1, 0, 0])
    
    def _get_occ_plane(self, bev_x_range:Tuple[float], bev_y_range:Tuple[float], center:np.ndarray, rotation:np.ndarray=None, texture_image:o3d.geometry.Image=None) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh()       
        vertices = np.array([
            [bev_x_range[0], bev_y_range[1], 0],    # Esquina inferior izquierda
            [bev_x_range[1], bev_y_range[1], 0],    # Esquina inferior derecha
            [bev_x_range[1], bev_y_range[0], 0],    # Esquina superior derecha
            [bev_x_range[0], bev_y_range[0], 0]     # Esquina superior izquierda
        ])
        triangles = np.array([ [0, 1, 2], [2, 3, 0] ])
        uvs = np.array([
            [0, 1], [1, 1], [1, 0], 
            [1, 0], [0, 0], [0, 1]])
        # UV coords
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
        if texture_image is None:
            mesh.paint_uniform_color(np.array([1.0, 0.0, 0.0]))
        else:
            mesh.textures = [texture_image]
        
        mesh.translate(-np.array([0.0, bev_y_range[1]-bev_y_range[0], 0.0]) / 2)
        mesh.translate(center)
        mesh.rotate(rotation, center=center)

        mesh.compute_triangle_normals()
        return mesh

    def _get_accum_pcd(self, cur_pcd:np.ndarray, cur_pcd_colors:np.ndarray, prev_pcd:o3d.geometry.PointCloud, transform_4x4:np.ndarray, lims:tuple = (10, 5, 30),):
        """prev_pcd is in odom frame and current is in camera frame"""
        mask = (abs(cur_pcd[:, 0]) <= lims[0]) & (abs(cur_pcd[:, 1]) <= lims[1]) & (abs(cur_pcd[:, 2]) <= lims[2])
        cur_pcd         = cur_pcd[mask]
        cur_pcd_colors  = cur_pcd_colors[mask]

        cur_points_4xN = utils.add_homogeneous_row( cur_pcd.T )
        cur_points_transformed_4xN = transform_4x4 @ cur_points_4xN
        cur_points_transformed_Nx3 = cur_points_transformed_4xN[:-1, :].T

        # Down-sampling
        cur_pcd = o3d.geometry.PointCloud()
        cur_pcd.points = o3d.utility.Vector3dVector(cur_points_transformed_Nx3)
        cur_pcd.colors = o3d.utility.Vector3dVector(cur_pcd_colors)

        if prev_pcd is None:
            return cur_pcd.voxel_down_sample(0.2)
        else:
            accum_pcd_points = np.vstack((np.asarray(prev_pcd.points), np.asarray(cur_pcd.points)))
            accum_pcd_colors = np.vstack((np.asarray(prev_pcd.colors), np.asarray(cur_pcd.colors)))
        accum_pcd = o3d.geometry.PointCloud()
        accum_pcd.points = o3d.utility.Vector3dVector(accum_pcd_points)
        accum_pcd.colors = o3d.utility.Vector3dVector(accum_pcd_colors)
        return accum_pcd.voxel_down_sample(0.2)
    
    def _get_bboxes(self,
                    semantic_type:str,
                    bboxes_data:List[dict],
                    transform_4x4:np.ndarray,
                    bbox_color: np.ndarray = np.array([102, 201, 102])
                    ) -> List[Tuple[o3d.geometry.LineSet, str, np.ndarray]]:
        """
        [{
            'inst_id': int, 
            'center': (x_pos, y_pos, z_pos), 
            'dimensions': (bbox_width, bbox_height, bbox_depth)  
        }]
        """
        if np.max(bbox_color) > 1.0:
            bbox_color = bbox_color.astype(float) / 255.0

        bboxes = []
        ids = []
        bbox_label_pos = []
        for bbox in bboxes_data:
            center = np.array([bbox['center'][0], bbox['center'][1], bbox['center'][2]])
            # center = center + np.array([bbox['dimensions'][0], 0.0, 0.0]) / 2.0
            bboxes.append( create_cuboid_edges(center, bbox['dimensions'], transform_4x4=transform_4x4, color=bbox_color) )
            ids.append(f"{semantic_type}_{bbox['inst_id']}")

            center_4x1 = utils.add_homogeneous_row(np.array(bbox['center']).reshape(3, -1))
            center_trans_4x1 = transform_4x4 @ center_4x1
            center_3x1 = center_trans_4x1[:3]
            bbox_label_pos.append( center_3x1 )
        return bboxes, ids, bbox_label_pos
    
    def _get_occupancy_occlusion_geometries(self, data:List[dict], fk:int):
        """
        'instance_bev_mask':[{
            'inst_id': int,
            'occupancy_mask': (H, W) binary mask,
            'oclussion_mask': (H, W) binary mask
        }]
        """
        # occupancy: oy
        # occlusion: on
                
        px_size = 0.5 # m
        py_size = 0.5 # m
        pz_size = 0.1 # Extrusion of the visualization plane
                
        # Obtain this from data
        height, width = 1024, 1024 
        bev_aspect_ratio = width / height
        bev_x_range = (-1.0, 30.0) 
        bev_y_range = (-((bev_x_range[1] - bev_x_range[0]) / bev_aspect_ratio) / 2,
                       ((bev_x_range[1] - bev_x_range[0]) / bev_aspect_ratio) / 2)

        # Create EGO grid
        ego_center, r_3x3, _ = self._get_ego_frame_data(fk)
        nx = int(( bev_x_range[1] - bev_x_range[0]) / px_size + 1)
        ny = int(( bev_y_range[1] - bev_y_range[0]) / py_size + 1)

        # Grid colors
        render_mask = np.zeros((height, width, 3), dtype=np.uint8)
        # grid_col = np.zeros((ny, nx, 3), dtype=np.uint8)
        
        # Grid coords
        x = np.linspace(bev_x_range[0], bev_x_range[1], nx)
        y = np.linspace(bev_y_range[1], bev_y_range[0], ny)
        xx, yy = np.meshgrid(x, y)
        
        def get_mask_centroid(mask):
            y_indices, x_indices = np.where(mask == 1)
            # All values are 0
            if len(y_indices) == 0: 
                return None, None
            # Get centroid
            ix = int(np.round(np.mean(x_indices)))
            iy = int(np.round(np.mean(y_indices)))
            return ix, iy

        def transform_idx(ix, iy):
            ix, iy = ix / width * nx, iy / width * ny
            ix, iy = np.round(ix), np.round(iy)
            if isinstance(ix, np.ndarray):
                ix[np.where(ix >= nx)] = nx -1
                iy[np.where(iy >= ny)] = ny -1
                return ix.astype(np.int64), iy.astype(np.int64) 
            ix = ix if ix < nx else nx -1
            iy = iy if iy < ny else ny -1
            return int(ix), int(iy)

        labels_data = [] # (label, x_3d_pos, y_3d_pos)
        for d in data:
            inst_id = d['inst_id']
            oy = d['occupancy_mask']
            on = d['oclussion_mask']
            assert oy.shape[0] == height and on.shape[0] == height
            assert oy.shape[1] == width and on.shape[1] == width
            
            oy_x_idxs, oy_y_idxs = np.where(oy == 1)
            on_x_idxs, on_y_idxs = np.where(on == 1)
            oy_x_idxs, oy_y_idxs = transform_idx(oy_x_idxs, oy_y_idxs) # Index transform
            on_x_idxs, on_y_idxs = transform_idx(on_x_idxs, on_y_idxs) # Index transform

            #
            # grid_col[oy_x_idxs, oy_y_idxs] = self.settings.oy_color
            # grid_col[on_x_idxs, on_y_idxs] = np.where(
            #     np.any(grid_col[on_x_idxs, on_y_idxs] != 0, axis=-1, keepdims=True),
            #     grid_col[on_x_idxs, on_y_idxs],
            #     self.settings.on_color
            # )

            occupancy = (oy[:, :, None] * self.settings.oy_color).astype(np.uint8)
            oclussion = (on[:, :, None] * self.settings.on_color).astype(np.uint8)
            render_mask = render_mask | occupancy | oclussion
            
            # Label 3d position
            ix, iy = get_mask_centroid(oy)
            if ix is None or iy is None:
                continue
            ix, iy = transform_idx(ix, iy) # Index transform
            l_pos =  np.array([xx[iy, ix], yy[iy, ix], 0.5], dtype=np.float32) + ego_center
            labels_data.append((str(inst_id), l_pos))


        # # Compute geometries
        # grid_geometries = []
        # for j in range(ny):
        #     for i in range(nx):
        #         color = grid_col[j, i]
        #         c = np.array([ego_center[0] + xx[j, i], ego_center[1] + yy[j, i], 0.0])
        #         s = np.array([px_size, py_size, pz_size])
        #         bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=s[0], height=s[1], depth=s[2])
        #         bbox_mesh.translate(c - s / 2) 
        #         bbox_mesh.rotate(r_3x3, center=(ego_center[0], ego_center[1], ego_center[2]))  
        #         bbox_mesh.paint_uniform_color(color)
        #         bbox_mesh.compute_vertex_normals()
        #         grid_geometries.append(bbox_mesh)
        # 
        # return grid_geometries

        # Create Textured plane
        # grid_col_uint8 = grid_col # (grid_col * 255).astype(np.uint8)
        # grid_col_uint8 = (np.random.rand(ny, nx, 3) * 255).astype(np.uint8)
        # texture_image = o3d.geometry.Image(grid_col_uint8)
        # texture_image = self.settings.sample_texture
        texture_image = o3d.geometry.Image(render_mask) 
        
        plane = self._get_occ_plane(bev_x_range, bev_x_range, ego_center, r_3x3, texture_image)
        mat = rendering.MaterialRecord()
        mat.shader = Settings.LIT
        mat.albedo_img = texture_image
        return (plane, mat, labels_data)

    def update(self, fk:int):
        """
        fk: frame_num
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
        assert fk >= 0 and fk <= self.settings.max_frame
        
        data = self._get_frame_data(fk)
        print(f"frame: {fk}")

        # Clear scene and 3d labels
        self._scene.scene.clear_geometry() # self._scene: SceneWidget
        for l in self.frame_3dlabels:
            self._scene.remove_3d_label(l)
        self.frame_3dlabels = []

        # Update scene data
        # self._scene.scene.show_axes(True)
        
        print(f"Selected PCD Semantics: {self.settings.selected_pcd_semantics}")
        print(f"Selected BBox Semantics: {self.settings.selected_bbox_semantics}")

        all_pcds                = []
        all_pcd_ids             = []
        all_bboxes              = []
        all_bbox_ids            = []
        bbox_label_positions    = []
        occ_geoms               = [] # [ (semantic_class, plane, texture_material, [labels...])... ]
        aux_pcds = [] # Debuggging

        for semantic_data in data:
            semantic_label = semantic_data['label']
            
            if semantic_label not in self.settings.scene_semantics:
                self.settings.scene_semantics.append(semantic_label)
                self._add_semantic_selector(semantic_label)
                

            cs_src = semantic_data['camera_name']
            transform_4x4, _ = self.vcd_scene.get_transform(cs_src=cs_src, cs_dst="odom", frame_num=fk)

            # Get Accumulated pcd
            if semantic_label in self.settings.selected_pcd_semantics:
                # For accumulating all frames pcds
                if fk not in self.frame_acc_pcd:
                    self.frame_acc_pcd[fk] = {}
                if semantic_label not in self.frame_acc_pcd[fk]:
                    prev_accum_pcd = None
                    if fk-1 in self.frame_acc_pcd and semantic_label in self.frame_acc_pcd[fk-1]:
                        prev_accum_pcd =  self.frame_acc_pcd[fk-1][semantic_label]
                    self.frame_acc_pcd[fk][semantic_label] = self._get_accum_pcd(semantic_data['pcd'], semantic_data['pcd_colors'], prev_accum_pcd, transform_4x4)  
                all_pcds.append(self.frame_acc_pcd[fk][semantic_label])
                all_pcd_ids.append(f"{semantic_label}_pcd_{fk}")

                # # Just save the las accum pcd
                # if not 0 in self.frame_acc_pcd:
                #     self.frame_acc_pcd[0] = {}
                #     self.frame_acc_pcd[0][semantic_label] = None
                # self.frame_acc_pcd[0][semantic_label] = self._get_accum_pcd(semantic_data['pcd'], semantic_data['pcd_colors'], self.frame_acc_pcd[0][semantic_label], transform_4x4)  
                # all_pcds.append(self.frame_acc_pcd[0][semantic_label])
                # all_pcd_ids.append(f"{semantic_label}_pcd")

            # Get 3d BBoxes
            if semantic_label in self.settings.selected_bbox_semantics and 'instance_3dboxes' in semantic_data:
                bboxes, bbox_ids, bbox_label_pos = self._get_bboxes(semantic_label, semantic_data['instance_3dboxes'], transform_4x4, bbox_color=self.settings.dt_bbox_color)
                all_bboxes += bboxes
                all_bbox_ids += bbox_ids
                bbox_label_positions += bbox_label_pos
            


            

            # Get Occupancy/Occlusion masks
            if 'instance_bev_mask' in semantic_data:
                # aux = self._get_occupancy_occlusion_geometries(semantic_data['instance_bev_mask'], fk) # (plane, texture, [labels])
                # occ_geoms.append((semantic_label, aux[0], aux[1], aux[2]))
                # aux_pcd = self.vehicle_gridmap.add_vehicle_grid(frame_num=fk, instance_bev_masks=semantic_data['instance_bev_mask'], camera_name=cs_src)
                # aux_pcds.append(aux_pcd)
                self.vehicle_gridmap.add_vehicle_grid(frame_num=fk, instance_bev_masks=semantic_data['instance_bev_mask'], camera_name=cs_src)


        # draw geometry
        for pcd, pcd_id in zip(all_pcds, all_pcd_ids):
            self._scene.scene.add_geometry(pcd_id, pcd, self.settings.material)
        line_mat = self.settings._materials[Settings.LINE]
        for bbox, bbox_id, bbox_label_pos in zip(all_bboxes, all_bbox_ids, bbox_label_positions):
            self._scene.scene.add_geometry(bbox_id, bbox, line_mat)
            l = self._scene.add_3d_label(bbox_label_pos, bbox_id)
            l.color = self.settings.dt_bbox_label_color
            self.frame_3dlabels.append(l)
        # for semantic_label, plane, mat, inst_labels in occ_geoms:
        #     self._scene.scene.add_geometry(f"occ_{semantic_label}_{fk}", plane, mat)
        #     for label, l_pos in inst_labels:
        #         l = self._scene.add_3d_label(l_pos, label)
        #         l.color = self.settings.occ_inst_color
        #         self.frame_3dlabels.append(l)
        # for i, pcd in enumerate(aux_pcds):
        #     self._scene.scene.add_geometry(f"debuggig_pcds_{i}_{fk}", pcd, self.settings.material)

        occ_pcd = self.vehicle_gridmap.get_pcd_for_debugging(frame_num=fk, oy_color=self.settings.oy_color, on_color=self.settings.on_color)
        self._scene.scene.add_geometry(f"occupancy_occlusion_pcd_{fk}", occ_pcd, self.settings.material)

        # Update camera and ego
        self._update_camera_ego(fk)
        
        
def main(scene_path:str, camera_name:str='CAM_FRONT'):
    # Paths
    scene_openlabel_path    = os.path.join(scene_path, "original_openlabel.json")
    print(f"scene_openlabel_path: {scene_openlabel_path}")
    check_paths([scene_openlabel_path,])

    BMM = BEVMapManager(scene_path=scene_path, gen_flags={'all': False})

    # Load OpenLABEL Scene
    vcd = core.OpenLABEL()
    vcd.load_from_file(scene_openlabel_path)

    # BEV Params
    raw2segmodel_path       = "models/segformer_nu_formatted/raw2segbev_mit-b0_v0.2"
    raw_seg2bev = Raw2Seg_BEV(raw2segmodel_path, None, device=None)
    raw_seg2bev.set_openlabel(vcd)

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.initialize()
    w = DebugBEVMap(width=1024, height=768, vcd=vcd, BMM=BMM, camera_name=camera_name, bev_params=raw_seg2bev.drawer.params)
    gui.Application.instance.run()

if __name__ == "__main__":
    scene_path = "./tmp/my_scene" # args.scene_path
    
    main(scene_path=scene_path, camera_name='CAM_FRONT')


