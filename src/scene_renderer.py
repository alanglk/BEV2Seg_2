import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from vcd import core, utils, scl
import numpy as np
import copy


from typing import Literal
from my_utils import check_paths, parse_mtl, parse_obj_with_materials
from bevmap_manager import BEVMapManager
import os

class Settings:
    # Available Shaders
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"
    
    def __init__(self):
        self.bg_color = gui.Color(0.185, 0.185, 0.185)
        self.use_ego_model = False
        
        self.ego_dims = (2.0, 2.0, 2.0) # sx, sy, sz in meters
        self.ego_bbox_color = (0.0, 1.0, 0.0)
        self.ego_path, self.last_ego_path = None, None
        self.ego_bbox = None
        self.ego_mesh = None
        
        self.max_frame = 0
        self.current_frame = 0
        self.camera_height = 5.0

        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        
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
        
class AppWindow:
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

        self.frame_label = gui.Label(f"Frame: {self.settings.current_frame}")
        scene_ctrls.add_fixed(separation_height)
        scene_ctrls.add_child(self.frame_label)

        self._settings_panel.add_child(scene_ctrls)

        # ---- Set Default Settings Values --------
        # In our case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout callback.
        w.set_on_layout(self._on_layout)
        w.set_on_key(self._on_key)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        self._apply_settings()
        
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
            self.update(self.settings.current_frame)
        elif event.key == gui.KeyName.RIGHT:
            fk = self.settings.current_frame + 1 # Avanzar un frame
            max_fk = self.settings.max_frame
            self.settings.current_frame = fk if fk <= max_fk else max_fk
            self.update(self.settings.current_frame)
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
        self.window.close_dialog()
    def _on_ego_path_done(self, selected_path):
        if selected_path is not None:
            print(f"[AppWindow] New Ego model path: {selected_path}")
            self.settings.ego_path = selected_path
            self._apply_settings()
        self.window.close_dialog()

class DebugBEVMap(AppWindow):
    def __init__(self, 
                 width:int, 
                 height:int,
                 vcd:core.OpenLABEL,
                 BMM:BEVMapManager,
                 camera_name:str='CAM_FRONT',
                 window_name:str="BEVMap Debugger", 
                 ego_path:str=None):
        super().__init__(width, height, window_name, ego_path)
        self.vcd = vcd
        self.vcd_scene = scl.Scene(vcd)
        self.camera_name = camera_name
        self.BMM = BMM

        frame_keys = self.vcd.data['openlabel']['frames'].keys()
        self.settings.max_frame = list(frame_keys)[-1]
        self.settings.current_frame = 0

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
            ego_bbox = o3d.geometry.OrientedBoundingBox(center, rotation, size)
            ego_mesh = self.settings.ego_mesh
            if ego_mesh is not None:
                # Initial rotation of ego_mesh    
                max_bounds = ego_mesh.get_max_bound()
                min_bounds = ego_mesh.get_min_bound()
                self.settings.ego_dims = np.abs(max_bounds - min_bounds)
                # Aling ego_mesh with odom frame  
                r_3x3 = o3d.geometry.get_rotation_matrix_from_xyz([0.0, -np.pi/2, -np.pi/2])
                ego_mesh.rotate(r_3x3) 
        
        return ego_bbox, ego_mesh     
    
    def _update_camera_ego(self, frame_num):
        """Update camera and ego position"""
        ego_bbox, ego_mesh      = self._get_ego_representation(frame_num)
        ego_center, r_3x3, _    = self._get_ego_frame_data(frame_num)
        ego_bbox.center = ego_center
        ego_bbox.R = r_3x3
        ego_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(ego_bbox)
        ego_set.paint_uniform_color(self.settings.ego_bbox_color)


        if self.settings.use_ego_model and self.settings.ego_mesh is not None:
            ego_mesh = copy.deepcopy(self.settings.ego_mesh)
            ego_mesh.rotate(r_3x3)
            ego_mesh_center = ego_center 
            ego_mesh_center[1] -= -self.ego_sizes[1] / 2
            ego_mesh.translate(ego_mesh_center)
            self._scene.scene.add_geometry('ego_mesh', ego_mesh, self.settings.material)

        self._scene.scene.add_geometry('ego_set', ego_set, self.settings.material)

        # Configurar la cámara para mirar hacia abajo (mirada top-down)
        # bounds = self._scene.scene.bounding_box
        # self._scene.setup_camera(60, bounds, bounds.get_center())
        
        cam_eye = np.array([ego_center[0], ego_center[1], ego_center[2] - self.settings.camera_height]).reshape((3, 1))
        self._scene.look_at(ego_center, cam_eye, [0, -1, 0])

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

        # self._scene: SceneWidget
        self._scene.scene.clear_geometry()
        self._scene.scene.show_axes(True)

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

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.initialize()
    w = DebugBEVMap(width=1024, height=768, vcd=vcd, BMM=BMM, camera_name=camera_name)
    gui.Application.instance.run()

if __name__ == "__main__":
    scene_path = "./tmp/my_scene" # args.scene_path
    
    main(scene_path=scene_path, camera_name='CAM_FRONT')


