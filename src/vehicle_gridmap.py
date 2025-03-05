

import numpy as np
from typing import Dict, List, Tuple
from vcd import core, scl, draw, utils
import open3d as o3d
from scipy.spatial import KDTree
import cv2

class FrameGridMap:
    def __init__(self, scene:scl.Scene, grid_x_range:Tuple[float], grid_y_range:Tuple[float], frame_num:int, bev_params:draw.TopView.Params, px_size:float=0.25, py_size:float=0.25, z_height:float=0.0):
        """
        Parameters
        --------------
        - px_size: grid cell x size
        - py_size: grid cell y size
        """ 
        self.scene = scene
        self.frame_num  = frame_num
        self.bev_params = bev_params
        
        nx = int(( grid_x_range[1] - grid_x_range[0]) / px_size + 1)
        ny = int(( grid_y_range[1] - grid_y_range[0]) / py_size + 1)

        x = np.linspace(grid_x_range[0], grid_x_range[1], nx)
        y = np.linspace(grid_y_range[0], grid_y_range[1], ny)
        self.z = np.zeros((ny, nx)) + z_height
        self.xx, self.yy = np.meshgrid(x, y)

        # For finding closest index
        grid_coords = np.column_stack((self.yy.ravel(), self.xx.ravel())) 
        self.kdtree = KDTree(grid_coords)

        self.oy_grid = np.zeros((ny, nx)) - 1 # Occupancy grid information
        self.on_grid = np.zeros((ny, nx)) - 1 # Occlusion grid information

        # self.oy_grid[np.zeros(nx, dtype=np.uint8).tolist(), list(range(nx))] = 1
        # self.oy_grid[(np.zeros(nx, dtype=np.uint8) + ny-1).tolist(), list(range(nx))] = 1

    def _get_grid_idx_from_mask(self, mask:np.ndarray):
        mask_y_idxs, mask_x_idxs = np.where(mask == 1) # FOKIN LINEA DE MIERDA
        if len(mask_x_idxs) == 0 or len(mask_y_idxs) == 0:
            return [], [], None

        pix_2d_coords_3xN   = utils.add_homogeneous_row(np.array([mask_x_idxs, mask_y_idxs]))
        pix_3d_coords_3xN   = utils.inv(self.bev_params.S).dot(pix_2d_coords_3xN) # 3xN

        # Find closes indexes
        x_coords = pix_3d_coords_3xN[0, :]
        y_coords = pix_3d_coords_3xN[1, :]
        _, nearest_idx = self.kdtree.query(np.column_stack((y_coords, x_coords)))
        idx_y, idx_x = np.unravel_index(nearest_idx, self.xx.shape)
        
        return idx_x, idx_y, pix_3d_coords_3xN

    def add_oy_info(self, inst_id:int, mask:np.ndarray, cs_src='CAM_FRONT', cs_dst='vehicle-iso8855'):
        """Add Occupancy information to grid
        Parameters
        ----------
        inst_id: object id
        mask: (H, W) binary mask
        """
        idx_x, idx_y, pix_3d_points_3xN = self._get_grid_idx_from_mask(mask)
        self.oy_grid[idx_y, idx_x] = inst_id
        return pix_3d_points_3xN # Debugging
    
    def add_on_info(self, inst_id:int, mask:np.ndarray, cs_src='CAM_FRONT', cs_dst='vehicle-iso8855'):
        """Add Occupancy information to grid
        Parameters
        ----------
        inst_id: object id
        mask: (H, W) binary mask
        """
        idx_x, idx_y, _ = self._get_grid_idx_from_mask(mask)
        self.on_grid[idx_y, idx_x] = inst_id


    def get_pcd_for_debugging(self, make_transform:bool=True, transform_4x4:np.ndarray=None, oy_color:np.ndarray=np.zeros((3, 1)), on_color:np.ndarray=np.zeros((3, 1))) -> o3d.geometry.PointCloud:
        if np.max(oy_color) > 1.0:
            oy_color = oy_color.astype(float) / 255.0
        if np.max(on_color) > 1.0:
            on_color = on_color.astype(float) / 255.0

        oy_idx_y, oy_idx_x = np.where(self.oy_grid >= 0)
        on_idx_y, on_idx_x = np.where(self.on_grid >= 0)

        oy_N = len(oy_idx_x)
        on_N = len(on_idx_x)

        oy_points_3xN = np.array([ self.xx[oy_idx_y, oy_idx_x], self.yy[oy_idx_y, oy_idx_x], self.z[oy_idx_y, oy_idx_x] ]) 
        on_points_3xN = np.array([ self.xx[on_idx_y, on_idx_x], self.yy[on_idx_y, on_idx_x], self.z[on_idx_y, on_idx_x] ]) 
        
        if make_transform:

            oy_points_4xN = utils.add_homogeneous_row(oy_points_3xN)
            oy_points_4xN = transform_4x4 @ oy_points_4xN
            oy_points_3xN = oy_points_4xN[:3, :]

            on_points_4xN = utils.add_homogeneous_row(on_points_3xN)
            on_points_4xN = transform_4x4 @ on_points_4xN
            on_points_3xN = on_points_4xN[:3, :]

        oy_colors = np.tile(oy_color, (oy_N, 1)) 
        on_colors = np.tile(on_color, (on_N, 1)) 
        
        
        pcd = o3d.geometry.PointCloud()
        if oy_N == 0 and on_N == 0:
            pass
        elif oy_N == 0:
            pcd.points = o3d.utility.Vector3dVector(on_points_3xN.T)
            pcd.colors = o3d.utility.Vector3dVector(on_colors)
        elif on_N == 0:
            pcd.points = o3d.utility.Vector3dVector(oy_points_3xN.T)
            pcd.colors = o3d.utility.Vector3dVector(oy_colors)
        else:
            all_points = np.vstack([oy_points_3xN.T, on_points_3xN.T])
            all_colors = np.vstack([oy_colors, on_colors])
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(all_colors)

        print(f"Debugging pcd shape: {np.asarray(pcd.points).shape}")        
        return pcd

class VehicleGridMap:
    def __init__(self, 
                 vcd:core.OpenLABEL,
                 scene:scl.Scene, 
                 bev_params:draw.TopView.Params, 
                 grid_frame:str='vehicle-iso8855',
                 grid_x_range:Tuple[float] = (-15, 15),
                 grid_y_range:Tuple[float] = (-15, 15),
                 px_size:float=0.25, 
                 py_size:float=0.25,
                 z_height:float=0.0):
        """
        Parameters
        --------------
        - px_size: grid cell x size
        - py_size: grid cell y size
        """
        self.vcd = vcd
        self.scene = scene
        self.bev_params = bev_params
        self.grid_frame = grid_frame

        self.px_size    = px_size
        self.py_size    = py_size
        self.z_height   = z_height

        self.grid_x_range = grid_x_range
        self.grid_y_range = grid_y_range

        self.frame_grids = {} # Dict[int, FrameGridMap]

    def get_frame_grid(self, frame_num:int) -> FrameGridMap:
        if frame_num not in self.frame_grids:
            self.frame_grids[frame_num] = FrameGridMap(self.scene, self.grid_x_range, self.grid_y_range, frame_num, self.bev_params, self.px_size, self.py_size, self.z_height)
        return self.frame_grids[frame_num]
    
    def add_vehicle_grid(self, instance_bev_masks:List[dict], frame_num:int, camera_name='CAM_FRONT'):
        """
        Parameters
        ----------
        - frame_num
        - instance_bev_masks: 
            ```
            [{
                'inst_id': int,
                'occupancy_mask': (H, W) binary mask,
                'oclussion_mask': (H, W) binary mask
            }]
            ```
        """
        all_points = np.empty((3, 0))
        for d in instance_bev_masks:
            inst_id = d['inst_id']
            oy = d['occupancy_mask']
            on = d['oclussion_mask']
            frame_grid = self.get_frame_grid(frame_num)
            pix_3d_points_3xN = frame_grid.add_oy_info(inst_id, oy, cs_src=camera_name, cs_dst=self.grid_frame)
            frame_grid.add_on_info(inst_id, on, cs_src=camera_name, cs_dst=self.grid_frame)
            if pix_3d_points_3xN is not None:
                all_points = np.hstack([all_points, pix_3d_points_3xN]) # 3xN
        
        # TODO: Borrar esto
        pcd = o3d.geometry.PointCloud()
        all_colors = np.tile(np.array([1.0, 0.0, 0.0]), (all_points.shape[1], 1)) 

        frame_properties = self.vcd.get_frame(frame_num=frame_num)['frame_properties']
        frame_odometry = frame_properties['transforms']['vehicle-iso8855_to_odom']['odometry_xyzypr']
        t_vec =  frame_odometry[:3]
        ypr = frame_odometry[3:]
        r_3x3 = utils.euler2R(ypr)
        transform_4x4 = utils.create_pose(R=r_3x3, C=np.array([t_vec]).reshape(3, 1))

        all_points_4xN = utils.add_homogeneous_row(all_points)
        all_points_4xN = transform_4x4 @ all_points_4xN
        all_points = all_points_4xN[:3, :].T

        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        return pcd


    def get_pcd_for_debugging(self, frame_num:int, oy_color:np.ndarray=None, on_color:np.ndarray=None) -> o3d.geometry.PointCloud:
        frame_grid = self.get_frame_grid(frame_num)
        oy_color = self.oy_color if oy_color is None else oy_color
        on_color = self.on_color if on_color is None else on_color
        
        frame_properties = self.vcd.get_frame(frame_num=frame_num)['frame_properties']
        frame_odometry = frame_properties['transforms']['vehicle-iso8855_to_odom']['odometry_xyzypr']
        t_vec =  frame_odometry[:3]
        ypr = frame_odometry[3:]
        r_3x3 = utils.euler2R(ypr)
        transform_4x4 = utils.create_pose(R=r_3x3, C=np.array([t_vec]).reshape(3, 1))
        
        return frame_grid.get_pcd_for_debugging(make_transform=True, transform_4x4=transform_4x4, oy_color=oy_color, on_color=on_color)



