from depth_pro.depth_pro import DepthProConfig
import open3d as o3d
from vcd import scl
import numpy as np
import depth_pro
import torch


class DepthEstimation():
    def __init__(self, model_path:str, device:torch.DeviceObjType):
        """
        Depth map estimation from monocular image
        """
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
        """
        Create pointcloud from image and depth map
        """
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
