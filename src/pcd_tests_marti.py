import os
import re
import argparse
import YOLOPv2.run
import depth_pro
import torch
import numpy as np
import cv2 as cv
import open3d as o3d
import matplotlib.pyplot as plt
import random
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.mixture import GaussianMixture

from vcd import core, draw, utils, scl
from tqdm import tqdm
from PIL import Image

OPENLABEL_FILE = "openlabel.json"
FOLDER_NAME = "test"
CAM_NAME = "CAM_FRONT"
CS_REF = "odom"

def validate_paths(paths_list):
    for directory in paths_list:
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist. Creating it now...")
            os.makedirs(directory)
def load_imgs_dataset(path):
    """Loads the image dataset from the specified directory."""
    return [file for file in os.listdir(path) 
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

class DepthEstimation:
    def __init__(self, data_folder, run_inference = True) -> None:
        self.data_folder = data_folder
        self.run_inference = run_inference
        
        self.images_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/images")
        self.output_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/depths")

        validate_paths(paths_list=[self.images_path, self.output_path])

        self.images_dataset = load_imgs_dataset(path=self.images_path)
        
    def run(self):
        if not self.run_inference:
            print(f"Run inference is set to False. Loading depth maps from caché. Folder: {self.output_path}")
            return
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")
        
        model, transform = depth_pro.create_model_and_transforms(
                                device=device, precision=torch.half,
                            )
        model.eval()

        for image_name in tqdm(self.images_dataset, desc="Inferring images' depth", unit="image"):
            image_path = os.path.join(self.images_path, image_name)
            img_name_clean = re.sub(r'^\.jpg$', '', image_name)[:-4]

            # Load and preprocess an image.
            image, _, _ = depth_pro.load_rgb(image_path)

            # Run inference.
            prediction = model.infer(transform(image))
            depth_image = prediction["depth"]
            #focal_length = prediction["focallength_px"] # estimated focal length

            # Convert the depth tensor to a NumPy array
            depth_image_np = depth_image.detach().cpu().numpy()
            depth_image_pil = Image.fromarray(depth_image_np.astype(np.float32), mode='F')
            depth_image_pil.save(f"{self.output_path}/dmap_{img_name_clean}.tiff")

class ScenePCD:
    def __init__(self, data_folder, run_inference = True) -> None:
        self.data_folder = data_folder
        self.run_inference = run_inference
        
        self.images_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/images")
        self.openlabel_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/openlabel")

        self.depths_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/depths")
        self.output_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/pcds/raw")

        validate_paths(paths_list=[self.images_path, self.depths_path, self.output_path, self.openlabel_path])
        
        self.vcd = core.OpenLABEL()
        self.vcd.load_from_file(os.path.join(self.openlabel_path, f"{OPENLABEL_FILE}"))
        self.scene = scl.Scene(vcd = self.vcd)
        self.camera = self.scene.get_camera(camera_name=CAM_NAME)

        self.images_dataset = load_imgs_dataset(path=self.images_path)
        self.dmap_dataset = load_imgs_dataset(path=self.depths_path)
    
    def run(self):
        if not self.run_inference:
            print(f"Run PseudoPCDs generation is set to False. Loading PCDs from caché. Folder: {self.output_path}")
            return
        
        total_frames = self.vcd.data["openlabel"]["frame_intervals"][0]["frame_end"]
        
        for frame_id in tqdm(range(0, total_frames + 1), desc="Depth Maps to PseudoPCDs", unit="dmap"):
            frame_data = self.vcd.get_frame(frame_num=frame_id)
            dmap_uri = "dmap_" + os.path.basename(frame_data["frame_properties"]["streams"][f"{CAM_NAME}"]["stream_properties"]["uri"])[:-4] + ".tiff"

            dmap_name_clean = re.sub(r'^dmap_|\.tiff$', '', dmap_uri)
            dmap_path = os.path.join(self.depths_path, dmap_uri)
            dmap_img = cv.imread(dmap_path, cv.IMREAD_UNCHANGED)

            h, w = dmap_img.shape
            # dmap_img = cv.imread(dmap_path, cv.IMREAD_UNCHANGED)

            depth_dmap_coords = np.argwhere(dmap_img != 0)
            depth_pixels_dict = {(i, j): dmap_img[i, j] / 100 for i, j in depth_dmap_coords} 

            # aux_xyz_3d_coords = np.zeros_like(dmap_img, dtype=np.float32)
            aux_xyz_3d_coords = np.zeros((dmap_img.shape[0], dmap_img.shape[1], 3), dtype=np.float32)

            # Get XY 3D rays from 2D img coordinates (in the camera cs)
            for j in range(0, h):
                for i in range(0, w):
                    if (j, i) in depth_pixels_dict:
                        # Get the intersection between depth plane and the 3D ray that passes through x, y coordinates
                        cam_2d_coords_3x1 = utils.add_homogeneous_row(np.array([i, j]).reshape(2,1))
                        cam_2d_ray3d_3x1 = self.camera.reproject_points2d(points2d_3xN=cam_2d_coords_3x1).reshape(1,3).flatten()

                        cam_2d_3d_coords_3x1 = depth_pixels_dict[j, i] * cam_2d_ray3d_3x1
                        cam_2d_3d_coords_4x1 = utils.add_homogeneous_row(cam_2d_3d_coords_3x1.reshape(3,1))
                        cam_2d_to_lidar_3d_coords_4x1 = np.array([cam_2d_3d_coords_4x1[2, 0],  # Get z as the first element
                                    -cam_2d_3d_coords_4x1[0, 0],  # Negate x for the second element
                                    -cam_2d_3d_coords_4x1[1, 0]]) 
                                                
                        # Fill aux array                            
                        aux_xyz_3d_coords[j, i][0] = cam_2d_to_lidar_3d_coords_4x1[0]
                        aux_xyz_3d_coords[j, i][1] = cam_2d_to_lidar_3d_coords_4x1[1]
                        aux_xyz_3d_coords[j, i][2] = cam_2d_to_lidar_3d_coords_4x1[2]
            
            # Filterout those pixels whose depth is 0
            non_zero_depth_mask = aux_xyz_3d_coords[:, :, 0] != 0
            pcd_3d_lidar_cs = np.asarray(aux_xyz_3d_coords[non_zero_depth_mask]).T
            self.pcd_4d_lidar_cs = utils.add_homogeneous_row(pcd_3d_lidar_cs)
            
            point_cloud_o3d = o3d.geometry.PointCloud()
            points = self.pcd_4d_lidar_cs.T[:,0:3] * 100
            points = points[(points[:, 0] <= 40) & (abs(points[:, 1]) <= 10) & (points[:, 2] <= 5)]
            point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(filename=f"{self.output_path}/raw_{dmap_name_clean}.pcd", pointcloud=point_cloud_o3d, write_ascii=True)

def cluster_PCD():
    def cluster_pcd(self):
        # Clusterize (DBSCAN) 3d bbox
        self.bbox_3d_coords_lidar_cs = self.pcd_4d_lidar_cs[:-1].T
        db = DBSCAN(eps=0.3, min_samples=30).fit(self.bbox_3d_coords_lidar_cs)
        self.labels = db.labels_
        clusters = defaultdict(list)

        for label, point_xyz in zip(self.labels, self.bbox_3d_coords_lidar_cs):
            clusters[label].append(point_xyz)
        clusters = {label: np.array(points) for label, points in clusters.items()}

        # Consider only those clusters that are labelled and sort them by id
        labelled_clusters = {k: v for k, v in clusters.items() if k != -1}
        return sorted(labelled_clusters.items(), key=lambda item: np.mean(item[1][:, 0]))
    
    def get_cuboid_size(self, fig, axis, dimension_idx, dimension_tag):
        gaussian_data = {}
        for ax, (cluster_id, data) in zip(axis, self.sorted_clusters):
            # Extract data from the desired axis
            data = data[:, dimension_idx]
            data_mean_point = (np.max(data) + np.min(data)) / 2
            # Fit gaussians
            gmm = GaussianMixture(n_components=1)
            gmm.fit(data.reshape(-1, 1))

            weights = gmm.weights_
            means = gmm.means_
            covariances = gmm.covariances_

            max_weight_index = np.argmax(weights)

            # Extract the mean and standard deviation of the most prominent Gaussian
            main_mean = means[max_weight_index][0]
            main_std = np.sqrt(covariances[max_weight_index][0][0])

            gaussian_data[cluster_id] = [data_mean_point, main_std]

            # Calculate histogram:
            _, _, _ = ax.hist(data, bins=60, density=True, alpha=0.5, label=f'Cluster {cluster_id}')

            # Plot the GMM distribution
            x = np.linspace(min(data), max(data), 1000)
            logprob = gmm.score_samples(x.reshape(-1, 1))
            pdf = np.exp(logprob)
            ax.plot(x, pdf, 'k--', linewidth=2)
            
            ax.legend()
                
        for ax in axis[len(self.sorted_clusters):]:
            fig.delaxes(ax)
        
        fig.suptitle(f"{dimension_tag} clusters' distributions")
        plt.tight_layout()

        return gaussian_data

    # Cluster pcd
    sorted_clusters = cluster_pcd()

    # Calculate cuboid dimensions and display distributions
    rows = (len(sorted_clusters) + 2 - 1) // 2
    
    fig1, axes1 = plt.subplots(nrows=rows, ncols=2, figsize=(12, 6 * rows))
    axes1 = axes1.flatten()
    depth_gdata = get_cuboid_size(fig=fig1, axis=axes1, dimension_idx=0, dimension_tag="depth")

    fig2, axes2 = plt.subplots(nrows=rows, ncols=2, figsize=(12, 6 * rows))
    axes2 = axes2.flatten()
    width_gdata = get_cuboid_size(fig=fig2, axis=axes2, dimension_idx=1, dimension_tag="width")

    fig3, axes3 = plt.subplots(nrows=rows, ncols=2, figsize=(12, 6 * rows))
    axes3 = axes3.flatten()
    height_gdata = get_cuboid_size(fig=fig3, axis=axes3, dimension_idx=2, dimension_tag="height")


    for cluster_id, cluster_data in sorted_clusters:
        pos_x = depth_gdata[cluster_id][0]
        pos_y = width_gdata[cluster_id][0]
        pos_z = height_gdata[cluster_id][0]

        cluster_depth_scl = cluster_data[:, 0]
        depth_size = abs(np.max(cluster_depth_scl) - np.min(cluster_depth_scl))
        
        cluster_width_scl = cluster_data[:, 1]
        width_size = abs(np.max(cluster_width_scl) - np.min(cluster_width_scl))
        
        cluster_height_scl = cluster_data[:, 2]
        height_size = abs(np.max(cluster_height_scl) - np.min(cluster_height_scl))

def main(data_folder):
    # Run Depth Estimation Inference
    depth_estimation = DepthEstimation(data_folder = data_folder, run_inference = True)
    depth_estimation.run()

    #DepthMap 2 PointCloud ( PseudoPointCloud Generation )
    pcd_generation = ScenePCD(data_folder = data_folder, run_inference = True)
    pcd_generation.run()

    # Cluster PCDs
    cluster_PCD()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth map estimation main script")
    parser.add_argument("--data_folder", type=str, default=f"data/", help="Path to the data folder")
    args = parser.parse_args()
    
    main(data_folder=args.data_folder)