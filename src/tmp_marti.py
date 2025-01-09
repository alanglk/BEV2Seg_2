from utils.imports import *

CLEAN_UP_DIRECTORIES = False

DRIVABLE_AREA = False
OPENLABEL_FILE = "openlabel_sample_denoise.json"

FOLDER_NAME = "sample_denoise"
CAM_NAME = "CAM_FRONT"
CS_REF = "vehicle-iso8855"

if DRIVABLE_AREA:
    PCDS_RAW_FOLDER = "drivable_area/raw"
    PCDS_RGB_FOLDER = "drivable_area/rgb"
else:
    PCDS_RAW_FOLDER = "raw"
    PCDS_RGB_FOLDER = "rgb"

def clean_up_output_paths(data_folder):
    depth_path = os.path.join(data_folder, f"output/{FOLDER_NAME}/depths/")
    drivable_area_path = os.path.join(data_folder, f"output/{FOLDER_NAME}/drivable_area")
    pcds_path = os.path.join(data_folder, f"output/{FOLDER_NAME}/pcds")
    pcds2rgb_path = os.path.join(data_folder, f"output/{FOLDER_NAME}/pcds2rgb")

    paths_list = [depth_path, drivable_area_path, pcds_path, pcds2rgb_path]

    def delete_folder(path):
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for directory in dirs:
                    os.rmdir(os.path.join(root, directory))
            os.rmdir(path)
            print(f"Deleted: {path}")
        else:
            print(f"Path does not exist: {path}")

    for path in paths_list:
        delete_folder(path)
def validate_paths(paths_list):
    for directory in paths_list:
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist. Creating it now...")
            os.makedirs(directory)
def load_imgs_dataset(path):
    """Loads the image dataset from the specified directory."""
    return [file for file in os.listdir(path) 
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
def load_pcds_dataset(path):
    """Loads the pcd dataset from the specified directory."""
    return [file for file in os.listdir(path) 
            if file.lower().endswith(('.pcd'))]
def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
# Depth estimation using the ml-depth-pro model
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
        # device = "cpu"    
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
            focal_length = prediction["focallength_px"] # estimated focal length
            print(f"Estimated focal_length: {focal_length}")
            print(f"IMG shape: {image.shape}")

            # Convert the depth tensor to a NumPy array
            depth_image_np = depth_image.detach().cpu().numpy()
            depth_image_pil = Image.fromarray(depth_image_np.astype(np.float32), mode='F')
            depth_image_pil.save(f"{self.output_path}/dmap_{img_name_clean}.tiff")
# Segment drivable area using the YoloPV2 model
class SegmentDrivableArea:
    def __init__(self, data_folder, run_inference = True) -> None:
        self.data_folder = data_folder
        self.run_inference = run_inference
        
        self.images_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/images")
        
        self.output_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/drivable_area")
        scene_mask_path = os.path.join(self.output_path, "images")
        ll_mask_path = os.path.join(self.output_path, "ll_mask")
        da_mask_path = os.path.join(self.output_path, "da_mask")
        labels_path = os.path.join(self.output_path, "labels")

        validate_paths(paths_list=[self.images_path, self.output_path, ll_mask_path, da_mask_path, labels_path, scene_mask_path])

    def run(self):
        if not self.run_inference:
            print(f"Run inference is set to False. Loading segmented masks from caché. Folder: {self.output_path}")
            return
        
        YOLOPv2.run.detect(project_name=FOLDER_NAME, img_dataset=self.images_path, output_path=self.output_path)
# From depth map generate the associated pointcloud for the entire scene
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
            #dmap_name = 
        #for dmap_name in tqdm(self.dmap_dataset, desc="Depth Maps to PseudoPCDs", unit="dmap"):
            #print(f"DMAP {dmap_uri}")
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
# From depth map generate the associated pointcloud for the drivable area
class DrivablePCD:
    def __init__(self, data_folder, run_inference = True) -> None:
        self.data_folder = data_folder
        self.run_inference = run_inference
        
        self.images_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/images")
        self.openlabel_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/openlabel")

        self.da_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/drivable_area/da_mask")
        self.depths_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/depths")
        self.output_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/pcds/drivable_area/raw")
        
        validate_paths(paths_list=[self.images_path, self.da_path, self.depths_path, self.output_path, self.openlabel_path])
        
        self.vcd = core.OpenLABEL()
        self.vcd.load_from_file(os.path.join(self.openlabel_path, f"{OPENLABEL_FILE}"))
        self.scene = scl.Scene(vcd = self.vcd)
        self.camera = self.scene.get_camera(camera_name=CAM_NAME)

        self.images_dataset = load_imgs_dataset(path=self.images_path)
        self.da_dataset = load_imgs_dataset(path=self.da_path)
        # self.dmap_dataset = load_imgs_dataset(path=self.depths_path)
    
    def run(self):
        if not self.run_inference:
            print(f"Run DrivablePCD generation is set to False. Loading drivable PCDs from caché. Folder: {self.output_path}")
            return
        
        total_frames = self.vcd.data["openlabel"]["frame_intervals"][0]["frame_end"]
        
        for frame_id in tqdm(range(0, total_frames + 1), desc="Depth Maps to Drivable PseudoPCDs", unit="dmap"):
            frame_data = self.vcd.get_frame(frame_num=frame_id)
            dmap_uri = "dmap_" + os.path.basename(frame_data["frame_properties"]["streams"][f"{CAM_NAME}"]["stream_properties"]["uri"])[:-4] + ".tiff"
            da_uri = "mask_" + os.path.basename(frame_data["frame_properties"]["streams"][f"{CAM_NAME}"]["stream_properties"]["uri"])

            dmap_name_clean = re.sub(r'^dmap_|\.tiff$', '', dmap_uri)
            dmap_path = os.path.join(self.depths_path, dmap_uri)
            dmap_img = cv.imread(dmap_path, cv.IMREAD_UNCHANGED)
            dmap_h, dmap_w = dmap_img.shape[:2]

            da_path = os.path.join(self.da_path, da_uri)
            da_img = cv.imread(da_path, cv.IMREAD_UNCHANGED)
            da_img = cv.resize(da_img, (dmap_w, dmap_h), interpolation=cv.INTER_LINEAR)

            depth_dmap_coords = np.argwhere(dmap_img != 0)
            da_depth_coords = [(i, j) for i, j in depth_dmap_coords if da_img[i, j] != 0]
            depth_pixels_dict = {(i, j): dmap_img[i, j] / 100 for i, j in da_depth_coords}
            aux_xyz_3d_coords = np.zeros((dmap_img.shape[0], dmap_img.shape[1], 3), dtype=np.float32)

            # Get XY 3D rays from 2D img coordinates (in the camera cs)
            for j in range(0, dmap_h):
                for i in range(0, dmap_w):
                    if (j, i) in depth_pixels_dict:
                        # Get the intersection between depth plane and the 3D ray that passes through x, y coordinates
                        cam_2d_coords_3x1 = utils.add_homogeneous_row(np.array([i, j]).reshape(2,1))
                        cam_2d_ray3d_3x1 = self.camera.reproject_points2d(points2d_3xN=cam_2d_coords_3x1).reshape(1,3).flatten()

                        cam_2d_3d_coords_3x1 = depth_pixels_dict[j, i] * cam_2d_ray3d_3x1
                        cam_2d_3d_coords_4x1 = utils.add_homogeneous_row(cam_2d_3d_coords_3x1.reshape(3,1))
                        cam_2d_to_lidar_3d_coords_4x1 = np.array([cam_2d_3d_coords_4x1[2, 0], -cam_2d_3d_coords_4x1[0, 0], -cam_2d_3d_coords_4x1[1, 0]]) 
                                                
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
# Paint pointcloud based on the rgb image values
class PaintPCD:
    def __init__(self, data_folder, run_inference = True) -> None:
        self.data_folder = data_folder
        self.run_inference = run_inference
        
        self.images_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/images")
        self.openlabel_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/openlabel")
        self.pcds_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/pcds/{PCDS_RAW_FOLDER}")
        
        self.tmp_output_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/pcds2rgb")
        self.output_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/pcds/{PCDS_RGB_FOLDER}")

        validate_paths(paths_list=[self.images_path, self.pcds_path, self.tmp_output_path, self.output_path, self.openlabel_path])
        
        self.vcd = core.OpenLABEL()
        self.vcd.load_from_file(os.path.join(self.openlabel_path, f"{OPENLABEL_FILE}"))
        self.scene = scl.Scene(vcd = self.vcd)
        self.camera = self.scene.get_camera(camera_name=CAM_NAME)

        self.images_dataset = load_imgs_dataset(path=self.images_path)
        self.pcds_dataset = load_pcds_dataset(path=self.pcds_path)
    
    def run(self) -> None:
        if not self.run_inference:
            print(f"Run PaintPCD is set to False. Loading colored PCDs from caché. Folder: {self.output_path}")
            return
        
        total_frames = self.vcd.data["openlabel"]["frame_intervals"][0]["frame_end"]
        
        for frame_id in tqdm(range(0, total_frames + 1), desc="Coloring PCDs", unit="pcd"):
            frame_data = self.vcd.get_frame(frame_num=frame_id)
            pcd_uri = "raw_" + os.path.basename(frame_data["frame_properties"]["streams"][f"{CAM_NAME}"]["stream_properties"]["uri"])[:-4] + ".pcd"
            pcd_path = os.path.join(self.pcds_path, pcd_uri)
            pcd = o3d.io.read_point_cloud(pcd_path)

            img_name = os.path.basename(frame_data["frame_properties"]["streams"][f"{CAM_NAME}"]["stream_properties"]["uri"])
            img_path = os.path.join(self.images_path, img_name)
            img = cv.imread(img_path)

            image_pcd_gt = img.copy()
            image_dmap2pcd = img.copy()

            image_height, image_width, _ = img.shape

            points_from_dmap_lidar_cs = np.asarray(pcd.points)
            points_from_dmap_cam_cs = np.column_stack((- points_from_dmap_lidar_cs[:, 1], - points_from_dmap_lidar_cs[:, 2], points_from_dmap_lidar_cs[:, 0]))
            pcd.points = o3d.utility.Vector3dVector(points_from_dmap_cam_cs) # convert to ccs
            pcd.paint_uniform_color([1, 0, 1]) # Purple

            # Project estimated PCD to img
            def pcd2img(image_original, image, points, color_code = None):
                points_rgb = np.zeros((points.shape[0], 3), dtype=np.uint8)

                pcd_projected, idx_valid = self.camera.project_points3d(points3d_4xN=utils.add_homogeneous_row(points.T))
                pcd_projected = pcd_projected[:, idx_valid].T

                values = points[:, 2]
                norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
                cmap = plt.get_cmap('viridis')

                for i, (pixel_coordinate, value) in enumerate(zip(pcd_projected, norm_values)):
                    x = int(pixel_coordinate[0])
                    y = int(pixel_coordinate[1])

                    if 0 <= x < image_width and 0 <= y < image_height:
                        pixel_color = image_original[y, x]  # OpenCV uses BGR format
                        points_rgb[i] = pixel_color  
                        color = cmap(value)
                        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
                        if color_code == "MAP":
                            cv.circle(image, (x, y), radius=1, color=color_bgr, thickness=-1)
                        else:
                            cv.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
                    else:
                        points_rgb[i] = (0, 0, 0)  # Assign a default color (e.g., black) for out-of-bounds points
                
                points_with_colors = np.hstack((points, points_rgb))  # Shape will be (N, 6)
                return image, points_with_colors

            image_dmap2pcd, o3d_dmap2pcd = pcd2img(image_original=img, image=image_dmap2pcd, points=points_from_dmap_cam_cs, color_code="MAP")
            combined_image = np.hstack((image_dmap2pcd, image_pcd_gt))

            cv.imwrite(f"{self.tmp_output_path}/pcd2img_{img_name}", combined_image)

            # Display PCDs in O3D
            pcd_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0.0, 0.0, 0.0]))
            # o3d.visualization.draw_geometries([pcd_frame, pcd_from_dmap_o3d, pcd_wide_o3d])

            pcd_dmap = o3d.geometry.PointCloud()
            #pcd_dmap_points_iso8855 = np.column_stack((o3d_dmap2pcd[:, 2], -o3d_dmap2pcd[:, 0], -o3d_dmap2pcd[:, 1]))
            pcd_dmap_points_iso8855 = np.column_stack((o3d_dmap2pcd[:, 0], o3d_dmap2pcd[:, 1], o3d_dmap2pcd[:, 2]))
            pcd_dmap.points = o3d.utility.Vector3dVector(pcd_dmap_points_iso8855)
            o3d_dmap2pcd[:, 3:] = o3d_dmap2pcd[:, [5, 4, 3]]
            pcd_dmap.colors = o3d.utility.Vector3dVector(o3d_dmap2pcd[:,3:]/255.0)

            # pcd_gt = o3d.geometry.PointCloud()
            # pcd_gt.points = o3d.utility.Vector3dVector(o3d_pcd_gt[:,0:3]) 
            # pcd_gt.colors = o3d.utility.Vector3dVector(o3d_pcd_gt[:,3:]/255.0)

            #o3d.visualization.draw_geometries([pcd_frame, pcd_dmap], window_name = "Window 3D")

            # Export the point cloud to a .pcd file
            # o3d.io.write_point_cloud(f"data/imgs_seq/nuscenes/colored/{CAM}_{IMG_ID}_bin_pcd.pcd", pcd_dmap)
            o3d.io.write_point_cloud(f"{self.output_path}/rgb_{img_name[:-4]}.pcd", pcd_dmap, write_ascii=True)
            # o3d.visualization.draw_geometries([pcd_gt])
# Align pointclouds using the Iterative Closest Point (ICP) algorithm
class ICPAlignment:
    def __init__(self, data_folder, run_inference = True) -> None:
        self.data_folder = data_folder
        self.run_inference = run_inference
        
        self.images_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/images")
        self.openlabel_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/openlabel")
        self.pcds_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/pcds/{PCDS_RGB_FOLDER}")
        
        self.output_path_odometry = os.path.join(self.data_folder, f"output/reconstructed_ground/{FOLDER_NAME}")
        self.output_path_aligned = os.path.join(self.data_folder, f"output/reconstructed_ground/{FOLDER_NAME}")        

        validate_paths(paths_list=[self.images_path, self.openlabel_path, 
                                   self.pcds_path, self.output_path_odometry, self.output_path_aligned])
        
        self.vcd = core.OpenLABEL()
        self.vcd.load_from_file(os.path.join(self.openlabel_path, f"{OPENLABEL_FILE}"))
        self.scene = scl.Scene(vcd = self.vcd)
        self.camera = self.scene.get_camera(camera_name=CAM_NAME)

        self.images_dataset = load_imgs_dataset(path=self.images_path)
        self.pcds_dataset = load_imgs_dataset(path=self.pcds_path)
    
    def run(self):
        if not self.run_inference:
            print(f"Run ICPAlignment generation is set to False. Loading aligned PCDs from caché. Folder: {self.output_path}")
            return
        
        total_frames = self.vcd.data["openlabel"]["frame_intervals"][0]["frame_end"]
        pcds_odom_transform = []

        for frame_id in tqdm(range(0, total_frames + 1), desc="Depth Maps to PseudoPCDs", unit="dmap"):
            frame_data = self.vcd.get_frame(frame_num=frame_id)
            pcd_uri = "rgb_" + os.path.basename(frame_data["frame_properties"]["streams"][f"{CAM_NAME}"]["stream_properties"]["uri"])[:-4] + ".pcd"

            pcd_path = os.path.join(self.pcds_path, pcd_uri)

            pcd_iso = o3d.io.read_point_cloud(f"{pcd_path}")
            pcd_ccs = o3d.io.read_point_cloud(f"{pcd_path}")

            points_lidar_ccs = utils.add_homogeneous_row(np.asarray(pcd_ccs.points).T)
            points_lidar_iso = self.scene.transform_points3d_4xN(points3d_4xN=points_lidar_ccs, cs_dst=f"{CS_REF}", cs_src=f"{CAM_NAME}", frame_num=frame_id)
            
            pcd_iso.points = o3d.utility.Vector3dVector(points_lidar_iso.T[:,0:3])
            # pcd_ccs.points = o3d.utility.Vector3dVector(points_lidar_ccs.T[:,0:3])

            pcds_odom_transform.append(pcd_iso)
        
        reconstructed_pcds = o3d.geometry.PointCloud()
        for pcd in pcds_odom_transform:
            reconstructed_pcds += pcd

        o3d.io.write_point_cloud(f"{self.output_path_odometry}/odometry_reconstruction_pcd.pcd", reconstructed_pcds, write_ascii=True)

class InstantiateSemanticSegmentations():
    def __init__(self, data_folder, run_inference = True) -> None:
        self.data_folder = data_folder
        self.run_inference = run_inference
        
        self.images_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/images")
        self.semantic_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/semantic")
        self.labels_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/drivable_area/labels")
        self.openlabel_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/openlabel")

        self.output_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/semantic")
        validate_paths(paths_list=[self.images_path, self.semantic_path, self.output_path])

        self.images_dataset = load_imgs_dataset(path=self.images_path)

        self.vcd = core.OpenLABEL()
        self.vcd.load_from_file(os.path.join(self.openlabel_path, f"{OPENLABEL_FILE}"))
        self.scene = scl.Scene(vcd = self.vcd)
        self.camera = self.scene.get_camera(camera_name=CAM_NAME)
    
    def run(self):
        if not self.run_inference:
            print(f"Run DenoisePCD is set to False. Loading denoised PCDs from caché. Folder: {self.output_path}")
            return
        total_frames = self.vcd.data["openlabel"]["frame_intervals"][0]["frame_end"]

        for frame_id in tqdm(range(0, total_frames + 1), desc="Img instantiation", unit="img"):
            frame_data = self.vcd.get_frame(frame_num=frame_id)
            
            img_uri = os.path.basename(frame_data["frame_properties"]["streams"][f"{CAM_NAME}"]["stream_properties"]["uri"])
            img_path = os.path.join(self.images_path, img_uri)
            img = cv.imread(img_path)

            labels_uri = "labels_" + img_uri[:-4] + ".txt"
            img_labels = os.path.join(self.labels_path, labels_uri)

            pcd_path = "data/output/sample_denoise/pcds/rgb/rgb_928d3dd5359b4fa3b540646da963da7f_raw.pcd"
            pcd = o3d.io.read_point_cloud(pcd_path)

            # Load labels from the text file (assuming the file is named 'labels.txt')
            with open(img_labels, 'r') as file:
                lines = file.readlines()

            # List to store bounding boxes and their IDs
            bounding_boxes = []
            pcd_scene = []
            for line in lines:
                elements = line.strip().split()
                object_id = int(elements[0])
                top_left_x = int(elements[1])
                top_left_y = int(elements[2])
                bottom_right_x = int(elements[3])
                bottom_right_y = int(elements[4])
                score = elements[5]
                bounding_boxes.append((object_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y))

            for id, box in enumerate(bounding_boxes):
                _, x1, y1, x2, y2 = box
                # Paint bboxes
                # Create instance-semantic pcd
                semantic_img = img.copy() #cv.imread(semantic_img_path)

                image_height, image_width, _ = img.shape

                points_from_dmap_lidar_cs = np.asarray(pcd.points)
                # points_from_dmap_cam_cs = np.column_stack((- points_from_dmap_lidar_cs[:, 1], - points_from_dmap_lidar_cs[:, 2], points_from_dmap_lidar_cs[:, 0]))
                pcd.points = o3d.utility.Vector3dVector(points_from_dmap_lidar_cs) # convert to ccs
                pcd.paint_uniform_color([1, 0, 1]) # Purple

                # Project estimated PCD to img
                def pcd2img(image_original, image, points, color_code = None):
                    points_rgb = np.zeros((points.shape[0], 3), dtype=np.uint8)
                    points_instance = np.zeros((points.shape[0], 3), dtype=np.uint8)

                    pcd_projected, idx_valid = self.camera.project_points3d(points3d_4xN=utils.add_homogeneous_row(points.T))
                    pcd_projected = pcd_projected[:, idx_valid].T

                    values = points[:, 2]
                    norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))

                    for i, (pixel_coordinate, value) in enumerate(zip(pcd_projected, norm_values)):
                        x = int(pixel_coordinate[0])
                        y = int(pixel_coordinate[1])

                        if x1 <= x < x2 and y1 <= y < y2:
                            pixel_color = image_original[y, x]  # OpenCV uses BGR format
                            points_rgb[i] = pixel_color
                            points_instance[i] = pixel_color
                        elif 0 <= x < image_width and 0 <= y < image_height:
                            pixel_color = image_original[y, x]  # OpenCV uses BGR format
                            points_rgb[i] = pixel_color
                        else:
                            points_rgb[i] = (0, 0, 0)  # Assign a default color (e.g., black) for out-of-bounds points
                            points_instance[i] = (0, 0, 0)
                    
                    points_with_colors = np.hstack((points, points_rgb))  # Shape will be (N, 6)
                    points_with_colors_instance = np.hstack((points, points_instance))  # Shape will be (N, 6)
                    mask = np.any(points_with_colors_instance[:, 3:] != 0, axis=1)
                    instance_points = points_with_colors[mask]

                    return points_with_colors, instance_points, mask

                o3d_dmap2pcd, instance_pcd, instance_mask = pcd2img(image_original=semantic_img, image=img, points=points_from_dmap_lidar_cs, color_code="MAP")

                # Display PCDs in O3D
                pcd_dmap = o3d.geometry.PointCloud()
                #pcd_dmap_points_iso8855 = np.column_stack((o3d_dmap2pcd[:, 2], -o3d_dmap2pcd[:, 0], -o3d_dmap2pcd[:, 1]))
                pcd_dmap_points_iso8855 = np.column_stack((o3d_dmap2pcd[:, 0], o3d_dmap2pcd[:, 1], o3d_dmap2pcd[:, 2]))
                pcd_dmap.points = o3d.utility.Vector3dVector(pcd_dmap_points_iso8855)
                # o3d_dmap2pcd[:, 3:] = o3d_dmap2pcd[:, [5, 4, 3]] / 255.0
                pcd_dmap.colors = o3d.utility.Vector3dVector(o3d_dmap2pcd[:,3:]/255.0)

                points_dmap = np.asarray(pcd_dmap.points)
                colors_dmap = np.asarray(pcd_dmap.colors)


                # Apply the inverse of the mask to remove points
                filtered_points = points_dmap[~instance_mask]
                filtered_colors = colors_dmap[~instance_mask]


                pcd_dmap.points = o3d.utility.Vector3dVector(filtered_points)
                pcd_dmap.colors = o3d.utility.Vector3dVector(filtered_colors)
                o3d.io.write_point_cloud(f"{self.output_path}/semantic_{id}.pcd", pcd_dmap, write_ascii=True)

                
                from sklearn.cluster import DBSCAN
                points = instance_pcd[:,0:3]
                colors = instance_pcd[:,3:]

                # 3. Apply DBSCAN for clustering
                dbscan = DBSCAN(eps=0.1, min_samples=15)  # You may need to tune these parameters
                labels = dbscan.fit_predict(points)

                # 4. Identify the cluster to keep
                # For example, if you want to keep the largest cluster:
                unique_labels, counts = np.unique(labels, return_counts=True)
                largest_cluster_label = unique_labels[np.argmax(counts)]  # Find the label of the largest cluster

                # 5. Filter out points that don't belong to the chosen cluster
                mask = labels == largest_cluster_label  # Mask for the largest cluster
                filtered_points = points[mask]
                filtered_colors = colors[mask]

                # Assuming pcd_dmap is your existing point cloud
                points_dmap = np.asarray(pcd_dmap.points)
                colors_dmap = np.asarray(pcd_dmap.colors)

                # Add the new points and colors to the existing ones
                new_points = np.vstack((points_dmap, filtered_points))  # Append the filtered points
                new_colors = np.vstack((colors_dmap, filtered_colors))  # Append the filtered colors

                # Update the PointCloud object
                pcd_dmap.points = o3d.utility.Vector3dVector(new_points)
                pcd_dmap.colors = o3d.utility.Vector3dVector(new_colors)

                # Export the point cloud to a .pcd file
                # o3d.io.write_point_cloud(f"data/imgs_seq/nuscenes/colored/{CAM}_{IMG_ID}_bin_pcd.pcd", pcd_dmap)
                # o3d.io.write_point_cloud(f"{self.output_path}/semantic_instance_{id}.pcd", pcd_dmap, write_ascii=True)
                pcd_scene.append(pcd_dmap)

            all_points = []
            all_colors = []

            # 2. Iterate through each point cloud and append its points and colors to the lists
            for pcd in pcd_scene:
                all_points.append(np.asarray(pcd.points))  # Extract points
                all_colors.append(np.asarray(pcd.colors))  # Extract colors

            # 3. Concatenate all points and colors into one array
            all_points = np.vstack(all_points)
            all_colors = np.vstack(all_colors)

            # 4. Create a new point cloud from the concatenated points and colors
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(all_points)
            merged_pcd.colors = o3d.utility.Vector3dVector(all_colors)

            # 5. Visualize the merged point cloud
            o3d.io.write_point_cloud(f"{self.output_path}/semantic_all.pcd", merged_pcd, write_ascii=True)


class  DenoisePCD:
    def __init__(self, data_folder, run_inference = True) -> None:
        self.data_folder = data_folder
        self.run_inference = run_inference
        
        self.images_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/images")
        self.semantic_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/semantic")
        self.pcd_rgb_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/pcds/raw")
        self.openlabel_path = os.path.join(self.data_folder, f"input/{FOLDER_NAME}/openlabel")
        self.labels_path = os.path.join(self.data_folder, f"data/output/{FOLDER_NAME}/drivable_area/labels")

        self.output_path = os.path.join(self.data_folder, f"output/{FOLDER_NAME}/pcds/denoised")
        validate_paths(paths_list=[self.images_path, self.semantic_path, self.pcd_rgb_path, self.output_path])

        self.images_dataset = load_imgs_dataset(path=self.images_path)

        self.vcd = core.OpenLABEL()
        self.vcd.load_from_file(os.path.join(self.openlabel_path, f"{OPENLABEL_FILE}"))
        self.scene = scl.Scene(vcd = self.vcd)
        self.camera = self.scene.get_camera(camera_name=CAM_NAME)
    
    def run(self):
        if not self.run_inference:
            print(f"Run DenoisePCD is set to False. Loading denoised PCDs from caché. Folder: {self.output_path}")
            return
        
        total_frames = self.vcd.data["openlabel"]["frame_intervals"][0]["frame_end"]
        
        for frame_id in tqdm(range(0, total_frames + 1), desc="Denoising PCDs", unit="pcd"):
            frame_data = self.vcd.get_frame(frame_num=frame_id)
            # pcd_uri = "raw_" + os.path.basename(frame_data["frame_properties"]["streams"][f"{CAM_NAME}"]["stream_properties"]["uri"])[:-4] + ".pcd"
            # pcd_path = os.path.join(self.pcd_rgb_path, pcd_uri)
            pcd_path = "data/output/sample_denoise/pcds/raw/raw_928d3dd5359b4fa3b540646da963da7f_raw.pcd"

            pcd = o3d.io.read_point_cloud(pcd_path)
            if pcd is not None:
                print("pcd has been loaded")

            # img_name = os.path.basename(frame_data["frame_properties"]["streams"][f"{CAM_NAME}"]["stream_properties"]["uri"])
            # img_path = os.path.join(self.images_path, img_name)
            # img = cv.imread(img_path)
            img = cv.imread("data/input/sample_denoise/semantic/928d3dd5359b4fa3b540646da963da7f_raw_color.png")
            if img is not None:
                print("img has been loaded")

            # semantic_img_path = os.path.join(self.semantic_path, img_name[:-4] + "_color.png")
            semantic_img = img.copy() #cv.imread(semantic_img_path)

            image_height, image_width, _ = img.shape

            points_from_dmap_lidar_cs = np.asarray(pcd.points)
            points_from_dmap_cam_cs = np.column_stack((- points_from_dmap_lidar_cs[:, 1], - points_from_dmap_lidar_cs[:, 2], points_from_dmap_lidar_cs[:, 0]))
            pcd.points = o3d.utility.Vector3dVector(points_from_dmap_cam_cs) # convert to ccs
            pcd.paint_uniform_color([1, 0, 1]) # Purple

            # Project estimated PCD to img
            def pcd2img(image_original, image, points, color_code = None):
                points_rgb = np.zeros((points.shape[0], 3), dtype=np.uint8)

                pcd_projected, idx_valid = self.camera.project_points3d(points3d_4xN=utils.add_homogeneous_row(points.T))
                pcd_projected = pcd_projected[:, idx_valid].T

                values = points[:, 2]
                norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
                cmap = plt.get_cmap('viridis')

                for i, (pixel_coordinate, value) in enumerate(zip(pcd_projected, norm_values)):
                    x = int(pixel_coordinate[0])
                    y = int(pixel_coordinate[1])

                    if 0 <= x < image_width and 0 <= y < image_height:
                        pixel_color = image_original[y, x]  # OpenCV uses BGR format
                        points_rgb[i] = pixel_color 
                        color = cmap(value)
                        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
                        if color_code == "MAP":
                            cv.circle(image, (x, y), radius=1, color=color_bgr, thickness=-1)
                        else:
                            cv.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
                    else:
                        points_rgb[i] = (0, 0, 0)  # Assign a default color (e.g., black) for out-of-bounds points
                
                points_with_colors = np.hstack((points, points_rgb))  # Shape will be (N, 6)
                return image, points_with_colors

            image_dmap2pcd, o3d_dmap2pcd = pcd2img(image_original=semantic_img, image=img, points=points_from_dmap_cam_cs, color_code="MAP")

            # Display PCDs in O3D
            pcd_dmap = o3d.geometry.PointCloud()
            #pcd_dmap_points_iso8855 = np.column_stack((o3d_dmap2pcd[:, 2], -o3d_dmap2pcd[:, 0], -o3d_dmap2pcd[:, 1]))
            pcd_dmap_points_iso8855 = np.column_stack((o3d_dmap2pcd[:, 0], o3d_dmap2pcd[:, 1], o3d_dmap2pcd[:, 2]))
            pcd_dmap.points = o3d.utility.Vector3dVector(pcd_dmap_points_iso8855)
            # o3d_dmap2pcd[:, 3:] = o3d_dmap2pcd[:, [5, 4, 3]] / 255.0
            pcd_dmap.colors = o3d.utility.Vector3dVector(o3d_dmap2pcd[:,3:]/255.0)

            # pcd_gt = o3d.geometry.PointCloud()
            # pcd_gt.points = o3d.utility.Vector3dVector(o3d_pcd_gt[:,0:3]) 
            # pcd_gt.colors = o3d.utility.Vector3dVector(o3d_pcd_gt[:,3:]/255.0)

            #o3d.visualization.draw_geometries([pcd_frame, pcd_dmap], window_name = "Window 3D")

            # Export the point cloud to a .pcd file
            # o3d.io.write_point_cloud(f"data/imgs_seq/nuscenes/colored/{CAM}_{IMG_ID}_bin_pcd.pcd", pcd_dmap)
            o3d.io.write_point_cloud(f"{self.output_path}/semantic_test.pcd", pcd_dmap, write_ascii=True)

            


# TODO
# class PCDAlignment()
# class MeshGeneration()
# class PCDDenoising
# realistic BEV

def main(data_folder):
    if CLEAN_UP_DIRECTORIES:
        clean_up_output_paths(data_folder = data_folder)

    # Run Depth Estimation Inference
    depth_estimation = DepthEstimation(data_folder = data_folder, run_inference = True)
    depth_estimation.run()

    #DepthMap 2 PointCloud ( PseudoPointCloud Generation )
    if DRIVABLE_AREA:
        da_segmentation = SegmentDrivableArea(data_folder = data_folder, run_inference = True)
        da_segmentation.run()
        pcd_generation = DrivablePCD(data_folder = data_folder, run_inference = True)
    else:
        pcd_generation = ScenePCD(data_folder = data_folder, run_inference = True)
    pcd_generation.run()

    # Paint PointClouds
    rgb_pcd = PaintPCD(data_folder=data_folder, run_inference = True)
    rgb_pcd.run()

    inst_seg = InstantiateSemanticSegmentations(data_folder=data_folder, run_inference=False)
    inst_seg.run()

    # PCD denoising
    denoised_pcd = DenoisePCD(data_folder=data_folder, run_inference=False)
    denoised_pcd.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel2World main script")
    parser.add_argument("--data_folder", type=str, default=f"data/", help="Path to the input image file")
    args = parser.parse_args()
    