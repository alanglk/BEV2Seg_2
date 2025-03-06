import open3d as o3d
from PIL import Image
import numpy as np
import cv2

from my_utils import check_paths

from typing import List
import pickle
import os


class BEVMapManager():
    GEN_FOLDERS = ['semantic', 'depth', 'pointcloud', 'instances', 'occ_bev_mask', 'tracking']
    
    def __init__(self, 
                 scene_path: str, 
                 gen_flags: dict = {}):
        """
            Folder and path manager for the BEVMap generation.
            gen_flags: {
                'semantic': False, 
                'depth': False, 
                'pointcloud': False, 
                'instances': False,
                'occ_bev_mask':False,
                'tracking': False
                'all': False
            }
        """
        # check if scene_path exists
        if not os.path.exists(scene_path):
            raise Exception(f"scene_path doesnt exist: {scene_path}")

        # Define Generation paths
        self.gen_folder_path  = os.path.join(scene_path, 'generated')
        self.gen_paths = {self.GEN_FOLDERS[i]: os.path.join(self.gen_folder_path, self.GEN_FOLDERS[i]) for i in range(len(self.GEN_FOLDERS))}
        gen_all_flag = not check_paths([self.gen_folder_path])[0]
        
        # check if the generation paths already exists.
        # if a path doesnt exist, create the folder and mark it to regenerate all that type data
        flag_list   = check_paths([p for _, p in self.gen_paths.items()])
        empty_list  = [ len(os.listdir(p)) == 0 for _, p in self.gen_paths.items()]
        self.gen_flags = {self.GEN_FOLDERS[i]: not flag_list[i] or empty_list[i] for i in range(len(self.GEN_FOLDERS))}
        self.gen_flags['all'] = gen_all_flag
        
        # Check user generation flags
        for flag, val in gen_flags.items():
            if flag not in self.gen_flags:
                raise Exception(f"[BEVMapManager]\t Unknown user flag '{flag}'.")
            if not self.gen_flags[flag]:
                self.gen_flags[flag] = self.gen_flags[flag] | val # Override default gen_flag whit user's one

        # DEBUG INFO
        for flag, val in self.gen_flags.items():
            if val == True:
                print(f"[BEVMapManager]\t {flag} is going to be regenerated.")

    def _get_path(self, image_name:str, gen_type:str, file_extension:str) -> str:
        assert gen_type in self.GEN_FOLDERS
        image_name = os.path.splitext(os.path.basename(image_name))[0]
        return os.path.join(self.gen_paths[gen_type],f"{image_name}{file_extension}")

    def exist_gen_file(self, image_name:str, gen_type, file_extension:str) -> bool:
        assert gen_type in self.GEN_FOLDERS
        image_name = os.path.splitext(os.path.basename(image_name))[0]
        path = os.path.join(self.gen_paths[gen_type],f"{image_name}{file_extension}")

        try:
            return check_paths([path]) # always returning true if no exception
        except:
            return False

    def load_semantic_images(self, image_name: str) -> List[np.ndarray]:
        """
            INPUT: image basename. 
                Ej ->  n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg
            OUTPUT: list [image_semantic, image_sb, image_bs] 
            
            Notation:
                sb: normal -> semantic -> bev 
                bs: normal -> bev -> semantic
        """
        image_paths = [
            self._get_path(image_name, 'semantic', '_semantic.png'),
            self._get_path(image_name, 'semantic', '_sb.png'),
            self._get_path(image_name, 'semantic', '_bs.png') ]
        check_paths(image_paths)
        return [cv2.imread(p) for p in image_paths]

    def load_depth_image(self, image_name:str) -> np.ndarray:
        depth_image_path = self._get_path(image_name, 'depth', '.tiff')
        check_paths([depth_image_path])
        return cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    def load_pointcloud(self, image_name:str):
        pcd_path = self._get_path(image_name, 'pointcloud', '.pcd')
        check_paths([pcd_path])
        pcd = o3d.io.read_point_cloud(pcd_path)
        return pcd

    def load_instance_pcds(self, image_name:str) -> dict:
        panoptic_pcd_path = self._get_path(image_name, 'instances', '.plk')
        check_paths([panoptic_pcd_path])

        with open(panoptic_pcd_path, "rb") as f:
            instance_pcds = pickle.load(f)
        return instance_pcds
    
    def load_occ_bev_masks(self, image_name:str) -> dict:
        occ_bev_mask_path = self._get_path(image_name, 'occ_bev_mask', '.pkl')
        check_paths([occ_bev_mask_path])
        with open(occ_bev_mask_path, "rb") as f:
            occ_bev_masks = pickle.load(f)
        return occ_bev_masks
    
    def load_tracking_frame(self, frame_num:int, instance_pcds:dict) -> dict:
        """Load files to update the instance dic object ids
        The file format for each frame is:
        |  center (x, y, z) | tracking_id | semantic label | index_pos |
        |-------------------|-------------|----------------|-----------|
        | x y z             | unknown     | vehicle.car    | 0         |
        | x y z             | unknown     | vehicle.car    | 1         |
        | x y z             | unknown     | vehicle.car    | 2         |
        """
        assert 'tracking' in self.GEN_FOLDERS
        data_path = os.path.join(self.gen_paths['tracking'],f"frame_{frame_num}.txt")
        check_paths([data_path])
        semantic_idx = {}

        for i, semantic_data in enumerate(instance_pcds):
            if semantic_data['label'] not in semantic_idx:
                semantic_idx[semantic_data['label']] = i

        with open(data_path, "r") as f:
            for line in f:
                obj_data = line.strip().split(" ")
                x, y, z = float(obj_data[0]), float(obj_data[1]), float(obj_data[2])
                inst_id         = obj_data[3]
                semantic_label  = obj_data[4]
                idx_pos         = int(obj_data[5])

                # Update object data with tracking
                s_data = instance_pcds[semantic_idx[semantic_label]] # semantic data
                if 'instance_pcds' in s_data: 
                    s_data['instance_pcds'][idx_pos]['inst_id']     = inst_id
                if 'instance_3dboxes' in s_data:
                    s_data['instance_3dboxes'][idx_pos]['inst_id']  = inst_id
                if 'instance_bev_mask' in s_data:
                    s_data['instance_bev_mask'][idx_pos]['inst_id'] = inst_id
        return instance_pcds

    def save_semantic_images(self, image_name:str, images:List[np.ndarray]):
        """
        INPUT:
            image_name: path or name of input image
            images: list of [raw semantic image, sb image, bs image]
        """
        image_paths = [
            self._get_path(image_name, 'semantic', '_semantic.png'),
            self._get_path(image_name, 'semantic', '_sb.png'),
            self._get_path(image_name, 'semantic', '_bs.png') ]
        
        for i, p in enumerate(image_paths):
            cv2.imwrite(p, images[i])

    def save_depth_image(self, image_name:str, depth_dmap: np.ndarray):
        depth_image_path = self._get_path(image_name, 'depth', '.tiff')
        depth_image_pil = Image.fromarray(depth_dmap.astype(np.float32), mode='F')
        depth_image_pil.save(depth_image_path)

    def save_pointcloud(self, image_name:str, pcd: o3d.geometry.PointCloud):
        pcd_path = self._get_path(image_name, 'pointcloud', '.pcd')
        o3d.io.write_point_cloud(filename=pcd_path, pointcloud=pcd, write_ascii=True)      

    def save_instance_pcds(self, image_name:str, instance_pcds: dict):
        inst_path = self._get_path(image_name, 'instances', '.plk')
        with open(inst_path, "wb") as f:
            pickle.dump(instance_pcds, f)
    
    def save_occ_bev_masks(self, image_name:str, occ_bev_masks: dict):
        occ_path = self._get_path(image_name, 'occ_bev_mask', '.pkl')
        with open(occ_path, "wb") as f:
            pickle.dump(occ_bev_masks, f)
    
    def save_tracking_frame(self, frame_num:int, instance_pcds:dict):
        """Save files to perform object trackig
        The file format for each frame is:
        |  center (x, y, z) | tracking_id | semantic label | index_pos |
        |-------------------|-------------|----------------|-----------|
        | x y z             | unknown     | vehicle.car    | 0         |
        | x y z             | unknown     | vehicle.car    | 1         |
        | x y z             | unknown     | vehicle.car    | 2         |
        """
        assert 'tracking' in self.GEN_FOLDERS
        data_path = os.path.join(self.gen_paths['tracking'],f"frame_{frame_num}.txt")
        data = []
        for semantic_data in instance_pcds:
            label = semantic_data['label']
            dynamic = semantic_data['dynamic']
            
            if not dynamic:
                continue # It has no instance computation

            instances = semantic_data['instance_3dboxes']
            for i, inst in enumerate(instances):
                inst_id     = inst['inst_id'] if inst['inst_id'] is not None else "unknown"
                x, y, z     = inst['center']
                sx, sy, sz  = inst['dimensions']
                obj_data = f"{x} {y} {z} {inst_id} {label} {i}\n"
                data.append(obj_data)
        
        # Save data
        with open(data_path, "w") as f:
            f.writelines(data)