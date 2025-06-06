import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage

# pip install nuscenes-devkit # Parser for NuImages dataset
# from nuimages import NuImages
# from nuimages.utils.utils import mask_decode

from oldatasets.NuImages.nuimages_sdk.nuimages import NuImages
from oldatasets.NuImages.nuimages_sdk.utils import mask_decode

# Utils from the same package
from oldatasets.NuImages.nulabels import nulabels, nuname2label, nuid2name, nuid2color
from oldatasets.common import Dataset2BEV, progress_bar, target2image

from vcd import core, types, scl, utils
import numpy as np
import cv2
import os
from PIL import Image

from typing import Union

class NuImagesOpenLABEL():
    def __init__(self, nudataroot:str, nuversion:str, sample_token:str, output_path: str = None, timestamp: str = None):
        """
        OpenLABEL class for one sample of the NuImages dataset
        - nudataroot:   root path of the NuImages dataset
        - output_path:  path of the vcd output file
        - nuversion:    version of the NuImages dataset
        - sample_token: token of the NuImages sample corresponding to this OpenLABEL file
        - timestamp:    timestamp del sample. default: None
        """

        self.vcd = core.VCD()
        self.vcd.add_metadata_properties({
            "nuImages_path":    nudataroot,
            "nuImages_version": nuversion,
            "sample_token":     sample_token,
            "timestamp":        timestamp
            })
        
        # Add frame interval
        self.vcd.add_frame_properties(0)
        
        if output_path is not None:
            self.output_path = os.path.join(output_path, sample_token + ".json")

    def add_coordinate_system(self):
        self.vcd.add_coordinate_system("odom", cs_type=types.CoordinateSystemType.scene_cs)
        C = np.array([0, 0, 0]).reshape(3, 1)
        R = utils.euler2R([0, 0, 0])
        scs_wrt_lcs = utils.create_pose(R, C)
        self.vcd.add_coordinate_system(name="vehicle-iso8855",
                                  cs_type=types.CoordinateSystemType.local_cs,
                                  parent_name="odom", 
                                  pose_wrt_parent=types.PoseData(
                                        val=scs_wrt_lcs.flatten().tolist(),
                                        t_type=types.TransformDataType.matrix_4x4))
    
    def add_sensor(self, sensor, calibrated_sensor, width, height):
        """
        sensor: sensor type in nuimages. Attributes:
                - channel:      CAM_BACK, CAM_FRONT...
                - modality:     camera, lidar...
                - token:        Reference to calibrated_sensor data
        calibrated_sensor: calibrated sensor data (intrinsics and extrinsics)
                - sensor_token: Identifier of the sensor
                - translation:  translation matrix
                - rotation:     rotation matrix (quaternion)
                - camera_intrinsic: intrinsic matrix
                - camera_distortion: distorsion matrix
        width: width resolution of the sensor in pixels
        height: height resolution of the sensor in pixels
        """
        assert sensor['token'] == calibrated_sensor['sensor_token']

        # Extract data
        channel     = sensor['channel']
        token       = sensor['token']
        traslation  = calibrated_sensor['translation']
        rotation_q  = calibrated_sensor['rotation'] # quaternion
        intrinsics  = calibrated_sensor['camera_intrinsic']
        distorsion  = calibrated_sensor['camera_distortion']

        # Add Sensor Extrinsics
        C = np.array(traslation).reshape(3, 1)
        R = utils.q2R(rotation_q[1], rotation_q[2], rotation_q[3], rotation_q[0])
        extrinsics = utils.create_pose(R, C) # scs_wrt_lcs
        extrinsics = types.PoseData(val=extrinsics.flatten().tolist(), t_type=types.TransformDataType.matrix_4x4)
        self.vcd.add_coordinate_system(name=channel, cs_type=types.CoordinateSystemType.sensor_cs,
                                  parent_name="vehicle-iso8855",
                                  pose_wrt_parent=extrinsics)
        
        # Add Sensor Intrinsics
        if sensor['modality'] == 'camera':
            fx = intrinsics[0][0]
            fy = intrinsics[1][1]
            cx = intrinsics[0][2]
            cy = intrinsics[1][2]
            K_3x4 = utils.fromPinholeParamsToCameraMatrix3x4(fx, fy, cx, cy)
            flatK3x4 = [item for sublist in K_3x4 for item in sublist]

            intrinsics = types.IntrinsicsPinhole(width_px=width,
                                                 height_px=height,
                                                 camera_matrix_3x4=flatK3x4,
                                                 distortion_coeffs_1xN=distorsion)
            
            self.vcd.add_stream(stream_name=channel, uri='', description=token, stream_type=core.StreamType.camera)
            self.vcd.add_stream_properties(stream_name=channel, intrinsics=intrinsics)

        elif sensor['modality'] == 'lidar':
            self.vcd.add_stream(stream_name=channel, uri='', 
                                description=token,
                                stream_type=core.StreamType.lidar)
        else:
            self.vcd.add_stream(stream_name=channel, uri='', 
                                description=token,
                                stream_type=core.StreamType.other)

    def add_frame(self, stream_name: str, frame_filename: str, frame_num = 0):
        # TODO: Esto es una chapuza para que funcione
        if "frames" not in self.vcd.data["openlabel"]:
            self.vcd.data["openlabel"]["frames"] = {}
        if frame_num not in self.vcd.data["openlabel"]["frames"]:
            self.vcd.data["openlabel"]["frames"][frame_num] = {}

        self.vcd.add_metadata_properties({"sample_path_hardcoded": frame_filename})
        return
        
        self.vcd.add_stream_properties(stream_name=stream_name,
                                       stream_sync=types.StreamSync(frame_vcd=frame_num),
                                       properties={"uri": frame_filename}
                                       )

    def save_openlabel(self):
        self.vcd.save(self.output_path)
          
class NuImagesDataset(Dataset):
    """
        Class for loading the NuImages Dataset
        NuImages schema:
        - https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuimages.md
        Pytorch Dataset:
        - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        
    """
    DATASET_VERSIONS = {
            'mini'  : 'v1.0-mini',
            'train' : 'v1.0-train',
            'val'   : 'v1.0-val',
            'test'  : 'v1.0-test'
    }
    
    def __init__(self, 
                 dataroot: str = '/data/sets/nuimages',  
                 version: str = 'mini',
                 camera: str = None,
                 transforms = None,
                 remove_empty: bool = True,
                 save_dataset = False,
                 output_path: str = None
                 ):
        """
        Parameters:
            - dataroot      -> root path of the dataset
            - version       -> version of the dataset: 
                            ['mini' (50 samples), 'train' (50 samples), 'val' (50 samples), 'test' (50 samples)]
            - camera        -> If set it will generate a Dataset just for the specific `camera` channel
            - transforms    -> transformations for the image data (Normalization, Scale...)
            - remove_empty  -> If it is a train set, remove entries with no annotations
        """
        super().__init__()

        self.dataroot = dataroot
        self.version = self.DATASET_VERSIONS[version]
        self.nuim = NuImages(dataroot=self.dataroot, version= self.version, verbose=True, lazy=True)
        
        self.camera = camera
        self.transforms = transforms
        
        self.save_dataset = save_dataset
        self.output_path = output_path
        if self.save_dataset and self.output_path is None:
            raise Exception("[NuImagesDataset] save_dataset specified but no output_path provided!!")

        # Check if the dataset has annotated objects or surfaces
        # If not, it is the test dataset
        self.train = len(self.nuim.object_ann) != 0 and len(self.nuim.surface_ann) != 0

        # If training, remove any samples wich doesn't contain annotation data
        self.samples_indices = []
        if remove_empty and self.train:
            total = len(self.nuim.object_ann) + len(self.nuim.surface_ann)
            current = 0
            print(f"[NuImagesDataset]   Loading {total} annotations...")
            sample_with_annotations = set()
            # Record which are the annotation tokens
            for o in self.nuim.object_ann:
                token = o['sample_data_token']
                sample_with_annotations.add(token)
                current += 1
                progress_bar(total, current)
            for s in self.nuim.surface_ann:
                token = s['sample_data_token']
                sample_with_annotations.add(token)
                current += 1
                progress_bar(total, current)

            # Add only indices of samples with annotations
            total = len(self.nuim.sample)
            print(f"[NuImagesDataset]   Loading {total} samples (with and without annotations)...")
            for i, sample in enumerate(self.nuim.sample):
                sample_token = sample['key_camera_token']
                if sample_token in sample_with_annotations:
                    
                    # If self.camera is not set it will add all samples
                    if self._is_sample_from_camera(sample_token):
                        self.samples_indices.append(i)

                progress_bar(total, i +1)
        else:
            # Add every sample index. If self.camera is not set it will add all samples.
            self.samples_indices = [ i for i, sample in enumerate(self.nuim.sample) if self._is_sample_from_camera(sample['key_camera_token'])]

        print(f"[NuImagesDataset]    {len(self.samples_indices)} remaining samples from {len(self.nuim.sample)}")

        # Create category_token to id map based on NuLabels        
        for i, cat in enumerate(self.nuim.category):
            name    = str(cat['name']).strip()
            if name not in nuname2label:
                raise Exception(f"[NuImagesDataset]   ERROR: Category name {name} not found on nuname2label dict")
            
            # Update nulabels with token data
            label_obj = nuname2label[name]
            label_obj.token = cat['token']

        for l in nulabels:
            if l.id != 0:
                assert l.token is not None # Check if all the labels have their tokens (except the background)
        
        self.nutoken2label = { label.token      : label for label in nulabels }
        self.id2color = nuid2color
        self.id2label = nuid2name
        self.label2id = { v : k for k, v in self.id2label.items() }
        print(f"[NuImagesDataset]    {len(self.nutoken2label)} different category tokens succesfully loaded")

        # Create sample to its annotations data map
        # sample_token: [ann0, ann1...]
        self.samle2ann = {}
        for o in self.nuim.object_ann:
            sample_token = o['sample_data_token']
            if sample_token not in self.samle2ann:
                self.samle2ann[sample_token] = []
            # Check if the object annotation has a real bbox and mask
            # Some elements of the NuImages dataset has bboxes with
            # 0 width and 0 height
            b = o['bbox']
            w = b[2] - b[0]
            h = b[3] - b[1]
            m = o['mask']

            if w > 0 and h > 0 and m is not None:
                self.samle2ann[sample_token].append(o)
        for s in self.nuim.surface_ann:
            sample_token = s['sample_data_token']
            if sample_token not in self.samle2ann:
                self.samle2ann[sample_token] = []
            self.samle2ann[sample_token].append(s)
        print(f"[NuImagesDataset]    {len(self.samle2ann)} samples")

    def _is_sample_from_camera(self, sample_token: str):
        """
        Function for checking if a sample is taken from a specific camera (channel)
        INPUT: sample_token
        OUTPUT: boolean
        """

        if self.camera is None:
            return True
        
        sample_data         = self.nuim.get('sample_data', sample_token)
        calibrated_sensor   = self.nuim.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        sensor              = self.nuim.get('sensor', calibrated_sensor['sensor_token'])
        return sensor['channel'] == self.camera

    def __len__(self):
        return len(self.samples_indices)
    
    def _get_imagepath_target(self, index):
        """Chapuza que se llama desde NuImagesFeatureExtractor"""
        assert index < len(self)

        # Get the current sample and associated data
        sample = self.nuim.sample[self.samples_indices[index]]        
        sample_token= sample['key_camera_token']

        sample_data             = self.nuim.get('sample_data', sample_token)
        img_width, img_height   = ( sample_data['width'], sample_data['height'] )
        filename                = sample_data['filename']
        raw_img_path            = os.path.join(self.dataroot, filename)

        # Annotations
        anns = self.samle2ann[sample_token]
        target = np.zeros((img_height, img_width)) # Semantic mask
        for ann in anns:
            mask = mask_decode(ann['mask'])
            label = self.nutoken2label[ann['category_token']]
            target[mask == 1] = label.trainId
               
        return raw_img_path, target

    def __getitem__(self, index):
        """
        INPUT:
            Index of the current sample in the dataset
        OUTPUT:
            image   -> torch.Tensor (H, W, 3) # BGR
            target  -> annotations of the image: {"mask": torch.Tensor ... } (H, W)
        """

        assert index < len(self)

        # Get the current sample and associated data
        sample = self.nuim.sample[self.samples_indices[index]]        
        sample_token= sample['key_camera_token']
        log_token   = sample['log_token']

        sample_data = self.nuim.get('sample_data', sample_token)
        log         = self.nuim.get('log', log_token)

        #print(f"Sample taken from vehicle: {log['vehicle']} on {log['date_captured']} in {log['location']}")
       
        # Extract info from sample_data
        img_width   = sample_data['width']
        img_height  = sample_data['height']
        filename    = sample_data['filename']
        # timestamp   = sample_data['timestamp']
        # ego_token   = sample_data['ego_pose_token']

        # Odometry
        # ego_pose    = self.nuim.get('ego_pose', ego_token)
        
        # Load Raw Image
        raw_img_path = os.path.join(self.dataroot, filename)
        image = cv2.imread(raw_img_path) # BGR

        assert img_height == image.shape[0]
        assert img_width  == image.shape[1]

        # Annotations
        anns = self.samle2ann[sample_token]
        target = np.zeros((img_height, img_width)) # Semantic mask
        for ann in anns:
            mask = mask_decode(ann['mask'])
            label = self.nutoken2label[ann['category_token']]
            target[mask == 1] = label.trainId
        
        # Apply transforms if necessary
        if self.transforms is not None:
            image = self.transforms(image)
            target = self.transforms(target)

        # Save the dataset in output_folder
        if self.save_dataset:
            raw_save_path       = os.path.join(self.output_path, sample_token + "_raw.png")
            color_save_path     = os.path.join(self.output_path, sample_token + "_color.png")
            target_save_path    = os.path.join(self.output_path, sample_token + "_semantic.png")
            
            cv2.imwrite(raw_save_path, image)
            cv2.imwrite(color_save_path, self.target2image(target))
            cv2.imwrite(target_save_path, target)

        return image, target

    def target2image(self, target: Union[torch.tensor, np.ndarray]) -> np.ndarray:
        """
        Converts target (seg mask) into BGR image
        - target: torch.tensor or numpy.ndarray (H, W)

        returns BGR image!!!
        """

        return target2image(target, self.id2color)



class NuImagesFeatureExtractionDataset(NuImagesDataset):
    """Image (semantic) segmentation dataset. BGR Format!!!"""

    def __init__(self, dataroot, version, image_processor, camera = None, transforms=None, remove_empty = True):
        super().__init__(dataroot, version, camera, ToTensor(), remove_empty, False, None)
        self.image_processor = image_processor
        # IMPORTANTE:
        #   Se considera que 0 es el background, pero en nuestro caso
        #   no queremos la clase background. Cuando generamos el BEVDataset se
        #   pone como 0 las regiones que no interesan así que hay que mappear los
        #   0s a 255 ("ignore")

        if image_processor.do_reduce_labels:
            self.id2label = {k-1: v for k, v in self.id2label.items()}
            self.label2id = {k: v-1 for k, v in self.label2id.items()}
            self.id2color = {k-1: v for k, v in self.id2color.items()}

        
        self.id2label[255] = 'ignore'
        self.label2id['ignore'] = 255
        # self.id2color[255] = (255, 255, 255)
    
    def __getitem__(self, index):
        """
        INPUT:
            Index of the current dataset sample
        OUTPUT:
            encoded bev image/target as follows:
            encoded_inputs = {
                "pixel_values": BGR image!!!
                "labels": target
            }
        """
        # Image shape: torch.Size([3, 900, 1600])
        # Target shape: torch.Size([1, 900, 1600])
        image, target = super().__getitem__(index)
        # target[target == 0] = 255

        # Perform data preparation with image_processor 
        # (it shoul be from transformers:SegformerImageProcessor)         
        encoded_inputs = self.image_processor(image, target, return_tensors="pt")
        
        # Remove the batch_dim from each sample
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()

        # encoded_inputs = {
        #     'pixel_values': image,
        #     'labels': target
        # }

        return encoded_inputs

class NuImagesBEVDataset(NuImagesDataset):
    """
    Create BEVDataset from NuImages Dataset
    """
    def __init__(self, 
                 dataroot = '/data/sets/nuimages',
                 output_path: str = None, 
                 save_bevdataset = False,
                 version: str = 'mini',
                 camera: str = None,
                 transforms = None, 
                 remove_empty = True):
        """                
        - output_path: root of the OpenLABEL files of each sample. If not set, the OpenLABEL of each
            sample will be generated on the loop and not saved.
        - save_bevdataset: Save the generated BEV Dataset files.
        
        > To load a previously generated BEVDataset see `datasets.BEV.BEVDataset` 
        
        Examples:
        ```
        # Generate BEV dataset directly from the 'mini' dataset
        NuImagesBEVDataset(dataroot="/nuimages_dataset", version='mini')  

        # Generate BEV dataset directly from the 'mini' dataset 
        # and save the OpenLABEL, images and masks files into output_path
        NuImagesBEVDataset(dataroot="/nuimages_dataset", 
                           output_path="/nuimages_vcd", 
                           save_bevdataset=True)

        # Generate BEV dataset by loading OpenLABEL files 
        # with online conversion to BEV
        NuImagesBEVDataset(dataroot="/nuimages_dataset", 
                           output_path="/nuimages_vcd")
        
        ```

        """
        super().__init__(dataroot, version, camera, transforms, remove_empty)

        self.output_path = output_path
        self.save_bevdataset = save_bevdataset
        
        # Load the OpenLABEL files if output_path is set
        vcd_list = [] 
        if self.output_path is not None and not self.save_bevdataset:
            if not os.path.isdir(self.output_path):
                raise Exception(f"{self.output_path} must be a folder path")
                
            for of in os.listdir(self.output_path):
                of = os.path.join(self.output_path, of)
                
                # Check if it is a .json file
                if os.path.isfile(of) and of.endswith('.json'):
                    vcd = core.OpenLABEL()
                    vcd.load_from_file(of)
                    vcd_list.append({'vcd': vcd, 'path': of})

        # OpenLABEL set to store the dataset info per sample
        self.token2vcd = {} # sample_token to vcd
        for obj in vcd_list:
            vcd  = obj['vcd']
            path = obj['path']
            metadata = vcd.get_metadata()
            if 'sample_token' not in metadata:
                raise Exception(f"{path} vcd file doesn't have 'sample_token' in metadata")
            sample_token = metadata['sample_token']
            self.token2vcd[sample_token] = vcd
        del vcd_list # Free mem space of vcd_list

    
    def __getitem__(self, index):
        """
        INPUT:
            Index of the current sample in the dataset
        OUTPUT:
            image   -> BEV torch.Tensor (H, W, 3) # BGR
            target  -> BEV annotations of the image: {"mask": torch.Tensor ... } (H, W)
        """
        assert index < len(self)
        image, target = super().__getitem__(index)

        # Get the current sample and associated data
        sample = self.nuim.sample[self.samples_indices[index]]
        sample_token= sample['key_camera_token']

        cam_name = None
        nuol = None
        if self.output_path is not None and not self.save_bevdataset:
            # Generate BEV from OpenLABEL file
            vcd = self.token2vcd[sample_token]
            scene = scl.Scene(vcd)
            cam_name = list(vcd.get_root()["streams"].keys())[0]

        else:
            # compute OpenLABEL file on the loop. I know this is redundant 
            # (It is also on the NuImagesDataset class).
            # Search for the sample_data
            sample_data = self.nuim.get('sample_data', sample_token)
            filename    = sample_data['filename']
            img_width   = sample_data['width']
            img_height  = sample_data['height']
            timestamp   = sample_data['timestamp']

            # Sensor data
            sensor_token= sample_data['calibrated_sensor_token']
            calibrated_sensor   = self.nuim.get('calibrated_sensor', sensor_token)
            sensor              = self.nuim.get('sensor', calibrated_sensor['sensor_token'])
            cam_name =  sensor['channel']

            # Create the OpenLABEL object
            nuol = NuImagesOpenLABEL(self.dataroot, self.version, 
                                     sample_token, output_path=self.output_path, 
                                     timestamp=timestamp)
            nuol.add_coordinate_system()
            nuol.add_sensor(sensor, calibrated_sensor, img_width, img_height)
            nuol.add_frame(sensor['channel'], filename)
            
            scene = scl.Scene(nuol.vcd)

        # Reproject to BEV
        # print(f"DEBUG: cam_name: {cam_name}")
        # print(f"DEBUG: scene.cameras: {scene}")
        # print(f"DEBUG: scene get_cam: {scene.get_camera(camera_name=cam_name, frame_num=0)}")
        
        d2bev = Dataset2BEV(cam_name=cam_name, scene=scene)
        
        image_bev, target_bev = d2bev.convert2bev(image, target)

        if self.save_bevdataset:
            raw_save_path       = os.path.join(self.output_path, sample_token + "_raw.png")
            bev_save_path       = os.path.join(self.output_path, sample_token + "_bev.png")
            color_save_path     = os.path.join(self.output_path, sample_token + "_color.png")
            target_save_path    = os.path.join(self.output_path, sample_token + "_semantic.png")
            
            cv2.imwrite(raw_save_path, image)
            cv2.imwrite(bev_save_path, image_bev)
            cv2.imwrite(color_save_path, self.target2image(target_bev))
            cv2.imwrite(target_save_path, target_bev)

            if nuol is not None:
                nuol.save_openlabel()

        return image_bev, target_bev

def generate_NuImagesFormatted_from_NuImages(dataset_path:str, out_path:str, version='mini', cam_name = None) -> tuple:
    """
    Read the NuImages Dataset and write it on the desired format:
    NuImagesFormatted/
        - mini/
            - image1_raw.png
            - image1_semantic.png
            - image1_color.png
        ...
    INPUT:
        - dataset_path:   root directory path of the NuImages dataset
        - out_path:       desired output folder
        - version:        version of the NuImages dataset. ['mini', train', 'val', 'test']
        - cam_name:       name of the camera channel to generate the BEVDataset. It not set all the cameras will be generated.
    OUTPUT: Number of correctly generated samples
    """

    dataset = NuImagesDataset(
                dataroot=dataset_path, 
                version=version,
                camera=cam_name,
                save_dataset=True,
                output_path=out_path)
    
    total = len(dataset)
    error_indexes = []
    for i in range(total):
        progress_bar(total, i +1)
        try:
            dataset.__getitem__(i)

        except Exception as e:
            error_indexes.append(i)
    
    total_generated = total - len(error_indexes)
    return total_generated, error_indexes

def generate_BEVDataset_from_NuImages(dataset_path:str, out_path:str, version='mini', cam_name = None) -> tuple:
    """
    Create the OpenLABEL, BEVImage amd BEVMask files for each sample of a NuImages dataset and store
    them in an output folder.
    INPUT:
        - dataset_path:   root directory path of the NuImages dataset
        - out_path:       desired output folder
        - version:        version of the NuImages dataset. ['mini', train', 'val', 'test']
        - cam_name:       name of the camera channel to generate the BEVDataset. It not set all the cameras will be generated.
    OUTPUT: Number of correctly generated samples
    """

    dataset = NuImagesBEVDataset(dataroot=dataset_path, 
                           output_path=out_path,
                           version=version, 
                           save_bevdataset=True,
                           camera=cam_name)
    
    total = len(dataset)
    error_indexes = []
    for i in range(total):
        progress_bar(total, i +1)
        try:
            dataset.__getitem__(i)

        except Exception as e:
            error_indexes.append(i)
    
    total_generated = total - len(error_indexes)
    return total_generated, error_indexes