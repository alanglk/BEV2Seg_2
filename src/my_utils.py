import numpy as np
import open3d as o3d
import cv2

from vcd import utils

from typing import List
import os

"""
Semantic/Instance dict sctructure:
instance_pointclouds: 
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
                }]
            }]
"""


def check_paths(paths: List[str]) -> List[bool]:
    """
        INPUT: path list of dirs and files
        OUPUT: flag_list with wheter paths exists or not. 
            If a folder path doesn't exist, it is created.
    """
    flag_list = []
    for path in paths:
        if os.path.exists(path):
            flag_list.append(True) # File or Folder exist
            continue
        
        # Set as non existing path and create it if it is a folder path
        flag_list.append(False)
        is_folder_path = not os.path.splitext(path)[1]
        if is_folder_path:  
            os.makedirs(path, exist_ok=True)  # use makedirs for creating intermediate directories
            print(f"Directory '{path}' was created.")
            continue
        
        raise Exception(f"Path '{path}' does not exist and it is not going to be created.")

    return flag_list

def get_pallete(N: int) -> np.ndarray:
    """
    Return Nx3 uint8 np.ndarray BGR color palette.
    """
    # Define a default pastel palette
    DEFAULT_PALLETE = [
        (255, 182, 193),  # Light Pink
        (135, 206, 250),  # Light Sky Blue
        (152, 251, 152),  # Pale Green
        (240, 230, 140),  # Khaki
        (221, 160, 221),  # Plum
        (255, 228, 196),  # Bisque
        (173, 216, 230),  # Light Blue
        (250, 250, 210),  # Light Goldenrod Yellow
        (216, 191, 216),  # Thistle
        (255, 240, 245),  # Lavender Blush
    ]

    final_pallete = np.empty((N, 3), dtype=np.uint8)
    for i in range(N):
        if i < len(DEFAULT_PALLETE):
            final_pallete[i] = np.asarray(DEFAULT_PALLETE[i])
        else:
            # Generate a random pastel color (BGR format)
            random_color = np.random.randint(128, 256, size=3)  # Pastel shades are closer to white
            final_pallete[i] = random_color
    return final_pallete

def get_blended_image(image_a:np.ndarray, image_b:np.ndarray, alpha:float=0.5):
    """
    INPUT: raw_image is image_a and semantic mask colored is image_b 
    OUPUT: blended image
    """
    if image_a.shape != image_b.shape:
        raise ValueError("Las imágenes deben tener las mismas dimensiones y número de canales.")
    
    blended_image = cv2.addWeighted(image_a, alpha, image_b, 1 - alpha, 0)
    return blended_image

def get_pcds_of_semantic_label(instance_pcds:dict, semantic_labels:list = None):
    pcds = []
    for semantic_pcd in instance_pcds:
        # skip if semantic_label is set and is not equal to the object label
        if semantic_labels is not None and semantic_pcd['label'] not in semantic_labels:
            continue 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(semantic_pcd['pcd'])
        pcd.colors = o3d.utility.Vector3dVector(semantic_pcd['pcd_colors'])
        pcds.append(pcd)
    return pcds

def intersection_factor(mask1, mask2):
    """
    Jaccard index based 
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0 # If there is no union 
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

DEFAULT_MERGE_DICT = {
    'vehicle.car': [
        "vehicle.bus.bendy", 
        "vehicle.bus.rigid", 
        "vehicle.car", 
        "vehicle.construction", 
        "vehicle.emergency.ambulance", 
        "vehicle.emergency.police", 
        "vehicle.trailer", 
        "vehicle.truck"
    ],
    "vehicle.motorcycle":[
        "vehicle.bicycle",
        "vehicle.motorcycle"
    ]
}

def merge_semantic_labels(semantic_mask, label2id, merge_dict:dict = DEFAULT_MERGE_DICT):
    # Merge labels
    for k, vals in merge_dict.items():
        to_id   = label2id[k]
        for v in vals:
            from_id = label2id[v]
            semantic_mask[semantic_mask == from_id] = to_id
    return semantic_mask


def AABB_intersect(A: dict, B:dict) -> bool:
    class AABB:
        def __init__(self, center: tuple, dims: tuple):
            self.minX = center[0] - dims[0]
            self.maxX = center[0] + dims[0]
            self.minY = center[1] - dims[1]
            self.maxY = center[1] + dims[1]
            self.minZ = center[2] - dims[2]
            self.maxZ = center[2] + dims[2]
    A = AABB(A['center'], A['dimensions'])
    B = AABB(B['center'], B['dimensions'])
    return A.minX <= B.maxX and A.maxX >= B.minX and A.minY <= B.maxY and A.maxY >= B.minY and A.minZ <= B.maxZ and A.maxZ >= B.minZ

def AABB_A_bigger_than_B(A: dict, B:dict) -> bool:
    A_dims = A['dimensions']
    B_dims = B['dimensions']
    return A_dims[0] * A_dims[1] * A_dims[2] >= B_dims[0] * B_dims[1] * B_dims[2]

def filter_instances(instance_pcds:dict, min_samples_per_instance:int = 150, max_distance:float = 15.0, max_height:float = 2.0, verbose=False):
    for semantic_pcd in instance_pcds:
        if not semantic_pcd['dynamic']:
            continue # Skip if non dynamic
        
        # Remove noise pcds from list
        for i, aux in enumerate(semantic_pcd['instance_pcds']):
            if aux['inst_id'] == -1:
                    semantic_pcd['instance_pcds'].pop(i)
        assert len(semantic_pcd['instance_pcds']) == len(semantic_pcd['instance_3dboxes'])
    
        # Remove far bboxes and pcds with few points
        added_AABBs = []
        removing_indices = []
        for i in range(len(semantic_pcd['instance_pcds'])):
            num_samples = semantic_pcd['instance_pcds'][i]['pcd'].shape[0]
            A = semantic_pcd['instance_3dboxes'][i] # Cuboid for instance pcd
            height = abs(A['center'][1])
            
            # Do not add the instance if it has few samples or the y position is above threshold
            if num_samples < min_samples_per_instance or height > max_height:
                removing_indices.append(i)
                continue
            
            # Do not add the instance if it is far away
            distance = np.linalg.norm( A['center'] )
            if distance > max_distance:
                removing_indices.append(i)
                continue
            # Do not add the instance if it's cuboid A intersects with an already added instance B 
            # Check if A is bigger than B. If its the case, remove B and add A
            replace_index = None
            for bbox_index in added_AABBs:
                B = semantic_pcd['instance_3dboxes'][bbox_index]
                if AABB_intersect(A, B):
                    if AABB_A_bigger_than_B(A, B):
                        replace_index = bbox_index # Replace B with A
                    replace_index = -1 # Dont add A
                    break
            # The instance has an intersection
            if replace_index is not None:
                if replace_index == -1:
                    # Do not add the current instance
                    removing_indices.append(i)
                    continue
                # Remove the B instance
                removing_indices.append(replace_index)
                added_AABBs.pop( added_AABBs.index(replace_index) )
                    
            # Add the instance
            added_AABBs.append(i)
        
        if verbose:
            print(f"class: {semantic_pcd['label']} removing indices: {removing_indices}")
        semantic_pcd['instance_pcds']       = [valor for idx, valor in enumerate(semantic_pcd['instance_pcds'])     if idx not in removing_indices]
        semantic_pcd['instance_3dboxes']    = [valor for idx, valor in enumerate(semantic_pcd['instance_3dboxes'])  if idx not in removing_indices]
    
    return instance_pcds

def create_cuboid_edges(center, 
                        dims, 
                        color = np.array([0.0, 1.0, 0.0]), 
                        transform_4x4: np.ndarray = None, 
                        initial_traslation_4x1:np.ndarray=None) -> o3d.geometry.LineSet:
    # Desglosamos el centro y las dimensiones
    cx, cy, cz = center # x, y, z
    w, h, d = dims # width, height, depth
    
    # Definimos los vértices de un cuboide centrado en 'center' con dimensiones 'width', 'height', 'depth'
    vertices_8x3 = np.array([
        [cx - w/2, cy - h/2, cz - d/2],  # Vértice 0
        [cx + w/2, cy - h/2, cz - d/2],  # Vértice 1
        [cx + w/2, cy + h/2, cz - d/2],  # Vértice 2
        [cx - w/2, cy + h/2, cz - d/2],  # Vértice 3
        [cx - w/2, cy - h/2, cz + d/2],  # Vértice 4
        [cx + w/2, cy - h/2, cz + d/2],  # Vértice 5
        [cx + w/2, cy + h/2, cz + d/2],  # Vértice 6
        [cx - w/2, cy + h/2, cz + d/2]   # Vértice 7
    ])

    if transform_4x4 is not None:
        initial_traslation_4x1 = np.zeros((4, 1)) if initial_traslation_4x1 is None else initial_traslation_4x1
        vertices_4x8 = utils.add_homogeneous_row(vertices_8x3.T)
        vertices_trans_4x8 = transform_4x4 @ vertices_4x8 - initial_traslation_4x1
        vertices_8x3 = vertices_trans_4x8[:3].T
    
    # Definir las aristas del cuboide, donde cada par de índices representa una línea
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Aristas de la cara inferior
        [4, 5], [5, 6], [6, 7], [7, 4],  # Aristas de la cara superior
        [0, 4], [1, 5], [2, 6], [3, 7]   # Aristas verticales
    ])
    
    # Crear un objeto LineSet para dibujar las aristas
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices_8x3)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.paint_uniform_color(color)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(edges))
    
    # cube = o3d.geometry.TriangleMesh.create_box(width=0.005, height=0.005, depth=0.005)
    # cube.translate([cx, cy, cz])
    # cube.paint_uniform_color([0,0,0])
    return line_set

def create_plane_at_y(y, size:int = 5):
    vertices = np.array([
        [-size, -y, -size],
        [size,  -y, -size],
        [-size, -y,  size],
        [size,  -y,  size]])
    faces = np.array([
        [0, 1, 2],
        [2, 1, 0],
        [3, 2, 1],
        [1, 2, 3]])
    
    # Crear la malla de triángulos
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(vertices)
    plane.triangles = o3d.utility.Vector3iVector(faces)
    plane.paint_uniform_color([0.5, 0.5, 0.5])
    
    return plane


# Function to parse MTL file and extract material colors
def parse_mtl(mtl_file):
    materials = {}
    current_material = None
    
    with open(mtl_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "newmtl":  # Start of a new material
                current_material = parts[1]
                materials[current_material] = {"Kd": [1.0, 1.0, 1.0]}  # Default color
            elif parts[0] == "Kd" and current_material:  # Diffuse color
                materials[current_material]["Kd"] = list(map(float, parts[1:4]))

    return materials

# Function to parse OBJ and extract face-material associations
def parse_obj_with_materials(obj_file):
    vertices = []
    faces = []
    face_materials = []
    current_material = None
    
    with open(obj_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == "v":  # Vertex
                vertices.append(list(map(float, parts[1:4])))
            elif parts[0] == "usemtl":  # Material assignment
                current_material = parts[1]
            elif parts[0] == "f":  # Face
                face = [int(p.split("/")[0]) - 1 for p in parts[1:4]]  # Only vertex indices
                faces.append(face)
                face_materials.append(current_material)  # Store associated material
    
    return np.array(vertices), np.array(faces), face_materials