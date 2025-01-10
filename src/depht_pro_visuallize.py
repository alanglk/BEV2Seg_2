import open3d as o3d
import pickle
import numpy as np
import sys
import os
import re
from tqdm import tqdm


#pcd = o3d.io.read_point_cloud(file_path)
#points = np.asarray(pcd.points)
#points[:, 0], points[:, 1], points[:, 2] = points[:, 2], -points[:, 0], -points[:, 1] # x -> z | y -> -x | z -> -y
#pcd.points = o3d.utility.Vector3dVector(points)


data_folder = None
if len(sys.argv) > 1:
    data_folder = sys.argv[1]
    print(f"Leyendo data_folder: {data_folder}")
else:
    print("No se ha proporcionado ningún path.")


instances_path  = os.path.join(data_folder, "output", "pointcloud", "instances")
semantic_path   = os.path.join(data_folder, "output", "pointcloud", "semantic")
cluster_path    = os.path.join(data_folder, "output", "pointcloud", "cluster")

def get_geometry_data(geometry_path:str):
    geometry_name = os.path.basename(geometry_path)
    # regex = r"([^/]+)-([^/]+)-(.*)-(-?\d+)\.*"
    regex = r"([^/]+)-([^/]+)-(.*)\.*"
    match = re.search(regex, geometry_name)
    
    if match is not None:
        token = match.group(1)
        label_name = match.group(2)
        camera_name = match.group(3)
        return token, label_name, camera_name
    
    raise Exception(f"{geometry_name} doesnt match regex expression")


def create_cuboid_edges(center, dims, color = (0.0, 1.0, 0.0)):
    # Desglosamos el centro y las dimensiones
    cx, cy, cz = center # x, y, z
    w, h, d = dims # width, height, depth
    
    # Definimos los vértices de un cuboide centrado en 'center' con dimensiones 'width', 'height', 'depth'
    vertices = np.array([
        [cx - w/2, cy - h/2, cz - d/2],  # Vértice 0
        [cx + w/2, cy - h/2, cz - d/2],  # Vértice 1
        [cx + w/2, cy + h/2, cz - d/2],  # Vértice 2
        [cx - w/2, cy + h/2, cz - d/2],  # Vértice 3
        [cx - w/2, cy - h/2, cz + d/2],  # Vértice 4
        [cx + w/2, cy - h/2, cz + d/2],  # Vértice 5
        [cx + w/2, cy + h/2, cz + d/2],  # Vértice 6
        [cx - w/2, cy + h/2, cz + d/2]   # Vértice 7
    ])

    # Definir las aristas del cuboide, donde cada par de índices representa una línea
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Aristas de la cara inferior
        [4, 5], [5, 6], [6, 7], [7, 4],  # Aristas de la cara superior
        [0, 4], [1, 5], [2, 6], [3, 7]   # Aristas verticales
    ])

    # Crear un objeto LineSet para dibujar las aristas
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.paint_uniform_color(color)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(edges))
    

    cube = o3d.geometry.TriangleMesh.create_box(width=0.005, height=0.005, depth=0.005)
    cube.translate([cx, cy, cz])
    cube.paint_uniform_color([0,0,0])

    return line_set

def get_bboxes_of_semantic_label(instances_path:str, image_token:str, semantic_labels:list = None):
    desc = f"Loading bboxes of {semantic_labels}" if semantic_labels is not None else "Loading bboxes"
    file_name = os.path.join(instances_path, f"{image_token}.plk")
    all_bboxes = []
    
    with open(file_name, "rb") as f:
        instance_data = pickle.load(f)
        if instance_data is None:
            raise Exception(f"Loaded None from {file_name}")

        for seg in tqdm(instance_data, desc=desc):
            # skip if semantic_label is set and is not equal to the object label
            if semantic_labels is not None and seg['label'] not in semantic_labels:
                continue 
            
            # skip non dynamic objects
            if not seg['dynamic']:
                continue
            
            # Read 3d_bboxes
            for bbox_data in seg['instance_3dboxes']:
                print(bbox_data)
                bbox = create_cuboid_edges(bbox_data['center'], bbox_data['dimensions'], color=(0.0, 1.0, 0.0))
                all_bboxes.append(bbox) 
            
    print(f"{len(all_bboxes)} instance_3dboxes loaded")
    return all_bboxes

def get_pcd_of_semantic_label(semantic_pcd_path:str, image_token:str, semantic_labels:list = None):
    desc = f"Loading semantic pcds of: {semantic_labels}" if semantic_labels is not None else "Loading semantic pcds"
    file_name = os.path.join(semantic_pcd_path, f"{image_token}.plk")
    pcds = []
    
    with open(file_name, "rb") as f:
        semantic_data = pickle.load(f)
        if semantic_data is None:
            raise Exception(f"Loaded None from {file_name}")

        for seg in tqdm(semantic_data, desc=desc):
            # skip if semantic_label is set and is not equal to the object label
            if semantic_labels is not None and seg['label'] not in semantic_labels:
                continue 
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(seg['pcd'])
            pcd.colors = o3d.utility.Vector3dVector(seg['pcd_colors'])
            pcds.append(pcd)

    print(f"{len(pcds)} loaded pointclouds")
    return pcds


image_token     = "60d367ec0c7e445d8f92fbc4a993c67e" 
image_token     = "0a1fca1d93d04f60a4b12961a22310bb" 

# Load Geometries
bboxes = get_bboxes_of_semantic_label(instances_path, image_token, semantic_labels=["vehicle.car"])
#pcds = get_pcd_of_semantic_label(cluster_path, image_token, semantic_labels=["vehicle.car", "flat.driveable_surface"])
pcds = get_pcd_of_semantic_label(semantic_path, image_token, semantic_labels=["vehicle.car", "flat.driveable_surface"])


# Mostrar la nube de puntos
all_geometries = pcds + bboxes
print(all_geometries)
o3d.visualization.draw_geometries(all_geometries, window_name="Point Cloud Viewer")