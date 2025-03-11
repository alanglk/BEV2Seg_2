import open3d as o3d
import numpy as np

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

# Load materials
mtl_file = "./assets/carlota_3d/carlota_mesh.mtl"
materials = parse_mtl(mtl_file)

# Load OBJ file with material information
obj_file = "./assets/carlota_3d/carlota_mesh.obj"
vertices, faces, face_materials = parse_obj_with_materials(obj_file)

# Assign colors per vertex (by averaging face colors affecting each vertex)
vertex_colors = np.ones((len(vertices), 3))  # Default white

# Map from vertices to the colors of the faces they belong to
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
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# Compute normals for better visualization
mesh.compute_vertex_normals()

mesh.translate(np.array([10, 0, 0]))

# Visualize
o3d.visualization.draw_geometries([mesh], window_name="Carlota Model")
