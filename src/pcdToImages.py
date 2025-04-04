#!/bin/python3

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
def save_pointcloud_images(pcd_path, output_dir="output_images", views=4):
    """
    Carga un archivo .pcd, genera imágenes desde distintas perspectivas y las guarda en el directorio especificado.
    
    :param pcd_path: Ruta del archivo .pcd
    :param output_dir: Directorio donde se guardarán las imágenes
    :param views: Número de perspectivas diferentes
    """
    # Cargar la nube de puntos
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd:
        print("Error: No se pudo cargar el archivo PCD.")
        return
    
    # Filtrar los puntos cercanos al origen dentro de los límites dados
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    mask = (np.abs(points[:, 0]) <= 10) & (np.abs(points[:, 1]) <= 5) & (points[:, 2] <= 30)
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    
    # Transformar la nube de puntos para corregir la perspectiva de la cámara
    R = np.array([[1, 0, 0],
                  [0, -1, 0],  # Invertir el eje Y para coincidir con la convención de Open3D
                  [0, 0, -1]]) # Invertir el eje Z para que apunte hacia adelante
    pcd.rotate(R, center=(0, 0, 0))
    
    # Scene centroid
    centroid = np.mean(points, axis=0) + np.array([0, 0, -10])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.translate(centroid - np.array([0, 0, 0]))  # Trasladar la esfera al centroide
    sphere.paint_uniform_color([1, 0, 0])  # Pinta la esfera de color rojo

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear una ventana de visualización
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    # vis.add_geometry(sphere)
    
    # Configurar la cámara
    view_control = vis.get_view_control()
    render_options = vis.get_render_option()
    render_options.point_size = 3.0  # Ajustar el tamaño de los puntos para mejor visualización
    
    # Definir posiciones de la cámara en una órbita alrededor de la nube de puntos
    radius      = 20  # Distancia de la cámara al centro de la nube
    y_height    = 10
    for i in range(views):
        theta = (2 * np.pi / views) * i  # Ángulo en el plano XY
        cam_pos = [
            radius * np.cos(theta),
            radius * np.sin(theta),
        ]
        view_control.set_lookat(centroid)
        view_control.set_front([cam_pos[1] + centroid[0], y_height + centroid[1], cam_pos[0] + centroid[2]])
        view_control.set_up([0, 1, 0])
        
        # Capturar imagen
        image_path = os.path.join(output_dir, f"view_{i}.png")
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(image_path)
        print(f"Imagen guardada: {image_path}")
    
    vis.destroy_window()

# Uso del script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera imágenes de una nube de puntos .pcd desde distintas perspectivas.")
    parser.add_argument("pcd_path", type=str, help="Ruta del archivo .pcd")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directorio donde se guardarán las imágenes")
    parser.add_argument("--views", type=int, default=4, help="Número de perspectivas diferentes")
    
    args = parser.parse_args()
    save_pointcloud_images(args.pcd_path, args.output_dir, args.views)
