import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

n_points = 1000
points = np.random.rand(n_points, 3)  # Nube de puntos en el espacio 3D

# Crear un objeto PointCloud y asignarle los puntos generados
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualizar la nube de puntos
# o3d.visualization.draw_geometries([pcd])

# Con Matplotlib
xyz = np.asarray(pcd.points)

# Crear una figura y un conjunto de ejes 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la nube de puntos
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=10, c='r', marker='o')

# Etiquetas y título
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Nube de Puntos 3D')

# Mostrar el gráfico
plt.show()