import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider

# ------------------ Funciones auxiliares ------------------

def generate_heatmap_data(size=(10, 10)):
    """Genera datos aleatorios para el heatmap."""
    return np.random.rand(*size)

def get_cube_vertices(center, size):
    """Genera los vértices de un cubo 3D dado su centro y tamaño."""
    cx, cy, cz = center
    sx, sy, sz = size
    x = [-sx/2, sx/2]
    y = [-sy/2, sy/2]
    z = [-sz/2, sz/2]
    
    vertices = np.array([[cx + i, cy + j, cz + k] for i in x for j in y for k in z])
    faces = [[vertices[i] for i in face] for face in [
        [0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [0,2,6,4], [1,3,7,5]
    ]]
    return faces

def plot_cuboid(ax, center, size, color='b', alpha=0.3):
    """Dibuja un cuboide en una figura 3D."""
    faces = get_cube_vertices(center, size)
    poly3d = Poly3DCollection(faces, alpha=alpha, edgecolor="k")
    poly3d.set_facecolor(color)
    ax.add_collection3d(poly3d)

def compute_iou_and_distance(cube1, cube2):
    """Calcula un IoU y una distancia ficticia entre dos cuboides."""
    iou = np.random.uniform(0, 1)  # Placeholder
    distance = np.linalg.norm(np.array(cube1) - np.array(cube2))  # Distancia Euclideana
    return iou, distance

# ------------------ Configuración de la figura principal ------------------

fig = plt.figure(figsize=(12, 6))

# Heatmap
ax1 = fig.add_subplot(2, 2, 1)
heatmap_data = generate_heatmap_data()
ax1.imshow(heatmap_data, cmap=cm.coolwarm, interpolation="nearest")
ax1.set_title("Heatmap")

# Espacio en blanco
ax2 = fig.add_subplot(2, 2, 2)
ax2.axis("off")
ax2.set_title("Espacio en blanco")

# Figuras 3D
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.set_title("IoU entre cuboides")
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
ax4.set_title("Distancia entre cuboides")

# ------------------ Configuración de datos ------------------

n = 10  # Número de elementos controlados por el slider
cuboids = [((np.random.rand(3) * 5, np.random.rand(3) + 0.5),
            (np.random.rand(3) * 5, np.random.rand(3) + 0.5)) for _ in range(n)]

# ------------------ Función de actualización del slider ------------------

def update(val):
    idx = int(slider.val)
    ax3.cla()
    ax4.cla()

    ax3.set_title("IoU entre cuboides")
    ax4.set_title("Distancia entre cuboides")

    cube1, cube2 = cuboids[idx]
    
    # Plot en la figura de IoU
    plot_cuboid(ax3, cube1[0], cube1[1], color='r', alpha=0.4)
    plot_cuboid(ax3, cube2[0], cube2[1], color='b', alpha=0.4)
    
    # Plot en la figura de Distancia
    plot_cuboid(ax4, cube1[0], cube1[1], color='g', alpha=0.4)
    plot_cuboid(ax4, cube2[0], cube2[1], color='purple', alpha=0.4)

    # Calcular métricas
    iou, dist = compute_iou_and_distance(cube1[0], cube2[0])
    ax3.set_xlabel(f"IoU: {iou:.2f}")
    ax4.set_xlabel(f"Distancia: {dist:.2f}")

    fig.canvas.draw_idle()

# ------------------ Slider ------------------

ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
slider = Slider(ax_slider, "Elemento", 0, n-1, valinit=0, valstep=1)
slider.on_changed(update)

# Inicializar visualización
update(0)

plt.tight_layout()
plt.show()
