import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from typing import Union

class Ray:
    def __init__(self, x:float, y:float, alpha:float):
        self.o = np.array([x, y])
        self.dir = np.array([ np.cos(alpha), np.sin(alpha) ])
        self.point_col = (1.0, 0.0, 0.0, 1.0)
        self.line_col = (1.0, 0.5, 0.0, 1.0)
        self.after_inter_col = (0.05, 0.05, 0.05, 1.0)

    def p(self, t):
        return self.o + t * self.dir

    def render(self,  ax: Axes, step:float, max_range:float):
        n = int(np.round(max_range / step))
        xs, ys = [], []
        for i in range(n):
            pt = self.p(step * i)
            xs.append(pt[0])
            ys.append(pt[1])

        #ax.scatter(xs, ys, color=self.point_col)
        ax.plot(xs, ys, color=self.line_col)

    def render_intersection(self, ax: Axes, intersection:np.ndarray):
        if intersection.shape[0] > 0:
            ax.plot(intersection[:, 0], intersection[:, 1] , color=self.point_col)
            ax.scatter(intersection[[0, -1], 0], intersection[[0, -1], 1], color=self.point_col)

    def render_after_intersection(self, ax:Axes, intersection:np.ndarray, max_range:float, step:float):
        if intersection.shape[0] > 0:
            # t_inter = np.linalg.norm(intersection[[0, -1]] - self.o / self.dir )
            pt_inter = intersection[-1]
            pt_last = self.p(max_range)
            
            x_values = [pt_inter[0], pt_last[0]]
            y_values = [pt_inter[1], pt_last[1]]
            ax.plot(x_values, y_values, color=self.after_inter_col)

class Circle:
    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r
        self.point_col = (0.0, 1.0, 0.0, 1.0)
        self.line_col = (0.0, 1.0, 0.5, 1.0)
    
    def p(self, t):
        return np.array([
            self.x + self.r * np.cos(t),
            self.y + self.r * np.sin(t),
        ])
    
    def render(self,  ax: Axes, step:float, render_points=False):
        patch = plt.Circle((self.x, self.y), self.r, color=self.line_col, fill=False)
        ax.add_patch(patch)
        if not render_points:
            return
        
        n = int(np.round(2*np.pi / step))
        xs, ys = [], []
        for i in range(n):
            pt = self.p(step * i)
            xs.append(pt[0])
            ys.append(pt[1])
        ax.scatter(xs, ys, color=self.point_col)

class Box:
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.edge_col = (0.0, 0.0, 1.0, 1.0)
        self.line_col = (0.0, 0.5, 1.0, 1.0)
        self.compute_corners()

    def compute_corners(self):
        """ Calcula las esquinas del rectángulo basado en (x1, y1) y (x2, y2). """
        self.corners = np.array([
            [self.x1, self.y1],  # Esquina inferior izquierda
            [self.x2, self.y1],  # Esquina inferior derecha
            [self.x2, self.y2],  # Esquina superior derecha
            [self.x1, self.y2]   # Esquina superior izquierda
        ])

    def render(self, ax: Axes):
        """ Dibuja la caja en el gráfico. """
        corners = np.vstack([self.corners, self.corners[0]])  # Cierra el polígono
        ax.plot(corners[:, 0], corners[:, 1], color=self.line_col)
        ax.scatter(corners[:, 0], corners[:, 1], color=self.edge_col)

    def _intersects(self, point:np.ndarray):
        px, py = point[0], point[1]
        # AABB
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def get_intersection_point(self, ray:Ray, step:float, max_range:float) -> Union[None, np.ndarray]:
        n = int(np.round(max_range / step))
        intersection = np.empty((0, 2))

        for i in range(n):
            pt = ray.p(step * i)
            if self._intersects(pt):
                intersection = np.vstack([intersection, pt])
        return intersection

class Oriented2DBox(Box):
    def __init__(self, x1: float, y1: float, x2: float, y2: float, alpha: float):
        super().__init__(x1, y1, x2, y2)
        self.alpha = alpha
        self.rotate(self.alpha)

    def rotate(self, alpha):
        """ Rota la caja respecto a su centro. """
        # Calcular el centro
        center = np.mean(self.corners, axis=0)
        
        # Rotation matrix
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        self.corners = (self.corners - center) @ rotation_matrix.T + center
    
    def rotate_points(self, alpha, points) -> np.ndarray:
        center = np.mean(self.corners, axis=0)
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        return (points - center) @ rotation_matrix.T + center

    def _intersects(self, point:np.ndarray):
        px, py = point[0], point[1]
        
        # Check if point is inside using cross products
        def is_same_side(p1, p2, a, b):
            """Returns True if p1 and p2 are on the same side of the line AB."""
            cross1 = np.cross(b - a, p1 - a)
            cross2 = np.cross(b - a, p2 - a)
            return np.sign(cross1) == np.sign(cross2)

        A, B, C, D = self.corners  # Assign corners in correct order

        # The point must be on the **same side** of all edges to be inside
        return (
            is_same_side(point, A, B, C) and
            is_same_side(point, B, C, D) and
            is_same_side(point, C, D, A) and
            is_same_side(point, D, A, B)
        )
    
    def get_intersection_point(self, ray:Ray, step:float, max_range:float) -> Union[None, np.ndarray]:
        # 1. transform box to the source frame
        aux_corners = self.corners.copy()
        self.rotate(-self.alpha)
        
        # 2. compute intersection in the source frame
        n = int(np.round(max_range / step))
        intersection = np.empty((0, 2))

        
        for i in range(n):
            pt = ray.p(step * i)
            aux_pt = self.rotate_points(-self.alpha, pt)
            if self._intersects(aux_pt):
                intersection = np.vstack([intersection, pt])
        
        # 3. transform box to its frame
        self.corners = aux_corners
        return intersection
        
def main():
    cir_step = 0.01
    ray_step = 0.001
    max_range = 4
    
    cir1 = Circle(0.0, 0.0, max_range)

    # box = Box(1, 1, 2, 2)
    box = Oriented2DBox(1, 1, 2, 2, np.pi/4)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    cir1.render(ax, cir_step)
    box.render(ax)

    n = int(np.round((np.pi / 4) / cir_step))
    xs, ys = [], []
    for i in range(n):
        x, y = cir1.p(cir_step * i)
        xs.append(x)
        ys.append(y)
        
        ray_angle = np.arctan2(y, x) # toward the origin
        ray = Ray(0.0, 0.0, ray_angle)
        intersection = box.get_intersection_point(ray, ray_step, max_range)
        ray.render(ax, ray_step, max_range)
        ray.render_intersection(ax, intersection)
        ray.render_after_intersection(ax, intersection, max_range, ray_step)
    ax.scatter(xs, ys, color=cir1.point_col)
    
    
    # ax.set_xbound(-max_range, max_range)
    # ax.set_ybound(-max_range, max_range)
    plt.show()



if __name__ == "__main__":
    main()