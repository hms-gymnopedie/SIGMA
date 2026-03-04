import numpy as np
import cv2
import logging
from pathlib import Path
from sigma.config import Map2DConfig

try:
    import open3d as o3d
except ImportError:
    o3d = None

logger = logging.getLogger(__name__)

class Map2DGenerator:
    def __init__(self, config: Map2DConfig):
        self.config = config

    def generate(self, pcd: "o3d.geometry.PointCloud") -> np.ndarray:
        """
        Generate 2D Occupancy Grid from Point Cloud.
        """
        points = np.asarray(pcd.points)
        
        # 1. Height Slicing
        z_min = self.config.z_slice_min
        z_max = self.config.z_slice_max
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        sliced_points = points[mask]
        
        logger.info(f"Sliced {len(points)} -> {len(sliced_points)} points (z: {z_min}~{z_max})")
        
        if len(sliced_points) == 0:
            logger.warning("No points in slice range!")
            return np.zeros((100, 100), dtype=np.uint8)

        # 2. BEV Projection
        # Determine bounds
        x_min, y_min = np.min(sliced_points[:, :2], axis=0)
        x_max, y_max = np.max(sliced_points[:, :2], axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        resolution = self.config.resolution
        
        grid_w = int(np.ceil(width / resolution)) + 1
        grid_h = int(np.ceil(height / resolution)) + 1

        logger.info(f"Grid Size: {grid_w} x {grid_h} (Resolution: {resolution}m)")

        # BEV projection via histogram2d
        H, _, _ = np.histogram2d(
            sliced_points[:, 1], sliced_points[:, 0],
            bins=[grid_h, grid_w],
            range=[[y_min, y_min + grid_h * resolution], [x_min, x_min + grid_w * resolution]]
        )
        
        # 3. Occupancy Grid
        # H is the count map. If count > threshold -> occupied
        occupancy_grid = (H > self.config.occupancy_threshold).astype(np.uint8) * 255
        
        # 4. Morphology (Cleanup)
        kernel_size = self.config.morphology_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Close gaps
        closed_grid = cv2.morphologyEx(occupancy_grid, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise (Open)
        opened_grid = cv2.morphologyEx(closed_grid, cv2.MORPH_OPEN, kernel)
        
        # 5. Boundary Detection
        if self.config.boundary_method == "canny":
            edges = cv2.Canny(opened_grid, 50, 150)
            return edges
        elif self.config.boundary_method == "contour":
            # Find contours and draw just the outlines
            contours, _ = cv2.findContours(opened_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = np.zeros_like(opened_grid)
            cv2.drawContours(contour_img, contours, -1, 255, 1)
            return contour_img
        
        return opened_grid

    def export_image(self, grid: np.ndarray, output_path: Path):
        cv2.imwrite(str(output_path), grid)
        logger.info(f"Saved 2D map to {output_path}")
