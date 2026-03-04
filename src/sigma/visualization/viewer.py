import logging
from pathlib import Path
try:
    import open3d as o3d
    import cv2
except ImportError:
    o3d = None
    cv2 = None

logger = logging.getLogger(__name__)

class SigmaViewer:
    def view_3d(self, model_path: Path):
        if o3d is None:
            logger.error("Open3D not installed.")
            return

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return

        pcd = o3d.io.read_point_cloud(str(model_path))
        o3d.visualization.draw_geometries([pcd], window_name=f"SIGMA 3D View: {model_path.name}")

    def view_2d(self, map_path: Path):
        if cv2 is None:
            logger.error("OpenCV not installed.")
            return
            
        img = cv2.imread(str(map_path))
        if img is None:
            logger.error(f"Could not read image {map_path}")
            return
            
        cv2.imshow(f"SIGMA 2D View: {map_path.name}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
