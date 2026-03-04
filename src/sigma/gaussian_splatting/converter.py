import numpy as np
from pathlib import Path
import logging
try:
    from plyfile import PlyData
    import open3d as o3d
except ImportError:
    PlyData = None
    o3d = None

logger = logging.getLogger(__name__)

class GaussianToPointCloud:
    def __init__(self, filter_opacity: float = 0.5):
        self.filter_opacity = filter_opacity
        if PlyData is None or o3d is None:
            logger.warning("plyfile/open3d not installed. Conversion will fail.")

    def convert(self, model_path: Path) -> "o3d.geometry.PointCloud":
        """
        Convert a 3DGS .ply file to an Open3D PointCloud.
        Filters points by opacity (sigmoid applied).
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path} not found")
            
        logger.info(f"Loading 3DGS model from {model_path}...")
        plydata = PlyData.read(str(model_path))
        
        xyz = np.stack((plydata.elements[0]['x'], 
                        plydata.elements[0]['y'], 
                        plydata.elements[0]['z']), axis=1)
        
        try:
            opacities = plydata.elements[0]['opacity']
        except ValueError:
            logger.warning("No 'opacity' field in PLY. Using all points without opacity filtering.")
            opacities = np.ones(len(xyz))

        # Apply sigmoid if values are outside [0, 1] (stored as logits in standard 3DGS)
        min_op = np.min(opacities)
        max_op = np.max(opacities)

        if min_op < 0 or max_op > 1:
            # Numerically stable sigmoid: avoid overflow in np.exp()
            activated_opacities = np.where(
                opacities >= 0,
                1 / (1 + np.exp(-opacities)),
                np.exp(opacities) / (1 + np.exp(opacities))
            )
        else:
            activated_opacities = opacities
            
        # Filter
        mask = activated_opacities > self.filter_opacity
        filtered_xyz = xyz[mask]
        
        logger.info(f"Filtered {len(xyz)} points -> {len(filtered_xyz)} points (opacity > {self.filter_opacity})")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
        
        # SH 0th order to RGB: RGB = SH_0 * C0 + 0.5
        C0 = 0.28209479177387814
        try:
            sh_R = plydata.elements[0]['f_dc_0'][mask]
            sh_G = plydata.elements[0]['f_dc_1'][mask]
            sh_B = plydata.elements[0]['f_dc_2'][mask]

            rgb = np.stack((sh_R, sh_G, sh_B), axis=1) * C0 + 0.5
            rgb = np.clip(rgb, 0, 1)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        except ValueError:
            logger.warning("No SH color fields (f_dc_*) in PLY. Point cloud will have no color.")
        
        return pcd
