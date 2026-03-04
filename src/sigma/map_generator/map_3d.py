import logging
from pathlib import Path
try:
    import open3d as o3d
except ImportError:
    o3d = None

from sigma.config import Map3DConfig

logger = logging.getLogger(__name__)

class Map3DGenerator:
    def __init__(self, config: Map3DConfig):
        self.config = config

    def export(self, pcd: "o3d.geometry.PointCloud", output_path: Path):
        """
        Export Point Cloud to 3D Mesh format (PLY, OBJ, GLB).
        """
        if o3d is None:
            logger.warning("Open3D not installed. 3D export limited.")
            return

        fmt = self.config.export_format
        
        if fmt == "ply":
            o3d.io.write_point_cloud(str(output_path), pcd)
            logger.info(f"Exported Point Cloud to {output_path}")
            
        elif fmt == "obj":
            # Attempt mesh generation
            logger.info("Attempting Poisson Reconstruction for Mesh generation...")
            if not pcd.has_normals():
                pcd.estimate_normals()
                
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            logger.info(f"Exported Mesh to {output_path}")
            
        elif fmt == "glb":
            # Open3D support for GLB is limited or requires specific version
            # Fallback to PLY or warn
            logger.warning("GLB export might not be fully supported. Saving as .ply instead.")
            o3d.io.write_point_cloud(str(output_path.with_suffix(".ply")), pcd)
