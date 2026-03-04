import logging
import time
from pathlib import Path
try:
    import pycolmap
except ImportError:
    pycolmap = None

from sigma.config import SfmConfig

logger = logging.getLogger(__name__)

class ColmapRunner:
    def __init__(self, config: SfmConfig):
        self.config = config
        
        if pycolmap is None:
            raise ImportError("pycolmap is not installed. Please install it with 'pip install pycolmap>=0.6.0' or via conda on the server.")

    def run(self, image_dir: Path, output_dir: Path) -> Path:
        """
        Run the full COLMAP pipeline: Feature extraction -> Matching -> Reconstruction -> Convert to 3DGS format.
        """
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        database_path = output_dir / "database.db"

        # 1. Feature Extraction
        logger.info(f"Extracting features using SIFT (device: {'GPU' if self.config.use_gpu else 'CPU'})...")
        pycolmap.extract_features(database_path, image_dir, sift_options={"use_gpu": self.config.use_gpu})
        
        # 2. Feature Matching
        logger.info(f"Matching features ({self.config.matching_type})...")
        if self.config.matching_type == "sequential":
            pycolmap.match_sequential(database_path, overlap=self.config.match_overlap, sift_options={"use_gpu": self.config.use_gpu})
        elif self.config.matching_type == "exhaustive":
            pycolmap.match_exhaustive(database_path, sift_options={"use_gpu": self.config.use_gpu})
        elif self.config.matching_type == "vocab_tree":
            # Requires vocab tree path, skipping for now or default
            logger.warning("Vocab tree matching not fully implemented. Falling back to sequential.")
            pycolmap.match_sequential(database_path, overlap=self.config.match_overlap, sift_options={"use_gpu": self.config.use_gpu})
            
        # 3. Incremental Reconstruction
        logger.info("Running incremental mapping...")
        maps = pycolmap.incremental_mapping(database_path, image_dir, output_dir)
        
        if not maps:
            logger.error("Reconstruction failed! No maps created.")
            raise RuntimeError("Reconstruction failed")
            
        # Select the largest reconstruction (usually map 0)
        reconstruction = maps[0]
        logger.info(f"Reconstruction successful. {reconstruction.num_points3D()} points, {reconstruction.num_images()} images.")
        
        # 4. Export for 3DGS (Standard COLMAP format)
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        reconstruction.write(sparse_dir)
        logger.info(f"Sparse reconstruction saved to {sparse_dir}")
        
        elapsed = time.time() - start_time
        logger.info(f"COLMAP pipeline completed in {elapsed:.2f} seconds.")
        
        return sparse_dir

if __name__ == "__main__":
    import argparse
    from sigma.config import SigmaConfig
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Run COLMAP pipeline for SIGMA")
    parser.add_argument("--image_dir", required=True, type=Path, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for COLMAP output")
    parser.add_argument("--config_path", default="configs/default.yaml", type=Path, help="Configuration file path")
    
    args = parser.parse_args()
    
    if not args.image_dir.exists():
        logger.error(f"Image directory {args.image_dir} does not exist.")
        exit(1)
        
    try:
        config = SigmaConfig.from_yaml(args.config_path)
        runner = ColmapRunner(config.sfm)
        runner.run(args.image_dir, args.output_dir)
    except Exception as e:
        logger.error(f"Error running COLMAP: {e}")
        exit(1)
