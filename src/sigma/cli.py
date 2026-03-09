import typer
import logging
from pathlib import Path

from sigma.config import SigmaConfig
from sigma.preprocessor.video import VideoPreprocessor
from sigma.gaussian_splatting.converter import GaussianToPointCloud
from sigma.map_generator.map_2d import Map2DGenerator
from sigma.map_generator.map_3d import Map3DGenerator
from sigma.visualization.viewer import SigmaViewer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SIGMA")

app = typer.Typer(help="SIGMA: Site Imagery to Geometric Map Automation CLI")

@app.command()
def extract_frames(
    video: Path = typer.Option(..., help="Path to input video file"),
    output: Path = typer.Option(..., help="Output directory for extracted frames"),
    config_path: Path = typer.Option("configs/default.yaml", "--config", help="Path to configuration file")
):
    """
    Extract frames from video (blur/duplicate filtering included).
    """
    if not video.exists():
        typer.echo(f"Error: Video file {video} not found", err=True)
        raise typer.Exit(code=1)
        
    config = SigmaConfig.from_yaml(config_path)
    preprocessor = VideoPreprocessor(config.preprocessor)
    
    preprocessor.extract_frames(video, output)

@app.command()
def generate_map(
    model: Path = typer.Option(..., help="Path to trained 3DGS model (point_cloud.ply)"),
    output: Path = typer.Option(..., help="Output directory for generated maps"),
    config_path: Path = typer.Option("configs/default.yaml", "--config", help="Path to configuration file")
):
    """
    Generate 2D floorplan and 3D export from trained model.
    """
    if not model.exists():
        typer.echo(f"Error: Model file {model} not found", err=True)
        raise typer.Exit(code=1)
        
    config = SigmaConfig.from_yaml(config_path)
    
    # 1. Convert/Load Point Cloud
    converter = GaussianToPointCloud()
    pcd = converter.convert(model)
    
    output.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate 2D Map
    map_gen = Map2DGenerator(config.map_2d)
    grid = map_gen.generate(pcd)
    map_path = output / "occupancy_map.png"
    map_gen.export_image(grid, map_path)
    
    # 3. Export 3D Map
    map_3d_gen = Map3DGenerator(config.map_3d)
    map_3d_path = output / f"model_export.{config.map_3d.export_format}"
    map_3d_gen.export(pcd, map_3d_path)
    
    typer.echo("Map generation complete.")

@app.command()
def run_all(
    video: Path = typer.Option(..., help="Path to input video file"),
    output: Path = typer.Option(..., help="Base output directory for all pipeline results"),
    config_path: Path = typer.Option("configs/default.yaml", "--config", help="Path to configuration file")
):
    """
    Run the full pipeline on a single machine (server).
    Executes: extract-frames → COLMAP → 3DGS training → generate-map.
    """
    from sigma.sfm.colmap_runner import ColmapRunner
    from sigma.gaussian_splatting.trainer import GaussianSplattingTrainer

    if not video.exists():
        typer.echo(f"Error: Video file {video} not found", err=True)
        raise typer.Exit(code=1)

    config = SigmaConfig.from_yaml(config_path)
    output.mkdir(parents=True, exist_ok=True)

    frames_dir = output / "frames"
    colmap_dir = output / "colmap_out"
    gs_model_dir = output / "gs_model"
    maps_dir = output / "maps"

    # Step 1: Extract frames
    logger.info("=" * 60)
    logger.info("Step 1/4: Extracting frames from video...")
    logger.info("=" * 60)
    preprocessor = VideoPreprocessor(config.preprocessor)
    preprocessor.extract_frames(video, frames_dir)

    # Step 2: COLMAP (SfM)
    logger.info("=" * 60)
    logger.info("Step 2/4: Running COLMAP (Structure-from-Motion)...")
    logger.info("=" * 60)
    colmap_runner = ColmapRunner(config.sfm)
    sparse_dir = colmap_runner.run(frames_dir, colmap_dir)

    # Step 3: 3DGS Training
    logger.info("=" * 60)
    logger.info("Step 3/4: Training 3D Gaussian Splatting...")
    logger.info("=" * 60)
    trainer = GaussianSplattingTrainer(config.gaussian_splatting)
    scene = trainer.load_scene(sparse_dir)
    model_path = trainer.train(scene, gs_model_dir)

    # Step 4: Generate maps
    logger.info("=" * 60)
    logger.info("Step 4/4: Generating maps...")
    logger.info("=" * 60)
    converter = GaussianToPointCloud()
    pcd = converter.convert(model_path)

    maps_dir.mkdir(parents=True, exist_ok=True)
    map_gen = Map2DGenerator(config.map_2d)
    grid = map_gen.generate(pcd)
    map_gen.export_image(grid, maps_dir / "occupancy_map.png")

    map_3d_gen = Map3DGenerator(config.map_3d)
    map_3d_gen.export(pcd, maps_dir / f"model_export.{config.map_3d.export_format}")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Frames:  {frames_dir}")
    logger.info(f"  COLMAP:  {colmap_dir}")
    logger.info(f"  Model:   {gs_model_dir}")
    logger.info(f"  Maps:    {maps_dir}")
    logger.info("=" * 60)

@app.command()
def view_2d(map_path: Path = typer.Option(..., "--map", help="Path to 2D map image")):
    viewer = SigmaViewer()
    viewer.view_2d(map_path)

@app.command()
def view_3d(model_path: Path = typer.Option(..., "--model", help="Path to 3D model ply")):
    viewer = SigmaViewer()
    viewer.view_3d(model_path)

if __name__ == "__main__":
    app()
