"""
Integration tests that exercise real code paths with synthetic data (no mocking).
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from sigma.config import SigmaConfig, PreprocessConfig, Map2DConfig, Map3DConfig


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_load_default_yaml(self):
        """Load the real configs/default.yaml and verify parsed values."""
        config = SigmaConfig.from_yaml(Path("configs/default.yaml"))
        assert config.preprocessor.extraction_fps == 2
        assert config.sfm.feature_type == "sift"
        assert config.gaussian_splatting.iterations == 30000
        assert config.map_2d.resolution == 0.05
        assert config.map_3d.export_format == "ply"

    def test_config_defaults(self):
        """SigmaConfig without args should use all defaults."""
        config = SigmaConfig()
        assert config.preprocessor.blur_threshold == 100.0
        assert config.sfm.matching_type == "sequential"

    def test_config_from_dict(self):
        """SigmaConfig from partial dict merges with defaults."""
        config = SigmaConfig(**{"preprocessor": {"extraction_fps": 5}})
        assert config.preprocessor.extraction_fps == 5
        assert config.preprocessor.max_dimension == 1920  # default kept


# ---------------------------------------------------------------------------
# Preprocessor – real OpenCV, synthetic video
# ---------------------------------------------------------------------------

def _create_synthetic_video(path: Path, num_frames: int = 60, fps: float = 30.0,
                            width: int = 160, height: int = 120):
    """Write a tiny mp4 (or avi) video with colored frames."""
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback to MJPG if XVID codec unavailable
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 4) % 256, dtype=np.uint8)
        # Add a shifting rectangle so frames aren't identical
        x = (i * 5) % (width - 20)
        cv2.rectangle(frame, (x, 10), (x + 20, 40), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class TestPreprocessorReal:
    def test_extract_frames_real_video(self, tmp_path):
        """Extract frames from a real (synthetic) video file."""
        from sigma.preprocessor.video import VideoPreprocessor

        video_path = _create_synthetic_video(tmp_path / "test.avi", num_frames=60, fps=30.0)
        output_dir = tmp_path / "frames"

        config = PreprocessConfig(
            extraction_fps=2.0,
            max_dimension=200,
            blur_threshold=0,   # disable blur filter
            dedup_threshold=0,  # disable dedup
        )
        preprocessor = VideoPreprocessor(config)
        frames = preprocessor.extract_frames(video_path, output_dir)

        # 60 frames @ 30fps, extracting at 2fps -> interval 15 -> expect 4 frames (0,15,30,45)
        assert len(frames) == 4
        for fp in frames:
            assert fp.exists()
            img = cv2.imread(str(fp))
            assert img is not None
            # Should be resized to max_dimension=200 (already smaller, so unchanged)
            assert max(img.shape[:2]) <= 200

    def test_blur_filter_real(self, tmp_path):
        """Blur filter should keep sharp frames and remove blurry ones."""
        from sigma.preprocessor.video import VideoPreprocessor

        output_dir = tmp_path / "frames"
        output_dir.mkdir()

        # Create a sharp frame (high-frequency content)
        sharp = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(sharp, (20, 20), (80, 80), 255, -1)
        sharp_bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
        sharp_path = output_dir / "sharp.jpg"
        cv2.imwrite(str(sharp_path), sharp_bgr)

        # Create a blurry frame (smooth gradient)
        blurry = np.full((100, 100, 3), 128, dtype=np.uint8)
        blurry = cv2.GaussianBlur(blurry, (31, 31), 10)
        blurry_path = output_dir / "blurry.jpg"
        cv2.imwrite(str(blurry_path), blurry)

        config = PreprocessConfig(blur_threshold=50.0)
        preprocessor = VideoPreprocessor(config)
        result = preprocessor.filter_blurry([sharp_path, blurry_path], threshold=50.0)

        assert sharp_path in result
        assert blurry_path not in result
        assert not blurry_path.exists()  # should be deleted

    def test_dedup_real(self, tmp_path):
        """Deduplication should remove near-identical frames."""
        from sigma.preprocessor.video import VideoPreprocessor

        output_dir = tmp_path / "frames"
        output_dir.mkdir()

        # Two identical images
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (90, 90), (200, 100, 50), -1)
        p1 = output_dir / "dup1.jpg"
        p2 = output_dir / "dup2.jpg"
        cv2.imwrite(str(p1), img)
        cv2.imwrite(str(p2), img)

        # One different image
        diff = np.full((100, 100, 3), 255, dtype=np.uint8)
        p3 = output_dir / "diff.jpg"
        cv2.imwrite(str(p3), diff)

        config = PreprocessConfig()
        preprocessor = VideoPreprocessor(config)
        result = preprocessor.deduplicate([p1, p2, p3], hash_size=16, threshold=5)

        assert p1 in result
        assert p2 not in result  # duplicate removed
        assert p3 in result
        assert not p2.exists()


# ---------------------------------------------------------------------------
# Map2DGenerator – real OpenCV + numpy, synthetic point cloud
# ---------------------------------------------------------------------------

class TestMap2DGeneratorReal:
    def _make_pcd_stub(self, points: np.ndarray):
        """Lightweight stand-in that behaves like o3d.geometry.PointCloud for Map2DGenerator."""
        class FakePC:
            pass
        pc = FakePC()
        pc.points = points
        return pc

    def test_generate_occupancy_grid(self):
        """Generate a grid from a synthetic block of points."""
        from sigma.map_generator.map_2d import Map2DGenerator

        config = Map2DConfig(
            z_slice_min=0.0, z_slice_max=2.0,
            resolution=0.1, occupancy_threshold=1,
            morphology_kernel_size=3, boundary_method="contour",
        )
        gen = Map2DGenerator(config)

        # Dense 2-m square block at z=1
        x = np.linspace(0, 2, 50)
        y = np.linspace(0, 2, 50)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        grid = gen.generate(self._make_pcd_stub(points))
        assert grid.shape[0] > 0 and grid.shape[1] > 0
        assert np.any(grid > 0), "Grid should contain occupied cells"

    def test_empty_slice_returns_blank(self):
        """Points outside the z-slice should yield a blank grid."""
        from sigma.map_generator.map_2d import Map2DGenerator

        config = Map2DConfig(z_slice_min=5.0, z_slice_max=10.0)
        gen = Map2DGenerator(config)

        points = np.random.rand(500, 3)  # z in [0,1] – outside slice
        grid = gen.generate(self._make_pcd_stub(points))
        assert np.all(grid == 0)

    def test_export_image(self, tmp_path):
        """export_image should write a valid PNG."""
        from sigma.map_generator.map_2d import Map2DGenerator

        config = Map2DConfig()
        gen = Map2DGenerator(config)
        grid = np.zeros((50, 50), dtype=np.uint8)
        grid[10:40, 10:40] = 255
        out = tmp_path / "map.png"
        gen.export_image(grid, out)
        assert out.exists()
        loaded = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        assert loaded is not None
        assert loaded.shape == (50, 50)


# ---------------------------------------------------------------------------
# CLI error handling
# ---------------------------------------------------------------------------

class TestCLIErrors:
    def test_extract_frames_missing_video(self, tmp_path):
        """CLI should exit with code 1 if video doesn't exist."""
        from typer.testing import CliRunner
        from sigma.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "extract-frames",
            "--video", str(tmp_path / "nonexistent.mp4"),
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 1

    def test_generate_map_missing_model(self, tmp_path):
        """CLI should exit with code 1 if model doesn't exist."""
        from typer.testing import CliRunner
        from sigma.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "generate-map",
            "--model", str(tmp_path / "nonexistent.ply"),
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 1
