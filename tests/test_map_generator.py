import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
from sigma.config import Map2DConfig

# Mock open3d IS REQUIRED before importing map_2d which imports it
sys.modules['open3d'] = MagicMock()
sys.modules['open3d.geometry'] = MagicMock()
from sigma.map_generator.map_2d import Map2DGenerator

@pytest.fixture
def mock_map_config():
    return Map2DConfig(
        z_slice_min=0.0,
        z_slice_max=1.0,
        resolution=0.1,
        occupancy_threshold=0,
        morphology_kernel_size=1, # Kernel 1 means no-op basically? Or 3x3 identity? No, kernel size for StructuringElement
        boundary_method="contour" # Contour is safer than Canny for binary blocks
    )

def test_generate_map(mock_map_config):
    generator = Map2DGenerator(mock_map_config)
    
    # Mock PointCloud
    pcd = MagicMock()
    
    # Create a dense block: 20x20 points
    # Min x=1.0, Max x=3.0 -> 2m -> 20 pixels
    x = np.linspace(1.0, 3.0, 20)
    y = np.linspace(1.0, 3.0, 20)
    xx, yy = np.meshgrid(x, y)
    
    points = np.stack((xx.flatten(), yy.flatten(), np.ones_like(xx.flatten())*0.5), axis=1)
    
    pcd.points = points
    
    # Run
    grid = generator.generate(pcd)
    
    # Assert
    # Grid should be approx 20x20
    # Should have non-zero pixels
    assert np.any(grid != 0)
