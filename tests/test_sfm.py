import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from sigma.config import SfmConfig
import sys

# Mock pycolmap before importing ColmapRunner
sys.modules['pycolmap'] = MagicMock()

from sigma.sfm.colmap_runner import ColmapRunner

@pytest.fixture
def mock_sfm_config():
    return SfmConfig(
        feature_type="sift",
        matching_type="sequential",
        match_overlap=5,
        use_gpu=False
    )

def test_colmap_runner_init(mock_sfm_config):
    runner = ColmapRunner(mock_sfm_config)
    assert runner.config == mock_sfm_config

@patch('sigma.sfm.colmap_runner.pycolmap')
def test_colmap_run(mock_pycolmap, mock_sfm_config, tmp_path):
    runner = ColmapRunner(mock_sfm_config)
    
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "colmap_out"
    
    # Mock return values
    mock_reconstruction = MagicMock()
    mock_reconstruction.num_points3D.return_value = 100
    mock_reconstruction.num_images.return_value = 10
    
    # incremental_mapping returns a dict of maps usually {0: recon}
    mock_pycolmap.incremental_mapping.return_value = {0: mock_reconstruction}
    
    # Run
    # The run method expects maps to be indexable like a list or dict if it iterates?
    # In my implementation: `maps = ...` then `maps[0]`. So dict with int key 0 works.
    result_path = runner.run(image_dir, output_dir)
    
    # Assert calls
    mock_pycolmap.extract_features.assert_called_once()
    mock_pycolmap.match_sequential.assert_called_once()
    mock_pycolmap.incremental_mapping.assert_called_once()
    mock_reconstruction.write.assert_called_once()
    assert result_path == output_dir / "sparse" / "0"
