import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
from sigma.cli import app
from pathlib import Path

runner = CliRunner()

@patch('sigma.cli.VideoPreprocessor')
@patch('sigma.cli.SigmaConfig')
def test_extract_frames_cli(mock_config_cls, mock_preprocessor_cls, tmp_path):
    mock_config = MagicMock()
    mock_config_cls.from_yaml.return_value = mock_config
    
    mock_preprocessor = MagicMock()
    mock_preprocessor_cls.return_value = mock_preprocessor
    mock_preprocessor.extract_frames.return_value = [Path("frame1.jpg")]
    
    video_path = tmp_path / "input.mp4"
    video_path.touch()
    output_dir = tmp_path / "frames"
    
    result = runner.invoke(app, ["extract-frames", "--video", str(video_path), "--output", str(output_dir), "--config", "dummy.yaml"])
    
    assert result.exit_code == 0
    # Core logic check
    mock_preprocessor.extract_frames.assert_called_once_with(video_path, output_dir)

@patch('sigma.cli.SigmaConfig')
@patch('sigma.cli.GaussianToPointCloud')
@patch('sigma.cli.Map2DGenerator')
@patch('sigma.cli.Map3DGenerator')
def test_generate_map_cli(mock_map3d_cls, mock_map2d_cls, mock_converter_cls, mock_config_cls, tmp_path):
    mock_config_cls.from_yaml.return_value = MagicMock()

    mock_converter = MagicMock()
    mock_converter_cls.return_value = mock_converter
    
    mock_map2d = MagicMock()
    mock_map2d_cls.return_value = mock_map2d
    
    mock_map3d = MagicMock()
    mock_map3d_cls.return_value = mock_map3d
    
    model_path = tmp_path / "model.ply"
    model_path.touch()
    output_dir = tmp_path / "maps"
    
    result = runner.invoke(app, ["generate-map", "--model", str(model_path), "--output", str(output_dir), "--config", "dummy.yaml"])
    
    assert result.exit_code == 0
    mock_converter.convert.assert_called_once_with(model_path)
    mock_map2d.generate.assert_called_once()
    mock_map2d.export_image.assert_called_once()
    mock_map3d.export.assert_called_once()
