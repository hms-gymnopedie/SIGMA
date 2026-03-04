import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from sigma.config import PreprocessConfig
from sigma.preprocessor.video import VideoPreprocessor
import numpy as np

@pytest.fixture
def mock_video_config():
    return PreprocessConfig(
        extraction_fps=1.0,
        max_dimension=100,
        blur_threshold=0.0, # Disable blur filter for simple extraction test
        dedup_threshold=0 # Disable dedup for simple extraction test
    )

@patch('cv2.VideoCapture')
@patch('cv2.imwrite')
def test_extract_frames(mock_imwrite, mock_video_capture, mock_video_config, tmp_path):
    # Setup mock video capture
    mock_cap = MagicMock()
    mock_video_capture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    
    # Video FPS 30. We want extract 1 FPS. Interval = 30.
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 300 # FPS=30, Count=300 (enough for multiple)
    
    # Mock reading frames.
    # We need loop to run enough times to catch interval frames at 0, 30, 60...
    # side_effect needs to be long enough.
    # Let's just create a generator that returns True/Frame for N times
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    def frame_generator():
        for _ in range(70): # Generate 70 frames. Expected extractions at 0, 30, 60 -> 3 frames.
            yield True, frame
        yield False, None
        
    mock_cap.read.side_effect = frame_generator()
    
    preprocessor = VideoPreprocessor(mock_video_config)
    output_dir = tmp_path / "frames"
    output_dir.mkdir()
    
    # Run
    frames = preprocessor.extract_frames(Path("dummy.mp4"), output_dir)
    
    # Assert
    # With 70 frames and interval 30:
    # 0 -> Saved
    # 30 -> Saved
    # 60 -> Saved
    # Loop ends at 70.
    assert len(frames) == 3 
    assert mock_imwrite.call_count == 3
