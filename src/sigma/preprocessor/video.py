import cv2
import numpy as np
from pathlib import Path
from typing import List
from sigma.config import PreprocessConfig
import logging

logger = logging.getLogger(__name__)

class VideoPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config

    def extract_frames(self, video_path: Path, output_dir: Path) -> List[Path]:
        """
        Extract frames from a video file at a specified FPS.
        """
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            cap.release()
            raise ValueError(f"Invalid video FPS ({video_fps}) for: {video_path}")

        target_fps = self.config.extraction_fps
        frame_interval = int(video_fps / target_fps)
        if frame_interval < 1:
            frame_interval = 1
            
        logger.info(f"Video FPS: {video_fps}, Target FPS: {target_fps}, Interval: {frame_interval}")
        
        extracted_paths = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Resize if needed
                h, w = frame.shape[:2]
                max_dim = self.config.max_dimension
                if max_dim and (w > max_dim or h > max_dim):
                    scale = max_dim / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Save frame temporarily
                frame_filename = f"frame_{saved_count:05d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                extracted_paths.append(frame_path)
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        logger.info(f"Extracted {saved_count} frames.")
        
        # Post-processing: Blur filtering
        if self.config.blur_threshold > 0:
            extracted_paths = self.filter_blurry(extracted_paths, self.config.blur_threshold)
            
        # Post-processing: De-duplication
        if self.config.dedup_threshold > 0:
            extracted_paths = self.deduplicate(extracted_paths, self.config.dedup_hash_size, self.config.dedup_threshold)
            
        return extracted_paths

    def filter_blurry(self, frames: List[Path], threshold: float) -> List[Path]:
        """
        Remove blurry frames based on Laplacian variance.
        """
        logger.info(f"Filtering blurry frames (threshold: {threshold})...")
        filtered_frames = []
        removed_count = 0
        
        for frame_path in frames:
            image = cv2.imread(str(frame_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if variance < threshold:
                logger.debug(f"Frame {frame_path.name} is blurry (variance: {variance:.2f} < {threshold})")
                frame_path.unlink() # Delete file
                removed_count += 1
            else:
                filtered_frames.append(frame_path)
                
        logger.info(f"Removed {removed_count} blurry frames. Remaining: {len(filtered_frames)}")
        return filtered_frames

    def deduplicate(self, frames: List[Path], hash_size: int = 16, threshold: int = 5) -> List[Path]:
        """
        Remove duplicate frames based on perceptual hashing (dhash).
        """
        logger.info(f"Deduplicating frames (threshold: {threshold})...")
        if not frames:
            return []
            
        unique_frames = []
        hashes = []
        removed_count = 0
        
        def dhash(image, hash_size=8):
            # Calculate difference hash
            resized = cv2.resize(image, (hash_size + 1, hash_size))
            diff = resized[:, 1:] > resized[:, :-1]
            return diff.flatten()

        for frame_path in frames:
            image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            current_hash = dhash(image, hash_size)
            
            is_duplicate = False
            for existing_hash in hashes:
                # Hamming distance
                distance = np.count_nonzero(current_hash != existing_hash)
                if distance < threshold:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                logger.debug(f"Frame {frame_path.name} is a duplicate.")
                frame_path.unlink()
                removed_count += 1
            else:
                unique_frames.append(frame_path)
                hashes.append(current_hash)
                
        logger.info(f"Removed {removed_count} duplicate frames. Remaining: {len(unique_frames)}")
        return unique_frames
