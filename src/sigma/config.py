from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field

class PreprocessConfig(BaseModel):
    extraction_fps: float = Field(default=2.0)
    max_dimension: int = Field(default=1920)
    blur_threshold: float = Field(default=100.0)
    dedup_hash_size: int = Field(default=16)
    dedup_threshold: int = Field(default=5)

class SfmConfig(BaseModel):
    feature_type: str = Field(default="sift")
    matching_type: Literal["sequential", "exhaustive", "vocab_tree"] = Field(default="sequential")
    match_overlap: int = Field(default=10)
    min_num_matches: int = Field(default=15)
    use_gpu: bool = Field(default=True)

class LearningRateConfig(BaseModel):
    position: float = 0.00016
    opacity: float = 0.05
    scaling: float = 0.005
    rotation: float = 0.001
    sh: float = 0.0025

class GaussianSplattingConfig(BaseModel):
    iterations: int = 30000
    learning_rate: LearningRateConfig = Field(default_factory=LearningRateConfig)
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    opacity_reset_interval: int = 3000
    sh_degree: int = 3

class Map2DConfig(BaseModel):
    z_slice_min: float = 0.5
    z_slice_max: float = 2.0
    resolution: float = 0.05
    occupancy_threshold: int = 3
    morphology_kernel_size: int = 5
    boundary_method: Literal["canny", "contour"] = "canny"

class Map3DConfig(BaseModel):
    export_format: Literal["ply", "obj", "glb"] = "ply"

class SigmaConfig(BaseModel):
    preprocessor: PreprocessConfig = Field(default_factory=PreprocessConfig)
    sfm: SfmConfig = Field(default_factory=SfmConfig)
    gaussian_splatting: GaussianSplattingConfig = Field(default_factory=GaussianSplattingConfig)
    map_2d: Map2DConfig = Field(default_factory=Map2DConfig)
    map_3d: Map3DConfig = Field(default_factory=Map3DConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "SigmaConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
