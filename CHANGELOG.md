# CHANGELOG

All notable changes to the SIGMA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial implementation of SIGMA framework
- 6-step hybrid pipeline (local preprocessing → server SfM/3DGS → local map generation)
- CLI commands: `extract-frames`, `generate-map`, `view-2d`, `view-3d`
- YAML-based configuration system via Pydantic
- Comprehensive test suite with mocked dependencies

### Fixed

#### [2026-02-14] Code Quality & Bug Fixes

**Numerical Stability & Error Handling**

- **[converter.py:34-50]** Fixed numerically unstable sigmoid computation
  - **Problem**: `1 / (1 + np.exp(-opacities))` overflows when opacities contain large negative values
  - **Solution**: Implemented numerically stable sigmoid using `np.where()` to handle positive/negative values separately
  ```python
  # Before: 1 / (1 + np.exp(-opacities))
  # After: np.where(opacities >= 0, 1/(1+exp(-x)), exp(x)/(1+exp(x)))
  ```

- **[converter.py:34-78]** Added graceful fallback for missing PLY fields
  - **Problem**: Crashes with `ValueError` when PLY files lack `opacity` or `f_dc_*` color fields
  - **Solution**: Wrapped field access in try/except blocks with warning messages
    - Missing `opacity` → defaults to all 1.0 (no filtering)
    - Missing `f_dc_*` → skips color assignment

- **[converter.py:5]** Removed unused `PlyElement` import

**Dead Code & Data Consistency**

- **[map_2d.py:54-66]** Removed dead code and fixed histogram range
  - **Problem**: Computed `indices_x` and `indices_y` but never used them (histogram2d handles binning directly)
  - **Problem**: Histogram range `[x_min, x_max + resolution]` mismatched grid dimensions `grid_w = ceil(width/resolution) + 1`
  - **Solution**:
    - Deleted unused index calculations (lines 54-66)
    - Changed range to `[x_min, x_min + grid_w * resolution]` for proper alignment

**Import Cleanup**

- **[colmap_runner.py:4-5]** Removed unused imports
  - Deleted: `subprocess`, `List` (from `typing`)

- **[colmap_runner.py:19-21]** Removed redundant warning before exception
  - **Problem**: `logger.warning()` followed immediately by `raise ImportError()` → warning never appears in logs
  - **Solution**: Removed warning, kept only the exception

- **[trainer.py:1-5]** Removed unused imports
  - Deleted: `time`, `math`, `numpy` (not used in placeholder implementation)
  - Deleted: `GaussianModel`, `CameraHandle` from gsplat (not used yet)

- **[cli.py:4-7]** Removed unnecessary `sys.path` manipulation
  - **Problem**: `sys.path.append(str(Path(__file__).parent.parent))` is unnecessary when package is properly installed via `pip install -e .`
  - **Solution**: Removed `import sys` and path hack

- **[config.py:2]** Removed unused `Optional` import

- **[video.py:4]** Removed unused `Optional` import

**Configuration Validation**

- **[config.py:42]** Removed unimplemented `"hough"` boundary method
  - **Problem**: `boundary_method: Literal["canny", "contour", "hough"]` allows "hough", but `map_2d.py` has no implementation for it
  - **Solution**: Changed to `Literal["canny", "contour"]` only

**Type Safety**

- **[map_3d.py:16]** Improved type annotation
  - **Before**: `def export(self, pcd: object, output_path: Path)`
  - **After**: `def export(self, pcd: "o3d.geometry.PointCloud", output_path: Path)`

**Robustness**

- **[video.py:26-29]** Added FPS validation
  - **Problem**: `video_fps = cap.get(cv2.CAP_PROP_FPS)` can return 0 for corrupt videos → `ZeroDivisionError` on line 31
  - **Solution**: Added check for `video_fps <= 0` with `cap.release()` and descriptive error message

- **[video.py:28]** Removed unused variable
  - Deleted: `duration = total_frames / video_fps` (computed but never used)

- **[viewer.py:18-20]** Added file existence check
  - **Problem**: `view_3d()` calls `o3d.io.read_point_cloud()` without checking if file exists → unclear error message
  - **Solution**: Added `if not model_path.exists()` check with clear error logging

---

## Summary of Changes

**Files Modified**: 9
**Lines Changed**: ~50 additions, ~40 deletions
**Bug Severity**:
- 🔴 Critical (crashes): 3 (sigmoid overflow, FPS division by zero, missing PLY fields)
- 🟡 Medium (incorrect results): 1 (histogram range mismatch)
- 🟢 Low (code quality): 5 (dead code, unused imports, weak types)

**Impact**:
- ✅ Improved numerical stability for edge-case 3DGS models
- ✅ Better error messages for invalid inputs (corrupt videos, missing files)
- ✅ Cleaner codebase (removed 15+ unused imports/variables)
- ✅ More accurate 2D map generation (fixed BEV projection)

---

## [0.1.0] - Initial Release

### Added
- Video frame extraction with blur/deduplication filtering
- COLMAP Structure-from-Motion wrapper
- 3D Gaussian Splatting training integration
- 2D occupancy grid generation from point clouds
- 3D model export (PLY/OBJ/GLB)
- Interactive 2D/3D visualization
- Configuration management via YAML + Pydantic
- Unit tests with mocked dependencies
- Integration tests with synthetic data
- Documentation (README, MANUAL_GUIDE, implementation plan)

### Known Limitations
- `trainer.py` is placeholder code (training loop not implemented)
- `vocab_tree` matching falls back to `sequential` (vocab tree path not configurable)
- Coordinate system (Y-down vs Z-up) not validated with real data
- No automated end-to-end test with real video footage

---

## Contributing

When making changes:
1. Update this CHANGELOG under `[Unreleased]` section
2. Use semantic commit messages: `fix:`, `feat:`, `refactor:`, `docs:`, etc.
3. Reference file paths and line numbers for bug fixes
4. Include before/after code snippets for clarity
5. Run tests: `pytest tests/ -v`
6. Format code: `black src/ tests/ && isort src/ tests/`

---

**Legend**:
- 🔴 Critical: Crashes, data corruption, security issues
- 🟡 Medium: Incorrect results, performance issues
- 🟢 Low: Code quality, documentation, minor improvements
