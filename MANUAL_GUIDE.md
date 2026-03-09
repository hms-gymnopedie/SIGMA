# SIGMA Manual Execution Guide

This guide explains how to run the SIGMA pipeline. You can either run **everything on a single GPU server**, or split the work between your **Local Mac** and **School Server**.

## Prerequisites

- **Local Mac**: Python 3.10+, `uv` installed (`pip install uv` or `brew install uv`)
- **School Server**: Python 3.10+, `uv` installed
- **SSH Access**: You should be able to SSH into your school server.

## Initial Setup

### Local Mac

```bash
cd sigma
uv venv
source .venv/bin/activate
uv pip install -e "."
```

### School Server (GPU)

```bash
cd sigma
uv venv
source .venv/bin/activate
uv pip install -e ".[server]"
chmod +x scripts/run_colmap.sh scripts/run_train.sh
```

---

## Option A: Server-Only Pipeline (Recommended)

Run the entire pipeline on a single GPU server with one command. No file transfer needed.

```bash
ssh user@server_address
cd ~/sigma_workspace/sigma
source .venv/bin/activate
```

### Using the CLI

```bash
sigma run-all \
    --video ~/data/site_video.mp4 \
    --output ~/sigma_workspace/project_001
```

### Using the Shell Script

```bash
./scripts/run_pipeline.sh \
    ~/data/site_video.mp4 \
    ~/sigma_workspace/project_001
```

**Output structure:**
```
project_001/
├── frames/          # Extracted video frames
├── colmap_out/      # COLMAP sparse reconstruction
├── gs_model/        # Trained 3DGS model (point_cloud.ply)
└── maps/            # Generated 2D/3D maps
    ├── occupancy_map.png
    └── model_export.ply
```

> **Tip**: Use `tmux` to keep the pipeline running if you disconnect.
> ```bash
> tmux new -s sigma
> sigma run-all --video ... --output ...
> # Ctrl+B, D to detach
> # tmux attach -t sigma to reconnect
> ```

---

## Option B: Hybrid Pipeline (Local ↔ Server)

Split the work between your Mac and a GPU server.

### 1. Local: Extract Frames from Video

On your **Mac** (with venv activated):

```bash
# Extract frames from your video file
sigma extract-frames \
    --video /path/to/your/video.mp4 \
    --output ./my_project/frames
```

This will create a `frames` directory with image sequences.

### 2. Transfer: Upload Frames to Server

Use `rsync` or `scp` to upload the `frames` folder to your server.

```bash
# Example
rsync -avz ./my_project/frames/ user@server_address:~/sigma_workspace/project_001/frames/
```

### 3. Server: Run COLMAP (Structure-from-Motion)

SSH into your **Server**:

```bash
ssh user@server_address
cd ~/sigma_workspace/sigma  # Go to sigma source directory
```

Run the COLMAP wrapper script:

```bash
# Run COLMAP
./scripts/run_colmap.sh \
    ~/sigma_workspace/project_001/frames \
    ~/sigma_workspace/project_001/colmap_out
```

This will generate `sparse/0` inside `colmap_out`.

### 4. Server: Run 3DGS Training

On your **Server**:

```bash
# Run Training
./scripts/run_train.sh \
    ~/sigma_workspace/project_001/colmap_out/sparse/0 \
    ~/sigma_workspace/project_001/gs_model
```

This will train the model and save `point_cloud.ply` in `gs_model`.

**Tip**: Use `tmux` to keep training running if you disconnect.

### 5. Transfer: Download Model to Local

Back on your **Mac**:

```bash
# Download the trained model folder
rsync -avz \
    user@server_address:~/sigma_workspace/project_001/gs_model/ \
    ./my_project/gs_model/
```

### 6. Local: Generate Maps

On your **Mac**:

```bash
# Generate 2D Floorplan and export 3D map
sigma generate-map \
    --model ./my_project/gs_model/point_cloud.ply \
    --output ./my_project/maps
```

You will get:
- `occupancy_map.png` (2D Floorplan)
- `model_export.ply` (3D Point Cloud/Mesh)

### 7. View Results

```bash
sigma view-2d --map ./my_project/maps/occupancy_map.png
sigma view-3d --model ./my_project/maps/model_export.ply
```

---

## Recent Updates (2026-02-14)

### Code Quality Improvements

The following bugs have been fixed:

1. **3DGS Model Conversion Stability** (`converter.py`)
   - Fixed sigmoid overflow for extreme negative values
   - Graceful fallback when PLY files lack `opacity` or `f_dc_*` fields

2. **2D Map Generation Accuracy** (`map_2d.py`)
   - Removed dead code (unused index calculations)
   - Fixed histogram range mismatch → improved BEV projection consistency

3. **Video Frame Extraction Robustness** (`video.py`)
   - Added FPS validation to prevent division by zero
   - Better error messages for corrupt video files

4. **3D Model Viewer Stability** (`viewer.py`)
   - File existence check before loading models
   - Clear error messages for missing files

5. **Configuration** (`config.py`)
   - Removed unimplemented `"hough"` boundary detection method

See `sigma/CHANGELOG.md` for detailed modification history.
