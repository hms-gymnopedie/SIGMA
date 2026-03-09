# SIGMA: Site Imagery to Geometric Map Automation

**SIGMA** is an end-to-end framework for automatically generating 2D floorplans and 3D digital twins from industrial site video footage.

## Features

- **Automated Frame Extraction**: Process video footage into optimal image sequences.
- **High-Quality 3D Reconstruction**: Utilizes 3D Gaussian Splatting for photorealistic 3D models.
- **2D Floorplan Generation**: Automatically generates occupancy maps and 2D floorplans from 3D data.
- **Flexible Architecture**: Run everything on a single GPU server, or split between local (Mac) and server environments.

## Installation

### Local (Mac/PC)

```bash
uv venv
source .venv/bin/activate
uv pip install -e "."
```

### Server (GPU)

Depending on your server's GPU driver and CUDA version, you may need to install a specific version of PyTorch before installing the project dependencies.

1. **Check CUDA version**: Run `nvidia-smi` to find the maximum supported CUDA version on your server.
2. **Install PyTorch**: Install the appropriate PyTorch version matching your CUDA environment.

   **For CUDA 12.1+:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

   **For CUDA 11.8:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install Project Dependencies**: Once the correct PyTorch is installed, install the rest of the package.
   ```bash
   uv pip install -e ".[server]"
   ```

## Usage

### Server-Only (Full Pipeline)

Run the entire pipeline on a single GPU server — no file transfer needed:

```bash
# One command: video → frames → COLMAP → 3DGS → maps
sigma run-all --video input.mp4 --output ./results

# Or via shell script
./scripts/run_pipeline.sh input.mp4 ./results
```

### Hybrid (Local ↔ Server)

Split work between your Mac and a GPU server:

```bash
# Local: Extract frames
sigma extract-frames --video input.mp4 --output ./frames

# (Manual Step: Upload frames to server)

# Server: Run COLMAP
python -m sigma.sfm.colmap_runner --image_dir ./frames --output_dir ./colmap_out

# Server: Run 3DGS Training
python -m sigma.gaussian_splatting.trainer --colmap_dir ./colmap_out --output_dir ./gs_model

# (Manual Step: Download model to local)

# Local: Generate Map
sigma generate-map --model ./gs_model/point_cloud.ply --output ./maps
```
