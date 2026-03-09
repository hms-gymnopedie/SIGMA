#!/bin/bash
set -euo pipefail
# Unified pipeline script: Video → Frames → COLMAP → 3DGS → Maps
# Runs the entire SIGMA pipeline on a single server.

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "Usage: $0 [video_path] [output_dir] [config_path]"
    echo ""
    echo "Example:"
    echo "  $0 ~/data/site_video.mp4 ~/sigma_workspace/project_001"
    echo ""
    echo "Output structure:"
    echo "  output_dir/"
    echo "    ├── frames/          # Extracted video frames"
    echo "    ├── colmap_out/      # COLMAP sparse reconstruction"
    echo "    ├── gs_model/        # Trained 3DGS model"
    echo "    └── maps/            # Generated 2D/3D maps"
    exit 1
fi

VIDEO_PATH=$1
OUTPUT_DIR=$2
CONFIG_PATH=${3:-"configs/default.yaml"}

echo "=== SIGMA Full Pipeline ==="
echo "Video:  $VIDEO_PATH"
echo "Output: $OUTPUT_DIR"
echo "Config: $CONFIG_PATH"
echo "=========================="

# Ensure sigma package is in python path
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

# Check required packages
python -c "import pycolmap" 2>/dev/null || { echo "Error: pycolmap not installed."; exit 1; }
python -c "import gsplat"   2>/dev/null || { echo "Error: gsplat not installed."; exit 1; }

# Run full pipeline via CLI
sigma run-all \
    --video "$VIDEO_PATH" \
    --output "$OUTPUT_DIR" \
    --config "$CONFIG_PATH"

echo "=== Pipeline finished ==="
