#!/bin/bash
set -euo pipefail
# Wrapper script for running COLMAP on the server

if [ -z "$1" ] || [ -z "${2:-}" ]; then
    echo "Usage: $0 [image_dir] [output_dir] [config_path]"
    echo "Example: $0 ./frames ./colmap_out"
    exit 1
fi

IMAGE_DIR=$1
OUTPUT_DIR=$2
CONFIG_PATH=${3:-"configs/default.yaml"}

echo "--- Starting COLMAP SfM ---"
echo "Image Dir: $IMAGE_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Config: $CONFIG_PATH"

# Ensure sigma package is in python path
export PYTHONPATH=$PYTHONPATH:.

# Check if pycolmap is installed
python -c "import pycolmap" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: pycolmap not installed. Please install it (e.g. conda install -c conda-forge pycolmap)."
    exit 1
fi

# Run
python -m sigma.sfm.colmap_runner \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config_path "$CONFIG_PATH"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "--- COLMAP Complete ---"
    echo "Output sparse model: $OUTPUT_DIR/sparse/0"
else
    echo "--- COLMAP Failed with error $EXIT_CODE ---"
    exit 1
fi
