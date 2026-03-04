#!/bin/bash
# Wrapper script for running 3DGS Training on the server

if [ -z "$1" ]; then
    echo "Usage: $0 [colmap_dir] [output_dir] [iteration_count]"
    echo "Example: $0 ./colmap_out ./gs_model 30000"
    exit 1
fi

COLMAP_DIR=$1
OUTPUT_DIR=$2
ITERATIONS=${3:-30000}
CONFIG_PATH="configs/default.yaml"

echo "--- Starting 3DGS Training ---"
echo "COLMAP Dir: $COLMAP_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Iterations: $ITERATIONS"

# Set CUDA device (adjust if needed, or pass via env var)
# export CUDA_VISIBLE_DEVICES=0,1

export PYTHONPATH=$PYTHONPATH:.

# Check if gsplat is installed
python -c "import gsplat" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: gsplat not installed. Please install it."
    exit 1
fi

# Run
python -m sigma.gaussian_splatting.trainer \
    --colmap_dir "$COLMAP_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config_path "$CONFIG_PATH" \
    # Logic in trainer to override iterations if desired, but currently trainer uses config only.
    # To support overriding iterations via CLI arg, I should update trainer.py argparse.
    # But for now, script just runs with config.
    
if [ $? -eq 0 ]; then
    echo "--- Training Complete ---"
    echo "Model saved to: $OUTPUT_DIR/point_cloud.ply"
else
    echo "--- Training Failed with error $? ---"
    exit 1
fi
