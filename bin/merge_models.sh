#!/bin/bash

# exit and pipefail
#
set -euo pipefail

# Check if merge strategy is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <merge_strategy> [weight_ratio]"
    echo "Available strategies: original, multiply, max, min, weighted_add, adaptive, magnitude_weighted, layer_selective, slerp, ties"
    exit 1
fi

MERGE_STRATEGY=$1
WEIGHT_RATIO=${2:-0.5}  # Default to 0.5 if not provided

# Base parameters
FIRST_MODEL="ddtasnet"
SECOND_MODEL="hstasnet"
FIRST_CONFIG="./configs/config_musdb18_hstasnet_lg.yaml"
SECOND_CONFIG="configs/config_musdb18_hstasnet_lg.yaml"
FIRST_CHECKPOINT="pretrained/model_ddtasnet_ep_39_sdr_1.6587.ckpt"
SECOND_CHECKPOINT="./results/model_hstasnet_ep_0_sdr_4.4599.ckpt"
BASE_OUTPUT_DIR="./results/merged_${FIRST_MODEL}_${SECOND_MODEL}"
TEST_SONG="aljames"
TEST_DATA_DIR="/Users/liampower/stepincto/Music-Source-Separation-Training/${TEST_SONG}"
CONFIG_PATH="configs/config_musdb18_ddtasnet_lubuntu.yaml"

# Create output directories
mkdir -p "$BASE_OUTPUT_DIR"

# Function to run inference and evaluate SDR
run_inference_and_evaluate() {
    local model_name=$1
    local model_path=$2
    local inference_dir=$3
    local model_type=$4
    
    echo "Running inference and evaluation for $model_name"
    
    # Create song-specific inference directory
    local song_inference_dir="$inference_dir/${TEST_SONG}"
    mkdir -p "$song_inference_dir"
    
    # Run inference
    python batch_ola.py \
        --model_path "$model_path" \
        --config_path "$CONFIG_PATH" \
        --input_audio "$TEST_DATA_DIR/mixture.wav" \
        --chunk_length 1 \
        --device cpu \
        --batch_size 86 \
        --model_type "$model_type" \
        --output_dir "$inference_dir"
    
    # Calculate SDR metrics
    python evaluate_sdr.py \
        --pred_dir "$song_inference_dir" \
        --gt_dir "$TEST_DATA_DIR" \
        --strategy "$model_name" \
        --output_file "$inference_dir/sdr_metrics.json"
    
    echo "----------------------------------------"
}

# Evaluate initial models only if results don't exist
echo "Checking initial models..."

# First model
FIRST_MODEL_DIR="$BASE_OUTPUT_DIR/initial_first_model"
if [ -f "$FIRST_MODEL_DIR/sdr_metrics.json" ]; then
    echo "Found existing results for first model, skipping evaluation"
else
    echo "Evaluating first model..."
    mkdir -p "$FIRST_MODEL_DIR"
    run_inference_and_evaluate "initial_first_model" "$FIRST_CHECKPOINT" "$FIRST_MODEL_DIR" "$FIRST_MODEL"
fi

# Second model
SECOND_MODEL_DIR="$BASE_OUTPUT_DIR/initial_second_model"
if [ -f "$SECOND_MODEL_DIR/sdr_metrics.json" ]; then
    echo "Found existing results for second model, skipping evaluation"
else
    echo "Evaluating second model..."
    mkdir -p "$SECOND_MODEL_DIR"
    run_inference_and_evaluate "initial_second_model" "$SECOND_CHECKPOINT" "$SECOND_MODEL_DIR" "$SECOND_MODEL"
fi

# Print initial results
echo "Initial Model Results:"
echo "----------------------------------------"
echo "First Model (${FIRST_MODEL}):"
cat "$FIRST_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"
echo "Second Model (${SECOND_MODEL}):"
cat "$SECOND_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"

# Define output paths for merged model
MODEL_OUTPUT="$BASE_OUTPUT_DIR/merged_model_${MERGE_STRATEGY}.ckpt"
INFERENCE_OUTPUT_DIR="$BASE_OUTPUT_DIR/inference_${MERGE_STRATEGY}"
mkdir -p "$INFERENCE_OUTPUT_DIR"

# Additional parameters for specific strategies
extra_args=""
if [ "$MERGE_STRATEGY" = "adaptive" ] || [ "$MERGE_STRATEGY" = "layer_selective" ]; then
    # Properly escape the validation data path
    escaped_path=$(printf "%q" "$TEST_DATA_DIR/mixture.wav")
    extra_args="--validation_data ${escaped_path}"
fi

# Add weight ratio for supported strategies
if [ "$MERGE_STRATEGY" = "weighted_add" ] || [ "$MERGE_STRATEGY" = "slerp" ]; then
    extra_args="${extra_args} --weight_ratio $WEIGHT_RATIO"
fi

# Run merge
echo "Running merge with strategy: $MERGE_STRATEGY"
python merge.py \
    --first_model "$FIRST_MODEL" \
    --second_model "$SECOND_MODEL" \
    --first_config "$FIRST_CONFIG" \
    --second_config "$SECOND_CONFIG" \
    --first_checkpoint "$FIRST_CHECKPOINT" \
    --second_checkpoint "$SECOND_CHECKPOINT" \
    --output_path "$MODEL_OUTPUT" \
    --merge_strategy "$MERGE_STRATEGY" \
    $extra_args

# Run inference and evaluate merged model
run_inference_and_evaluate "${MERGE_STRATEGY}" "$MODEL_OUTPUT" "$INFERENCE_OUTPUT_DIR" "$FIRST_MODEL"

# Print final comparison
echo "Final Results:"
echo "----------------------------------------"
echo "First Model (${FIRST_MODEL}):"
cat "$FIRST_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"
echo "Second Model (${SECOND_MODEL}):"
cat "$SECOND_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"
echo "Merged Model (${MERGE_STRATEGY}):"
cat "$INFERENCE_OUTPUT_DIR/sdr_metrics.json"
echo "----------------------------------------" 
