#!/bin/bash

# Base parameters
FIRST_MODEL="ddtasnet"
SECOND_MODEL="hstasnet"
FIRST_CONFIG="./configs/config_musdb18_hstasnet_lg.yaml"
SECOND_CONFIG="configs/config_musdb18_hstasnet_lg.yaml"
FIRST_CHECKPOINT="pretrained/model_ddtasnet_ep_39_sdr_1.6587.ckpt"
SECOND_CHECKPOINT="./results/model_hstasnet_ep_0_sdr_4.4599.ckpt"
BASE_OUTPUT_DIR="./results/ddtasnet_hstasnet_merged"
TEST_SONG="Al James - Schoolboy Facination"
TEST_DATA_DIR="/Users/liampower/.datasets/musdb18hq/test/${TEST_SONG}"
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

# Evaluate initial models
echo "Evaluating initial models..."

# First model
FIRST_MODEL_DIR="$BASE_OUTPUT_DIR/initial_first_model"
mkdir -p "$FIRST_MODEL_DIR"
run_inference_and_evaluate "initial_first_model" "$FIRST_CHECKPOINT" "$FIRST_MODEL_DIR" "$FIRST_MODEL"

# Second model
SECOND_MODEL_DIR="$BASE_OUTPUT_DIR/initial_second_model"
mkdir -p "$SECOND_MODEL_DIR"
run_inference_and_evaluate "initial_second_model" "$SECOND_CHECKPOINT" "$SECOND_MODEL_DIR" "$SECOND_MODEL"

# Print initial results
echo "Initial Model Results:"
echo "----------------------------------------"
echo "First Model (${FIRST_MODEL}):"
cat "$FIRST_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"
echo "Second Model (${SECOND_MODEL}):"
cat "$SECOND_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"


# Define all merge strategies
basic_strategies=(
    "original" 
    "multiply" 
    "max" 
    "min" 
    "weighted_add"
    "adaptive"
    "magnitude_weighted"
    "layer_selective"
    "slerp"
    "ties"
)
weight_ratios=(0.3 0.5 0.7)

# Process each basic strategy
for strategy in "${basic_strategies[@]}"; do
    echo "Processing merge strategy: $strategy"
    
    # Define output paths for this strategy
    MODEL_OUTPUT="$BASE_OUTPUT_DIR/merged_model_${strategy}.ckpt"
    INFERENCE_OUTPUT_DIR="$BASE_OUTPUT_DIR/inference_${strategy}"
    mkdir -p "$INFERENCE_OUTPUT_DIR"
    
    # Additional parameters for specific strategies
    extra_args=""
    if [ "$strategy" = "adaptive" ] || [ "$strategy" = "layer_selective" ]; then
        extra_args="--validation_data $TEST_DATA_DIR/mixture.wav"
    fi
    
    # Run merge with current strategy
    python merge.py \
        --first_model "$FIRST_MODEL" \
        --second_model "$SECOND_MODEL" \
        --first_config "$FIRST_CONFIG" \
        --second_config "$SECOND_CONFIG" \
        --first_checkpoint "$FIRST_CHECKPOINT" \
        --second_checkpoint "$SECOND_CHECKPOINT" \
        --output_path "$MODEL_OUTPUT" \
        --merge_strategy "$strategy" \
        $extra_args
    
    # Run inference and evaluate
    run_inference_and_evaluate "$strategy" "$MODEL_OUTPUT" "$INFERENCE_OUTPUT_DIR" "$FIRST_MODEL"
done

# Process weighted_add with different ratios
for ratio in "${weight_ratios[@]}"; do
    strategy="weighted_add"
    strategy_name="${strategy}_${ratio}"
    echo "Processing merge strategy: $strategy_name"
    
    # Define output paths for this strategy
    MODEL_OUTPUT="$BASE_OUTPUT_DIR/merged_model_${strategy_name}.ckpt"
    INFERENCE_OUTPUT_DIR="$BASE_OUTPUT_DIR/inference_${strategy_name}"
    mkdir -p "$INFERENCE_OUTPUT_DIR"
    
    # Run merge with current strategy
    python merge.py \
        --first_model "$FIRST_MODEL" \
        --second_model "$SECOND_MODEL" \
        --first_config "$FIRST_CONFIG" \
        --second_config "$SECOND_CONFIG" \
        --first_checkpoint "$FIRST_CHECKPOINT" \
        --second_checkpoint "$SECOND_CHECKPOINT" \
        --output_path "$MODEL_OUTPUT" \
        --merge_strategy "$strategy" \
        --weight_ratio "$ratio"
    
    # Run inference and evaluate
    run_inference_and_evaluate "$strategy_name" "$MODEL_OUTPUT" "$INFERENCE_OUTPUT_DIR" "$FIRST_MODEL"
done

# Print final comparison
echo "Final SDR Comparison:"
echo "----------------------------------------"
echo "Initial Models:"
echo "First Model (${FIRST_MODEL}):"
cat "$FIRST_MODEL_DIR/sdr_metrics.json"
echo "Second Model (${SECOND_MODEL}):"
cat "$SECOND_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"
echo "Merged Models:"
for strategy in "${basic_strategies[@]}"; do
    if [ "$strategy" != "weighted_add" ]; then
        echo "Strategy: $strategy"
        cat "$BASE_OUTPUT_DIR/inference_${strategy}/sdr_metrics.json"
        echo "----------------------------------------"
    fi
done

for ratio in "${weight_ratios[@]}"; do
    echo "Strategy: weighted_add_${ratio}"
    cat "$BASE_OUTPUT_DIR/inference_weighted_add_${ratio}/sdr_metrics.json"
    echo "----------------------------------------"
done

# For SLERP:
python merge.py \
    --merge_strategy "slerp" \
    --weight_ratio 0.3 \
    [other arguments...]

# For TIES:
python merge.py \
    --merge_strategy "ties" \
    [other arguments...]


FIRST_CONFIG="configs/config_musdb18_hstasnet_lg.yaml"
SECOND_CONFIG="./configs/config_musdb18_hstasnet_lg.yaml"
FIRST_CHECKPOINT="./results/model_hstasnet_ep_0_sdr_4.4599.ckpt"
SECOND_CHECKPOINT="pretrained/model_ddtasnet_ep_39_sdr_1.6587.ckpt"
BASE_OUTPUT_DIR="./results/merged_models"
TEST_SONG="Al James - Schoolboy Facination"
TEST_DATA_DIR="/Users/liampower/.datasets/musdb18hq/test/${TEST_SONG}"
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

# Evaluate initial models
echo "Evaluating initial models..."

# First model
FIRST_MODEL_DIR="$BASE_OUTPUT_DIR/initial_first_model"
mkdir -p "$FIRST_MODEL_DIR"
run_inference_and_evaluate "initial_first_model" "$FIRST_CHECKPOINT" "$FIRST_MODEL_DIR" "$FIRST_MODEL"

# Second model
SECOND_MODEL_DIR="$BASE_OUTPUT_DIR/initial_second_model"
mkdir -p "$SECOND_MODEL_DIR"
run_inference_and_evaluate "initial_second_model" "$SECOND_CHECKPOINT" "$SECOND_MODEL_DIR" "$SECOND_MODEL"

# Print initial results
echo "Initial Model Results:"
echo "----------------------------------------"
echo "First Model (${FIRST_MODEL}):"
cat "$FIRST_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"
echo "Second Model (${SECOND_MODEL}):"
cat "$SECOND_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"


# Define all merge strategies
basic_strategies=(
    "original" 
    "multiply" 
    "max" 
    "min" 
    "weighted_add"
    "adaptive"
    "magnitude_weighted"
    "layer_selective"
)
weight_ratios=(0.3 0.5 0.7)

# Process each basic strategy
for strategy in "${basic_strategies[@]}"; do
    echo "Processing merge strategy: $strategy"
    
    # Define output paths for this strategy
    MODEL_OUTPUT="$BASE_OUTPUT_DIR/merged_model_${strategy}.ckpt"
    INFERENCE_OUTPUT_DIR="$BASE_OUTPUT_DIR/inference_${strategy}"
    mkdir -p "$INFERENCE_OUTPUT_DIR"
    
    # Additional parameters for specific strategies
    extra_args=""
    if [ "$strategy" = "adaptive" ] || [ "$strategy" = "layer_selective" ]; then
        extra_args="--validation_data '$TEST_DATA_DIR/mixture.wav'"
    fi
    
    # Run merge with current strategy
    python merge.py \
        --first_model "$FIRST_MODEL" \
        --second_model "$SECOND_MODEL" \
        --first_config "$FIRST_CONFIG" \
        --second_config "$SECOND_CONFIG" \
        --first_checkpoint "$FIRST_CHECKPOINT" \
        --second_checkpoint "$SECOND_CHECKPOINT" \
        --output_path "$MODEL_OUTPUT" \
        --merge_strategy "$strategy" \
        $extra_args
    
    # Run inference and evaluate
    run_inference_and_evaluate "$strategy" "$MODEL_OUTPUT" "$INFERENCE_OUTPUT_DIR" "$FIRST_MODEL"
done

# Process weighted_add with different ratios
for ratio in "${weight_ratios[@]}"; do
    strategy="weighted_add"
    strategy_name="${strategy}_${ratio}"
    echo "Processing merge strategy: $strategy_name"
    
    # Define output paths for this strategy
    MODEL_OUTPUT="$BASE_OUTPUT_DIR/merged_model_${strategy_name}.ckpt"
    INFERENCE_OUTPUT_DIR="$BASE_OUTPUT_DIR/inference_${strategy_name}"
    mkdir -p "$INFERENCE_OUTPUT_DIR"
    
    # Run merge with current strategy
    python merge.py \
        --first_model "$FIRST_MODEL" \
        --second_model "$SECOND_MODEL" \
        --first_config "$FIRST_CONFIG" \
        --second_config "$SECOND_CONFIG" \
        --first_checkpoint "$FIRST_CHECKPOINT" \
        --second_checkpoint "$SECOND_CHECKPOINT" \
        --output_path "$MODEL_OUTPUT" \
        --merge_strategy "$strategy" \
        --weight_ratio "$ratio"
    
    # Run inference and evaluate
    run_inference_and_evaluate "$strategy_name" "$MODEL_OUTPUT" "$INFERENCE_OUTPUT_DIR" "$FIRST_MODEL"
done

# Print final comparison
echo "Final SDR Comparison:"
echo "----------------------------------------"
echo "Initial Models:"
echo "First Model (${FIRST_MODEL}):"
cat "$FIRST_MODEL_DIR/sdr_metrics.json"
echo "Second Model (${SECOND_MODEL}):"
cat "$SECOND_MODEL_DIR/sdr_metrics.json"
echo "----------------------------------------"
echo "Merged Models:"
for strategy in "${basic_strategies[@]}"; do
    if [ "$strategy" != "weighted_add" ]; then
        echo "Strategy: $strategy"
        cat "$BASE_OUTPUT_DIR/inference_${strategy}/sdr_metrics.json"
        echo "----------------------------------------"
    fi
done

for ratio in "${weight_ratios[@]}"; do
    echo "Strategy: weighted_add_${ratio}"
    cat "$BASE_OUTPUT_DIR/inference_weighted_add_${ratio}/sdr_metrics.json"
    echo "----------------------------------------"
done

