#!/bin/bash

# Default values (optional)
model_name=""
config_path=""
checkpoint_path=""

# Function to display help
usage() {
  echo "Usage: $0 --model_name <model_name> --config_path <config_path> --checkpoint_path <checkpoint_path>"
  echo
  echo "Arguments:"
  echo "  --model_name        Name of the model"
  echo "  --config_path       Path to the configuration file"
  echo "  --checkpoint_path   Path to the checkpoint file"
  exit 1
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --model_name)
    model_name="$2"
    shift
    ;;
  --config_path)
    config_path="$2"
    shift
    ;;
  --checkpoint_path)
    checkpoint_path="$2"
    shift
    ;;
  -h | --help) usage ;;
  *)
    echo "Unknown parameter passed: $1"
    usage
    ;;
  esac
  shift
done

# Validate required arguments
if [[ -z "$model_name" || -z "$config_path" || -z "$checkpoint_path" ]]; then
  echo "Error: Missing required arguments."
  usage
fi

# Execute the Python script
python inference.py \
  --model_type "$model_name" \
  --config_path "$config_path" \
  --start_check_point "$checkpoint_path" \
  --input_folder input/ \
  --store_dir "separation_results/$model_name/" \
  --draw_spectro 30 \
  --force_cpu \
  # --use_tta \
