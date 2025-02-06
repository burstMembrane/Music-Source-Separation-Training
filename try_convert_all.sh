#!/bin/bash

# Define model names and corresponding checkpoints/configs
declare -A checkpoints=(
  ["mdx23c"]="musdb_models/model_mdx23c_ep_168_sdr_7.0207.ckpt"
  ["bs_mamba2"]="musdb_models/model_bs_mamba2_ep_11_sdr_6.8723.ckpt"
  ["scnet"]="musdb_models/scnet_checkpoint_musdb18.ckpt"
  ["scnet_large"]="musdb_models/model_scnet_sdr_9.3244.ckpt"
  ["scnet_large_starrytong"]="musdb_models/SCNet-large_starrytong_fixed.ckpt"
  ["scnet_xl"]="musdb_models/model_scnet_ep_54_sdr_9.8051.ckpt"
  ["bs_roformer"]="musdb_models/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
)

declare -A configs=(
  ["mdx23c"]="musdb_models/config_musdb18_mdx23c.yaml"
  ["bs_mamba2"]="musdb_models/config_musdb18_bs_mamba2.yaml"
  ["scnet"]="musdb_models/config_musdb18_scnet.yaml"
  ["scnet_large"]="musdb_models/config_musdb18_scnet_large.yaml"
  ["scnet_large_starrytong"]="musdb_models/config_musdb18_scnet_large_starrytong.yaml"
  ["scnet_xl"]="musdb_models/config_musdb18_scnet_xl.yaml"
  ["bs_roformer"]="musdb_models/config_bs_roformer_384_8_2_485100.yaml"
)

# Function to extract the base model type (before the first underscore)
get_model_type() {
  local model_name="$1"
  echo "${model_name%%_*}"
}

# Iterate through models and run export script
for model in "${!checkpoints[@]}"; do
  model_type=$(get_model_type "$model")
  output_file="torchscript_models/${model}_torchscript.pt"
  mkdir -p "torchscript_models"
  echo "Running export for $model (model_type: $model_type)..."
  .venv/bin/python convert.py --model "$model_type" --checkpoint "${checkpoints[$model]}" --output "$output_file" --config "${configs[$model]}"

  if [ $? -eq 0 ]; then
    echo "Export successful: $output_file"
  else
    echo "Export failed for $model"
  fi
done
