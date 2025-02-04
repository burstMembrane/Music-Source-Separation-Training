#!/bin/bash

# Define model names and corresponding checkpoints/configs
declare -A checkpoints=(
  ["hstasnet"]="pretrained/model_hstasnet_030225_ep_3_sdr_3.6091.ckpt"
  ["scnet"]="pretrained/scnet_checkpoint_musdb18.ckpt"
  ["mdx23c"]="pretrained/model_mdx23c_ep_168_sdr_7.0207.ckpt"
  ["bs_roformer"]="pretrained/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
)

declare -A configs=(
  ["hstasnet"]="pretrained/config_bs_roformer_384_8_2_485100.yaml"
  ["scnet"]="pretrained/config_musdb18_scnet.yaml"
  ["mdx23c"]="pretrained/config_musdb18_mdx23c.yaml"
  ["bs_roformer"]="pretrained/config_bs_roformer_384_8_2_485100.yaml"
)

# Iterate through models and run export script
for model in "${!checkpoints[@]}"; do
  output_file="${model}.onnx"

  echo "Running export for $model..."
  python export.py --model "$model" --checkpoint "${checkpoints[$model]}" --output "$output_file" --config "${configs[$model]}"

  if [ $? -eq 0 ]; then
    echo "Export successful: $output_file"
  else
    echo "Export failed for $model"
  fi
done
