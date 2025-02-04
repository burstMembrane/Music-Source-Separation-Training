#!/bin/bash

# Directory to save the downloaded models
OUTPUT_DIR="musdb_models"
mkdir -p "$OUTPUT_DIR"

# Array of model URLs
URLS=(
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/config_musdb18_mdx23c.yaml"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/model_mdx23c_ep_168_sdr_7.0207.ckpt"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_bs_mamba2.yaml"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/model_bs_mamba2_ep_11_sdr_6.8723.ckpt"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.8/config_musdb18_scnet_large.yaml"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.8/model_scnet_sdr_9.3244.ckpt"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml"
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
)

# Download all files in parallel using aria2c
aria2c --dir="$OUTPUT_DIR" --input-file=<(printf "%s\n" "${URLS[@]}") --max-connection-per-server=16 --split=16 --continue --auto-file-renaming=false