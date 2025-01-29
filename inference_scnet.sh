#!/bin/bash

./inference.sh --model_name scnet \
  --config_path pretrained/config_musdb18_scnet_lora.yaml \
  --checkpoint_path pretrained/scnet_checkpoint_musdb18.ckpt
