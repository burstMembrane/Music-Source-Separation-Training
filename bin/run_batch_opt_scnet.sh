#!/bin/bash

python bayesian_ola.py \
  --model_type scnet \
  --config_path pretrained/config_musdb18_scnet.yaml \
  --model_path pretrained/scnet_checkpoint_musdb18.ckpt \
  --input_audio input/schoolboyfascination.wav \
  --device cpu \
  --num_calls 10
