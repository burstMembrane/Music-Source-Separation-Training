#!/bin/bash

python optimize.py \
  --model_type hstasnet \
  --config_path configs/config_musdb18_hstasnet.yaml \
  --results_path results/ \
  --start_check_point ./results/model_hstasnet_ep_4_sdr_3.4707.ckpt \
  --data_path ~/.datasets/musdb18hq/train \
  --valid_path ~/.datasets/musdb18hq/test \
  --num_workers 4 \
  --use_l1_loss \
  --wandb_key 1fd8c5d21d89fb952358e0b4553ddb7e2d76c0eb \
  --device_ids 0 \
  --n_trials 10