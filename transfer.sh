#!/bin/bash
python transfer_eval.py \
  --hst-config configs/config_musdb18_hstasnet.yaml \
  --hst-ckpt pretrained/model_hstasnet_ep_0_sdr_3.6020.ckpt \
  --pt-config pretrained/config_bs_roformer_384_8_2_485100.yaml \
  --pt-ckpt pretrained/model_bs_roformer_ep_17_sdr_9.6568.ckpt \
  --input-dir input/ \
  --output-dir separated_output/ \
  --reference-dir "/Users/liampower/.datasets/musdb18hq/test/Al James - Schoolboy Facination/"