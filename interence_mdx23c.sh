#!/bin/bash
python inference.py \
--model_type mdx23c \
--config_path pretrained/config_musdb18_mdx23c.yaml \
--start_check_point pretrained/model_mdx23c_ep_168_sdr_7.0207.ckpt \
--input_folder input/ \
--store_dir separation_results/ \
--force_cpu
