#!/bin/bash
python inference.py \
--model_type bs_roformer \
--config_path ./pretrained/config_bs_roformer_384_8_2_485100.yaml \
--start_check_point pretrained/model_bs_roformer_ep_17_sdr_9.6568.ckpt \
--input_folder input/ \
--store_dir separation_results/ \
--force_cpu
