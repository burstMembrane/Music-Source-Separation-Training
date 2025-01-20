#!/bin/bash
python train.py \                                                             
        --model_type mel_band_roformer \                                          
        --config_path configs/config_mel_band_roformer_vocals.yaml \              
        --start_check_point results/model.ckpt \                                  
        --results_path results/ \                                                 
        --data_path 'datasets/musdb18hq/train' 'datasets/dataset2' \                     
        --valid_path datasets/musdb18hq/test \                                    
        --num_workers 4 \                                                         
        --device_ids 0                                                            

