#!/bin/bash
p=$1
python tiny_image_search.py --path ${p} --target_hardware soc --grad_reg_loss_type pipeline --grad_binary_mode two --n_cell_stages 3,3,3,3 --stride_stages 2,2,2,1  --n_epochs 3000 --resume  --last_channel=512 --latency_data data/tiny/latency.csv --weight_only=True