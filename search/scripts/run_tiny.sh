#!/bin/bash
python tiny_image_search.py --path tiny_origin_cpu_only/ --target_hardware mobile --grad_binary_mode two --n_cell_stages 3,3,3,3 --stride_stages 2,2,2,1  --n_epochs 3000 --resume --last_channel=512 --latency_data data/tiny/latency.csv 