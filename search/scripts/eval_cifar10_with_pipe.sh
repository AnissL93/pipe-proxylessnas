#!/bin/bash
python cifar10_arch_search.py --eval_pipe_only=1 --path cifar_search_origin_cpu_only-small/ --target_hardware mobile  --grad_binary_mode two --n_cell_stages 1,1,1,1 --stride_stages 2,2,2,1  --n_epochs 3000 --resume
