#!/usr/bin/env python
import torch

from torchview import draw_graph

from pathlib import Path
import sys
sys.path.append("/home/huiying/projects/nas/pipeproxylessnas/search")

from from_net import read_from_config

from fire import Fire

def to_onnx(path: str, out_path : str):
    net = read_from_config(Path(path))
    torch.onnx.export(net, torch.randn(1, 3, 32, 32), out_path)

def to_pt(path: str, out_path : str):
    net = read_from_config(Path(path))
    torch.save(net, out_path)

def to_png(path: str, out_path : str):
    net = read_from_config(Path(path))
    draw_graph(model = net, input_size=(1, 3, 32, 32), save_graph=True, filename=out_path, depth=5, expand_nested=True)

def to_file(path : str, out_path : str, file_type : str):
    if file_type == "onnx":
        to_onnx(path, out_path + ".onnx")
    elif file_type == "pt":
        to_pt(path, out_path + ".pt")
    elif file_type == "png":
        to_png(path, out_path + ".pdf")

Fire(to_file)
