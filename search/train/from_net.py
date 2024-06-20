#!/usr/bin/env python
import sys
from pathlib import Path
import torch
import json

sys.path.append("/home/huiying/projects/nas/pipeproxylessnas/search")

from models.normal_nets.proxyless_nets import ProxylessNASNets

def read_from_config(path : Path, pretrain=False, dropout=0.):
    net_config = path / "net.config"
    weights = path / "init"

    net_config_json = json.load(open(net_config, 'r'))
    net = ProxylessNASNets.build_from_config(net_config_json)

    if pretrain and weights.exists():
        model = torch.load(weights)
        state_dict = model["state_dict"]
        net.load_state_dict(state_dict)

    return net


if __name__ == "__main__":
    net = read_from_config(Path("./cifar_origin1/learned_net"))
    print(net)
