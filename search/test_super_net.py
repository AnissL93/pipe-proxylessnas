from models.super_nets.super_proxyless import SuperProxylessNASNets, MixedEdge
from models.super_nets.pipeline_super_proxyless import PipelineSuperProxylessNASNets, PipelineConfig
from modules.mix_op import LatencyEstimator, HeteroLatencyEstimator
from modules.hetero_mix_op import HeteroMixEdge
from utils import LatencyEstimator, PipelineConfig, cpu_config
from utils.find_split_point import *
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
from modules.block_info import *
from collections import OrderedDict
from pprint import pprint as pp
# import logging
from utils.logging_utils import logger
from torchviz import make_dot
from torchview import draw_graph


def get_supernet_for_tiny_image():
    net = PipelineSuperProxylessNASNets(
        width_stages=[24, 40, 80, 96, 192, 320],
        n_cell_stages=[3, 3, 3, 3, 1],
        stride_stages=[2, 2, 2, 2, 1],
        conv_candidates=['3x3_MBConv3', '3x3_MBConv6',
                         '5x5_MBConv3', '5x5_MBConv6'],
        n_classes=200,
        last_c=512,
        lat_data="data/tiny/latency.csv",
        input_size=64
    )
    net.reset_binary_gates()
    return net

if __name__ == "__main__":
    cifar10_supernet = get_supernet_for_tiny_image()
    print(cifar10_supernet.get_expected_latency_for_all_proc(True))
    print(cifar10_supernet.get_expected_latency_for_all_proc(False))

    print(cifar10_supernet.current_pipeline_latency())
    print(cifar10_supernet.latency_loss())

    # cifar10_supernet.train()

    # data = torch.rand(1, 3, 32, 32)
    # make_dot(cifar10_supernet(data), params=dict(
    #     list(cifar10_supernet.named_parameters()))).render("rnn_torchviz", format="png")

    # draw_graph(cifar10_supernet, input_size=(1, 3, 32, 32), depth=10,
    #            save_graph=True, filename="cifar10_supernet.png")

    l = cifar10_supernet.latency_loss()
    config, lat = cifar10_supernet.get_best_config()
    print(config)
    print(lat)