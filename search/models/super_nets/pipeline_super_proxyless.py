import sys

sys.path.append("/home/huiying/projects/nas/pipeproxylessnas/search")
from models.super_nets.super_proxyless import SuperProxylessNASNets, MixedEdge
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


class PipelineSuperProxylessNASNets(SuperProxylessNASNets):
    def __init__(
        self,
        width_stages,
        n_cell_stages,
        conv_candidates,
        stride_stages,
        n_classes=1000,
        width_mult=1,
        bn_param=(0.1, 1e-3),
        dropout_rate=0,
        last_c=1280,
        n_proc=3,
        lat_data=None,
        input_size=None,
    ):
        super().__init__(
            width_stages,
            n_cell_stages,
            conv_candidates,
            stride_stages,
            n_classes,
            width_mult,
            bn_param,
            dropout_rate,
            last_c,
            input_size,
        )

        self.PROC_param = Parameter(torch.Tensor(self.block_num, n_proc).to("cuda"))
        self.PROC_param_binary = Parameter(
            torch.Tensor(self.block_num, n_proc).to("cuda")
        )
        # info for get latency
        self.block_info = []
        self.n_cell_stages = n_cell_stages
        self.stride_stages = stride_stages
        self.n_proc = n_proc
        # self.PROC_param = Parameter(torch.Tensor(n_proc))  # architecture parameters
        self.latency_model = HeteroLatencyEstimator(lat_data)
        all_blocks = [self.first_conv]
        for b in self.blocks:
            all_blocks.append(b)
        all_blocks.append(self.feature_mix_layer)
        all_blocks.append(self.classifier)

    @property
    def probs_over_processors(self):
        probs = F.softmax(self.PROC_param, dim=-1)
        return probs.to("cuda")

    def get_block_info(self, in_h, in_w):
        dicts = {}

        # first conv
        d = from_module(self.first_conv, in_h, in_w, in_h // 2, in_w // 2)
        dicts.update(d)

        in_h = in_h // 2
        in_w = in_w // 2

        # first block
        first_block = self.blocks[0]
        d = from_module(first_block.mobile_inverted_conv, in_h, in_w, in_h, in_w)
        dicts.update(d)

        block_idx = 1
        for n_cell, s in zip(self.n_cell_stages, self.stride_stages):
            for i in range(n_cell):
                blk = self.blocks[block_idx]
                if i == 0:
                    stride = s
                else:
                    stride = 1

                out_h = in_h // stride
                out_w = in_w // stride

                d = from_module(blk.mobile_inverted_conv, in_h, in_w, out_h, out_w)
                dicts.update(d)
                in_h = out_h
                in_w = out_w
                block_idx += 1

        d = from_module(self.feature_mix_layer, in_h, in_w, in_h, in_w)
        dicts.update(d)
        d = from_module(self.classifier, in_h, in_w, 1, 1)
        dicts.update(d)
        return dicts

    @property
    def current_assignment(self):
        binary = self.proc_binary
        return torch.argmax(binary, dim=1)

    @property
    def proc_binary(self):
        proc = self.probs_over_processors.detach()
        binary = torch.zeros_like(proc)
        for i in range(self.block_num):
            max_proc = torch.argmax(proc[i])
            for j in range(self.n_proc):
                if j == max_proc.item():
                    binary[i][int(max_proc.item())] = 1.0
                else:
                    binary[i][j] = 0.0
        return binary

    @property
    def chosen_proc(self):
        return torch.argmax(self.probs_over_processors, dim=-1).to(dtype=torch.int)

    def sum_of_latency(self, lat_with_prob: torch.Tensor):
        return lat_with_prob.sum(dim=0)

    @property
    def block_num(self):
        # first_conv + conv_1 + logits + len(block)
        return 3 + len(self.blocks)

    def get_best_config(self):
        el = self.get_expected_latency_for_all_proc(True)
        best_config = find_best_config(el)
        return str(best_config[0]), best_config[1]

    def pipeline_latency(self, lat_for_all):
        """Expected latency for each processors"""
        # Find latency for all proc, and find the best configuration
        best_config = find_best_config(lat_for_all)

        # Find the expected latency involving the probability of assigning processor
        lat_with_proc_prob = torch.mul(lat_for_all, self.probs_over_processors)

        # print("best configurations: ", str(best_config[0]))

        # get the EL of each processor while applying best configuration
        return best_config[0].get_latency(lat_with_proc_prob)

    def get_latency_with_proc_prob(self, all_lat):
        return torch.mul(all_lat, self.probs_over_processors)

    def current_pipeline_latency(self, do_max=True):
        # print("weights:  ", self.probs_over_processors)
        all_proc_binary = self.get_expected_latency_for_all_proc(True).detach()
        lat = torch.sum(torch.mul(self.proc_binary, all_proc_binary), dim=0)
        if do_max:
            return torch.max(lat)
        else:
            return lat

    def get_expected_latency_for_all_proc(self, use_binary=False):
        inhw = self.input_size
        lat = [None for i in range(self.block_num)]
        block_idx = 0

        def add_latency(l, b):
            torch_l = []
            for ll in list(l):
                if isinstance(ll, torch.Tensor):
                    torch_l.append(ll)
                else:
                    torch_l.append(
                        torch.tensor(ll, dtype=torch.float, requires_grad=True)
                    )

            lat[b] = torch.stack(torch_l).to("cuda")
            return b + 1

        # first conv
        _, l = self.latency_model.predict_for_all_procs(
            "Conv",
            [inhw, inhw, 3],
            [inhw // 2, inhw // 2, self.first_conv.out_channels],
        )
        block_idx = add_latency(l.values(), block_idx)

        # blocks
        fsize = inhw // 2
        for block in self.blocks:
            shortcut = block.shortcut
            if shortcut is None or shortcut.is_zero_layer():
                idskip = 0
            else:
                idskip = 1

            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                if not mb_conv.is_zero_layer():
                    out_fz = fsize // mb_conv.stride
                    _, l = self.latency_model.predict_for_all_procs(
                        "expanded_conv",
                        [fsize, fsize, mb_conv.in_channels],
                        [out_fz, out_fz, mb_conv.out_channels],
                        expand=mb_conv.expand_ratio,
                        kernel=mb_conv.kernel_size,
                        stride=mb_conv.stride,
                        idskip=idskip,
                    )
                    block_idx = add_latency(l.values(), block_idx)
                    fsize = out_fz
                continue

            probs_over_ops = mb_conv.probs_over_ops
            if use_binary:
                chosen_path = torch.argmax(probs_over_ops.detach())

            out_fsize = fsize
            block_lat = [0.0 for k in range(self.latency_model.n_proc)]
            for i, op in enumerate(mb_conv.candidate_ops):
                if use_binary and i != chosen_path:
                    continue

                if op is None or op.is_zero_layer():
                    continue
                out_fsize = fsize // op.stride
                _, op_latency = self.latency_model.predict_for_all_procs(
                    "expanded_conv",
                    [fsize, fsize, op.in_channels],
                    [out_fsize, out_fsize, op.out_channels],
                    expand=op.expand_ratio,
                    kernel=op.kernel_size,
                    stride=op.stride,
                    idskip=idskip,
                )
                if use_binary:
                    for p_idx in range(self.latency_model.n_proc):
                        block_lat[p_idx] += op_latency[
                            self.latency_model.get_procs()[p_idx]
                        ]
                else:
                    prob_of_this_path = probs_over_ops[i]
                    for p_idx in range(self.latency_model.n_proc):
                        block_lat[p_idx] += (
                            op_latency[self.latency_model.get_procs()[p_idx]]
                            * prob_of_this_path
                        )

            block_idx = add_latency(block_lat, block_idx)
            fsize = out_fsize

        _, l = self.latency_model.predict_for_all_procs(
            "Conv_1",
            [fsize, fsize, self.feature_mix_layer.in_channels],
            [fsize, fsize, self.feature_mix_layer.out_channels],
        )
        block_idx = add_latency(l.values(), block_idx)
        # classifier
        _, l = self.latency_model.predict_for_all_procs(
            "Logits",
            [fsize, fsize, self.classifier.in_features],
            [self.classifier.out_features],
        )
        block_idx = add_latency(l.values(), block_idx)
        return torch.stack(lat)

    def latency_loss(self):
        """Latency loss
        Including following parts:
        1. minimize the gap between different stages
        2. minimize the latency of first stage
        3. minimize the gap between sum_all_latency and pipeline_latency (to force continuous)
        """

        in_hw = self.input_size

        el = self.get_expected_latency_for_all_proc()
        pipe_lat = self.pipeline_latency(el)

        loss_12 = torch.tensor(0.0, requires_grad=True)
        for i in range(self.n_proc):
            if i == 0:
                loss_12 = pipe_lat[0]
            else:
                l2_norm = torch.norm(pipe_lat[i] - pipe_lat[i - 1])
                loss_12 = torch.add(loss_12, l2_norm)

        # part 3
        sum_lat = self.sum_of_latency(self.get_latency_with_proc_prob(el))
        loss_diff = torch.sum(torch.sub(sum_lat, pipe_lat) ** 2)
        return loss_12, loss_diff

    # def expected_latency(self, latency_model: HeteroLatencyEstimator):
    #     """return latency for different stages according to split point
    #     {
    #         (0, 10) : 0.3,
    #         (10, 25) : 0.5
    #     }

    #     """
    #     in_h = 32
    #     in_w = 32
    #     expected_latency = {k: 0 for k in self.pipe_config.split_points}

    #     # first conv
    #     sp_first_conv = self.pipe_config.get_stage(0)
    #     proc = self.pipe_config.get_proc(sp_first_conv)
    #     expected_latency[sp_first_conv] += latency_model.predict(proc, 'Conv', [in_h, in_w, 3],
    #                                                              [in_h // 2, in_w // 2, self.first_conv.out_channels])

    #     # blocks
    #     fsize = in_h
    #     block_index = 1
    #     for block in self.blocks:
    #         sp = self.pipe_config.get_stage(block_index)
    #         proc = self.pipe_config.get_proc(sp)
    #         block_index += 1

    #         shortcut = block.shortcut
    #         if shortcut is None or shortcut.is_zero_layer():
    #             idskip = 0
    #         else:
    #             idskip = 1

    #         mb_conv = block.mobile_inverted_conv
    #         if not isinstance(mb_conv, MixedEdge):
    #             if not mb_conv.is_zero_layer():
    #                 out_fz = fsize // mb_conv.stride
    #                 op_latency = latency_model.predict(
    #                     proc,
    #                     'expanded_conv', [fsize, fsize, mb_conv.in_channels], [
    #                         out_fz, out_fz, mb_conv.out_channels],
    #                     expand=mb_conv.expand_ratio, kernel=mb_conv.kernel_size, stride=mb_conv.stride, idskip=idskip
    #                 )
    #                 expected_latency[sp] = expected_latency[sp] + op_latency
    #                 fsize = out_fz
    #             continue

    #         probs_over_ops = mb_conv.current_prob_over_ops
    #         out_fsize = fsize
    #         block_lat = 0
    #         for i, op in enumerate(mb_conv.candidate_ops):
    #             if op is None or op.is_zero_layer():
    #                 continue
    #             out_fsize = fsize // op.stride
    #             op_latency = latency_model.predict(
    #                 proc,
    #                 'expanded_conv', [fsize, fsize, op.in_channels], [
    #                     out_fsize, out_fsize, op.out_channels],
    #                 expand=op.expand_ratio, kernel=op.kernel_size, stride=op.stride, idskip=idskip
    #             )
    #             block_lat += op_latency * probs_over_ops[i]
    #             # expected_latency[sp] = expected_latency[sp] + op_latency * probs_over_ops[i]

    #         print(f">>> {block_index}: {block_lat}")
    #         expected_latency[sp] += block_lat
    #         fsize = out_fsize

    #     sp = self.pipe_config.get_stage(block_index)
    #     proc = self.pipe_config.get_proc(sp)
    #     expected_latency[sp] += latency_model.predict(
    #         proc,
    #         'Conv_1', [7, 7, self.feature_mix_layer.in_channels], [
    #             7, 7, self.feature_mix_layer.out_channels]
    #     )
    #     block_index += 1
    #     sp = self.pipe_config.get_stage(block_index)
    #     proc = self.pipe_config.get_proc(sp)

    #     # classifier
    #     expected_latency[sp] += latency_model.predict(
    #         proc,
    #         'Logits', [7, 7, self.classifier.in_features], [
    #             self.classifier.out_features]
    #     )

    #     return expected_latency


# args.bn_momentum=0.1
# args.bn_eps=0.001


def get_supernet_for_tiny_image():
    net = PipelineSuperProxylessNASNets(
        width_stages=[24, 40, 80, 96, 192, 320],
        n_cell_stages=[3, 3, 3, 3, 1],
        stride_stages=[2, 2, 2, 2, 1],
        conv_candidates=["3x3_MBConv3", "3x3_MBConv6", "5x5_MBConv3", "5x5_MBConv6"],
        n_classes=200,
        last_c=512,
        lat_data="data/tiny/latency.csv",
        input_size=64,
    )
    net.reset_binary_gates()
    return net


def get_supernet_for_cifar10():
    net = PipelineSuperProxylessNASNets(
        width_stages=[24, 40, 80, 96, 192, 320],
        n_cell_stages=[2, 2, 2, 2, 1],
        stride_stages=[2, 2, 1, 2, 1],
        conv_candidates=["3x3_MBConv3", "3x3_MBConv6", "5x5_MBConv3", "5x5_MBConv6"],
        n_classes=10,
        last_c=512,
        lat_data="data/cifar10/latency.csv",
        input_size=32,
    )
    net.reset_binary_gates()
    return net


if __name__ == "__main__":
    cifar10_supernet = get_supernet_for_cifar10()
    print(cifar10_supernet)
    # print(cifar10_supernet.get_expected_latency_for_all_proc(True))
    # print(cifar10_supernet.get_expected_latency_for_all_proc(False))

    # print(cifar10_supernet.current_pipeline_latency())
    # print(cifar10_supernet.latency_loss())

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

    # print(l)
    # net_params = cifar10_supernet.parameters()
    # print(net_params)
    # optimizer = torch.optim.SGD(net_params, lr=0.025, momentum=0.9, nesterov=True,
    #                                         weight_decay=4e5)
    # optimizer.zero_grad()

    # l.backward()
    # optimizer.step()
    # print(cifar10_supernet.PROC_param)

    # print(cifar10_supernet.chosen_proc)

    # d = cifar10_supernet.get_block_info(32, 32)
    # print(cifar10_supernet.proc_param.get_latency_for_conv(32, 0))
    # print(cifar10_supernet.proc_param.get_latency_for_logits(2, -1))
    # for i in range(cifar10_supernet.block_num):
    #     f_size = cifar10_supernet.proc_param.get_outhw(f_size, i)
    #     info(f"{i=}: {f_size=}")
    # pp(cifar10_supernet.get_expected_latency_for_all_proc(32))
    # pp(cifar10_supernet.proc_param.probs_over_processors)
    # pp(cifar10_supernet.pipeline_latency(32))
    # pp(cifar10_supernet.continuous_penelty())
    # pp(cifar10_supernet.latency_loss(32))
