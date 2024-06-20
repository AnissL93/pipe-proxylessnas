# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import numpy as np

from torch.nn.parameter import Parameter
import torch.nn.functional as F

from modules.layers import *
from modules.mix_op import MixedEdge


class HeteroMixEdge(MixedEdge):

    def __init__(self, candidate_ops, n_proc=3):
        super().__init__(candidate_ops)
        self.PROC_param = Parameter(torch.Tensor(n_proc))
        self.PROC_param_binary = Parameter(torch.Tensor(n_proc))

    @property
    def probs_over_procs(self):
        probs = F.softmax(self.PROC_param_binary, dim=0)
        return probs

    @property
    def chosen_proc_index(self):
        probs = self.probs_over_procs.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    def get_latency_with_proc(self,
                    in_hw,
                    latency_model: HeteroLatencyEstimator):
        """
        calculate latency for the chosen candidate operation
        i.e., lat_cpu * prob_of_cpu + lat_gpu * prob_of_gpu + ...
        """
        cand = self.candidate_ops[self.chosen_index[0]]
        if cand is None or cand.is_zero_layer():
            return 0
        else:
            out_size = in_hw // cand.stride
            _, all_latency = latency_model.predict_for_all_procs("expanded_conv", [in_hw, in_hw, cand.in_channels], [
                                                  out_size, out_size, cand.out_channels], expand=cand.expand_ratio, kernel=cand.kernel_size, stride=cand.stride, idskip=cand.idskip)

            lat = 0
            for i, p in enumerate(latency_model.get_procs()):
                lat += self.probs_over_procs[i] * all_latency[p]

            return lat
