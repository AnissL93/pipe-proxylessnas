import torch
from collections import OrderedDict
from pprint import pprint as pp

def get_orders(n):
    """Enumerate all combinations in a Cnn manner
    """
    possible = range(n)
    order = []

    def _get_order(cur, possib):
        if len(possib) == 0:
            order.append(cur)
            return
        else:
            for p in possib:
                new_p = [x for x in possib if x != p]
                _get_order(cur + [p], new_p)
    _get_order([], possible)
    return order

def sp_to_range(sp, n):
    """extend split points into a list of ranges

    e.g.,
    sp = (3,6), n = 10, will return:
    [(0,4), (4, 7), (7, 10)]

    (0,4) means 0 <= i < 4
    """
    ret = []
    st = 0
    ed = 0
    for i in range(len(sp)+1):
        if i == 0:
            ret.append((0, sp[0]+1))
        elif i==len(sp):
            ret.append((sp[i-1]+1, n))
        else:
            ret.append((sp[i-1]+1, sp[i]+1))
    return ret

def sps_to_ranges(sps, n):
    ret = []
    for sp in sps:
        ret.append(sp_to_range(sp, n))

    return ret

def find_split_points(latency : torch.Tensor):
    n_block = latency.size()[0]
    sp1 = [i for i in range(n_block-1)]
    all_sp = []
    for sp in sp1:
        for sp2 in range(sp+1, n_block-1):
            if sp2 ==sp:
                continue
            all_sp.append((sp, sp2))

    return sps_to_ranges(all_sp, n_block)

class Config:

    def __init__(self, ranges : list, proc_order : list) -> None:
        self.ranges = ranges
        self.proc_order = proc_order
        pass

    def latency(self, all_lat):
        lat = torch.zeros(len(self.proc_order))
        # print(f"{self.proc_order}, range {self.ranges}")
        for i in range(len(self.proc_order)):
            r = self.ranges[i]
            p = self.proc_order[i]
            # print(f"range {r}, proc {p}")
            for j in range(r[0], r[1]):
                lat[p] += all_lat[j][p]
            # print(f"lat is {lat}")
        return lat

    def __str__(self) -> str:
        return f"{self.ranges}-{self.proc_order}"

class PipelineConfig:

    def __init__(self, latency) -> None:
        self.n_block = latency.size()[0]
        self.n_proc = latency.size()[1]
        self.latency = latency
        pass

    @property
    def enumerate_configs(self):
        all_sp = find_split_points(self.latency)
        orders = get_orders(self.n_proc)
        ret = []
        for sp in all_sp:
            for order in orders:
                ret.append(Config(sp, order))

        return ret

    @property
    def best_config(self):
        all_lat = {}
        best_v = 1000000
        best_lat = None
        best_k = None
        for i, config in enumerate(self.enumerate_configs):
            clat = config.latency(self.latency)
            max_lat = torch.max(clat)
            if max_lat < best_v:
                best_v = max_lat
                best_k = config
                best_lat = clat
        return (best_k, best_lat)

if __name__ == "__main__":
    x = torch.rand(10,3)
    c = Config([(0, 8), (8, 9), (9, 10)], [0,2,1])
    print(c.latency(x))
    configs = PipelineConfig(x)
    c = configs.best_config
    print(c[0])
    print(c[1])

# pp(ranges)

# pp(est_latency(x, False))

# x = all_est_latency(x, True)
# print("=============dfsdfds\n\n")
# print(x)
