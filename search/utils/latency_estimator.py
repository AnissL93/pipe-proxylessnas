# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import yaml
import os
import sys
import pandas as pd

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def download_url(url, model_dir="~/.torch/proxyless_nas", overwrite=False):
    target_dir = url.split("//")[-1]
    target_dir = os.path.dirname(target_dir)
    model_dir = os.path.expanduser(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split("/")[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file


class LatencyEstimator(object):
    def __init__(
        self,
        url="https://raw.githubusercontent.com/han-cai/files/master/proxylessnas/mobile_trim.yaml",
        model_dir="~/.torch/proxyless_nas",
        latency_file=None,
    ):
        if url is not None:
            fname = download_url(url, model_dir=model_dir, overwrite=True)

            with open(fname, "r") as fp:
                self.lut = yaml.load(fp, Loader=yaml.Loader)
        elif latency_file is not None:
            with open(latency_file, "r") as fp:
                if str(latency_file).endswith("yaml"):
                    self.lut = yaml.load(fp, Loader=yaml.Loader)
                elif str(latency_file).endswith("csv"):
                    data = pd.read_csv(latency_file)
                    self.lut = {}
                    for index, ins in data.iterrows():
                        key = ins["block_name"]
                        cpub = ins["cpu_b"]
                        cpus = ins["cpu_s"]
                        gpu = ins["gpu"]
                        self.lut[key] = {
                            "cpu_b": cpub,
                            "cpu_s": cpus,
                            "gpu": gpu,
                        }
            pass

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return "x".join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def get_key(
        self,
        ltype: str,
        _input,
        output,
        expand=None,
        kernel=None,
        stride=None,
        idskip=None,
    ):
        infos = [
            ltype,
            "input:%s" % self.repr_shape(_input),
            "output:%s" % self.repr_shape(output),
        ]

        if ltype in ("expanded_conv",):
            assert None not in (expand, kernel, stride, idskip)
            infos += [
                "expand:%d" % expand,
                "kernel:%d" % kernel,
                "stride:%d" % stride,
                "idskip:%d" % idskip,
            ]
        key = "-".join(infos)
        return key

    def predict(
        self,
        ltype: str,
        _input,
        output,
        expand=None,
        kernel=None,
        stride=None,
        idskip=None,
    ):
        """
        :param ltype:
            Layer type must be one of the followings
                1. `Conv`: The initial 3x3 conv with stride 2.
                2. `Conv_1`: The upsample 1x1 conv that increases num_filters by 4 times.
                3. `Logits`: All operations after `Conv_1`.
                4. `expanded_conv`: MobileInvertedResidual
        :param _input: input shape (h, w, #channels)
        :param output: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param kernel: kernel size
        :param stride:
        :param idskip: indicate whether has the residual connection
        """
        key = self.get_key(ltype, _input, output, expand, kernel, stride, idskip)
        if key in self.lut.keys():
            # print(f"Get time from cpu_b {self.lut[key]['cpu_b']=}")
            return self.lut[key]["cpu_b"]
        else:
            # print(f"Get time from cpu_b {key}")
            exit(-1)


class HeteroLatencyEstimator(LatencyEstimator):
    def __init__(self, fname):
        self.record_config = open("data/missed_config.txt", "w+")

        with open(fname, "r") as fp:
            if str(fname).endswith("yaml"):
                self.lut = yaml.load(fp, Loader=yaml.Loader)
            elif str(fname).endswith("csv"):
                data = pd.read_csv(fname)
                self.lut = {}
                for index, ins in data.iterrows():
                    key = ins["block_name"]
                    cpub = ins["cpu_b"]
                    cpus = ins["cpu_s"]
                    gpu = ins["gpu"]
                    self.lut[key] = {
                        "cpu_b": cpub,
                        "cpu_s": cpus,
                        "gpu": gpu,
                    }
                self.procs = ["cpu_b", "cpu_s", "gpu"]

    @property
    def n_proc(self):
        return len(self.procs)

    def get_procs(self):
        return self.procs

    def predict_for_all_procs(
        self,
        ltype: str,
        _input,
        output,
        expand=None,
        kernel=None,
        stride=None,
        idskip=None,
    ):
        return self.get_key(ltype, _input, output, expand, kernel, stride, idskip), {
            "cpu_b": self.predict(
                "cpu_b", ltype, _input, output, expand, kernel, stride, idskip
            ),
            "cpu_s": self.predict(
                "cpu_s", ltype, _input, output, expand, kernel, stride, idskip
            ),
            "gpu": self.predict(
                "gpu", ltype, _input, output, expand, kernel, stride, idskip
            ),
        }

    def predict(
        self,
        proc: str,
        ltype: str,
        _input,
        output,
        expand=None,
        kernel=None,
        stride=None,
        idskip=None,
    ):
        key = self.get_key(ltype, _input, output, expand, kernel, stride, idskip)
        if key in self.lut.keys():
            l = self.lut[key][proc]
            # print(f"Found Get {key=} of {proc=}, latency = {l}")
            return l
        else:
            # print(f"Key {key} not found in {proc} lookup table")
            self.record_config.write(key)
            self.record_config.write("\n")

            default_lat = {"cpu_b": 0.203079, "cpu_s": 0.292553, "gpu": 1.023}
            return default_lat[proc]


if __name__ == "__main__":
    est = HeteroLatencyEstimator("data/latency.yaml")
    s = est.predict(
        "gpu",
        "expanded_conv",
        _input=(112, 112, 16),
        output=(56, 56, 24),
        expand=3,
        kernel=5,
        stride=2,
        idskip=0,
    )

    est = HeteroLatencyEstimator("data/cifar10/latency_cifar10.csv")
    s = est.predict("gpu", "Conv", _input=(32, 32, 3), output=(16, 16, 32))
    # print(s)
    s = est.predict("cpu_b", "Conv", _input=(32, 32, 3), output=(16, 16, 32))
    print(s)

    # with open("arm_perf.yaml", 'r') as fp:
    #     x = yaml.load(fp, Loader=yaml.Loader)
    #     print(x.keys())
    #     for k in x.keys():
    #         x[k]["big"] = 0.4
    #         x[k]["small"] = 0.2

    #     with open("a.yaml", "w") as ff:
    #         yaml.dump(x, ff, Dumper=yaml.Dumper)
