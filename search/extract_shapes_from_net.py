from models.super_nets.pipeline_super_proxyless import PipelineSuperProxylessNASNets
from pprint import pprint as pp
import json

def gen_mbconv_config(hin, cin, cout, s, k, exp, skip):

    def get_pad(in_size, out_size, stride, kernel):
        padded_in_size = (out_size - 1) * stride + kernel
        assert padded_in_size >= in_size
        p = padded_in_size - in_size
        assert int(((in_size + p) - kernel)/stride + 1) == out_size
        l_pad = int((padded_in_size - in_size)/2)
        r_pad = p - l_pad
        return l_pad, r_pad

    hout = hin // s
    if skip == 1:
        if hin != hout or cin != cout:
            return {}

    pads = get_pad(hin, hout, s, k)
    key = f"expanded_conv-input:{hin}x{hin}x{cin}-output:{hout}x{hout}x{cout}-expand:{exp}-kernel:{k}-stride:{s}-idskip:{skip}"
    return { key: {
        "type": "expanded_conv",
        "input": [hin, hin, cin],
        "output": [hout, hout, cout],
                "expand": exp,
                "kernel": k,
                "stride": s,
                "idskip": skip,
                "pad_l": pads[0],
                "pad_r": pads[1]
    }}

def enumerate_configs():
    feat_c = [16, 24, 32, 40, 80, 96, 192, 320, 512]
    feat_pair = [(feat_c[i], feat_c[j]) for i in range(len(feat_c)) for j in range(i+1, len(feat_c))]
    feat_pair += [(feat_c[i], feat_c[i]) for i in range(len(feat_c))]

    hw_cand = [32, 16, 8, 4, 2]
    stride = [1,2]
    kernel = [3, 5, 7]
    expand_ratio = [3, 6]
    skip = [0, 1]

    configs = {}

    print(f"Total configs: {len(feat_pair) * len(hw_cand) * 2 * 3 * 2 * 2}")

    i = 0
    for feats in feat_pair:
        for hw in hw_cand:
            for s in stride:
                for k in kernel:
                    for exp in expand_ratio:
                        for sk in skip:
                            i+=1
                            x = gen_mbconv_config(hw, feats[0], feats[1], s, k, exp, sk)
                            configs.update(x)

    return configs
    

def get_supernet_for_cifar10():
    net = PipelineSuperProxylessNASNets(
        width_stages=[24, 40, 80, 96, 192, 320],
        n_cell_stages=[2, 2, 2, 2, 1],
        stride_stages=[2, 2, 1, 2, 1],
        conv_candidates=['3x3_MBConv3', '3x3_MBConv6',
                         '5x5_MBConv3', '5x5_MBConv6'],
        n_classes=200,
        last_c=512,
        lat_data="data/tiny_image/latency.csv",
        input_size=64
    )
    net.reset_binary_gates()
    return net

def get_supernet_for_tiny_image():
    net = PipelineSuperProxylessNASNets(
        width_stages=[24, 40, 80, 96, 192, 320],
        n_cell_stages=[3, 3, 3, 3],
        stride_stages=[2, 2, 2, 1],
        conv_candidates=['3x3_MBConv3', '3x3_MBConv6',
                         '5x5_MBConv3', '5x5_MBConv6'],
        n_classes=200,
        last_c=512,
        lat_data="data/tiny/latency.csv",
        input_size=64
    )
    net.reset_binary_gates()
    return net


def extract_shapes(net : PipelineSuperProxylessNASNets, out_file):

    info = enumerate_configs()

    with open(out_file, "w") as fp:
        json.dump(info, fp, indent=4)
    
    pass

# f1 is larget than f2
def remove_redundant(f1, f2):
    ret = {}
    with open(f1, "r") as fp1:
        d1 = json.loads(fp1.read())
        with open(f2, "r") as fp2:
            d2 = json.loads(fp2.read())
            # remove item if both d1 and d2 has it.
            for k in d1.keys():
                if k not in d2:
                    ret[k] = d1[k]

    return ret


    pass

if __name__ == "__main__":
    # extract_shapes(get_supernet_for_tiny_image(), "tiny_image.json")
    r = remove_redundant("tiny_image.json", "cifar10_p.json")
    print(len(r.keys()))
    with open("tiny.json", "w") as fp:
        json.dump(r, fp, indent=4)
    # configs = enumerate_configs()
    # print(len(configs.keys()))