import yaml
from pprint import pprint as pp
import pandas as pd
import json

"""
Extract shapes from a super network
"""
layer_types = [
    "Conv_1",
    "Conv",
    "Logits",
    "expanded_conv"
]

def get_pad(in_size, out_size, stride, kernel):
    padded_in_size = (out_size - 1) * stride + kernel
    assert padded_in_size >= in_size
    p = padded_in_size - in_size
    assert int(((in_size + p) - kernel)/stride + 1) == out_size
    l_pad = int((padded_in_size - in_size)/2)
    r_pad = p - l_pad
    return l_pad, r_pad

def get_shape(x):
    shape = x.split(":")[1].split("x")
    return [int(s) for s in shape]


def get_int(x):
    return int(x.split(":")[1])


def extract(layer):
    layer = str(layer)
    params = layer.split("-")
    type = params[0]
    if type == "Conv_1" or type == "Conv" or type == "Logits":
        input = get_shape(params[1])
        output = get_shape(params[2])
        content = {
            "type": type,
            "input": input,
            "output": output,
        }
        return content
    elif type == "expanded_conv":
        input, output, expand, kernel, stride, idskip = params[1:]

        input_shape = get_shape(input)
        output_shape = get_shape(output)
        expand = get_int(expand)
        kernel = get_int(kernel)
        stride = get_int(stride)
        idskip = get_int(idskip)
        pads = get_pad(input_shape[0], output_shape[0], stride, kernel)
        content = {
            "type": type,
            "input": input_shape,
            "output": output_shape,
            "expand": expand,
            "kernel": kernel,
            "stride": stride,
            "idskip": idskip,
            "pad_l": pads[0],
            "pad_r": pads[1],
        }
        return content
    else:
        print(f"wrong type: {type}")
        exit(-1)


def extract_layer_params(keys):
    all_param = {}
    for k in keys:
        print(f"./graph_expand_conv --config=params.json --name={k}")
        param = extract(k)
        all_param[k] = param
    return all_param


ids = [
    "Conv-input:224x224x3-output:112x112x32",
    "expanded_conv-input:112x112x16-output:56x56x32-expand:3-kernel:5-stride:2-idskip:0",
    "expanded_conv-input:56x56x32-output:56x56x32-expand:3-kernel:3-stride:1-idskip:1",
    "expanded_conv-input:56x56x32-output:28x28x40-expand:3-kernel:7-stride:2-idskip:0"
]

all_params = extract_layer_params(ids)

print(all_params)


# with open("/home/huiying/.torch/proxyless_nas/raw.githubusercontent.com/han-cai/files/master/proxylessnas/mobile_trim.yaml", "r") as fp:
#     content = yaml.load(fp, Loader=yaml.Loader)
#     l = list(content.keys())
#     param = extract_layer_params(content)
#     with open("params.json", "w") as fp:
#         json.dump(param, fp, indent=4)

# with open("/home/huiying/.torch/proxyless_nas/raw.githubusercontent.com/han-cai/files/master/proxylessnas/mobile_trim.yaml", "r") as fp:
#     big = pd.read_csv("data/blocks_cpu_b.csv")
#     small = pd.read_csv("data/blocks_cpu_s.csv")
#     gpu = pd.read_csv("data/blocks_gpu.csv")
#
#     content = yaml.load(fp, Loader=yaml.Loader)
#     print(content)
#
#     for i in range(len(big)):
#         print(i)
#         block_name = big.loc[i]["block_name"]
#         print(big.loc[i]["latency(ms)"])
#         content[block_name]["big"] = float(big.loc[i]["latency(ms)"])
#         print(content[block_name])
#
#     for i in range(len(small)):
#         block_name = small.loc[i]["block_name"]
#         content[block_name]["small"] = float(small.loc[i]["latency(ms)"])
#     for i in range(len(gpu)):
#         block_name = gpu.loc[i]["block_name"]
#         content[block_name]["gpu"] = float(gpu.loc[i]["latency(ms)"])
#
#     print(content)
#     with open("data/latency.yaml", "w+") as fp_out:
#         yaml.dump(content, fp_out, Dumper=yaml.Dumper)
#


