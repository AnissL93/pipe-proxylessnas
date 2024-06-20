from utils.pytorch_utils import accuracy, AverageMeter
import torch

import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import json

from models.normal_nets.proxyless_nets import *

def get_shape_str(t):
  s = [str(x) for x in t.shape]
  return '-'.join(s)

class SaveOutput:
    def __init__(self, _cond):
        self.outputs = []
        self.cond = _cond

    def __call__(self, module, module_in, module_out):
        if self.cond is None:
            self.outputs.append(module_out)
        elif self.cond(module):
            self.outputs.append(module_out)
        else:
            return

    def clear(self):
        self.outputs = []


def traverse_all_leave(m, hook, prefix = []):
  child_num = len(list(m.children()))
  if child_num == 0:
    hook(m, prefix)
    return
  else:
    for n, child in m.named_children():
      traverse_all_leave(child, hook, prefix + [n])


full_name_map = {}

def register_hook_save_output(module: ProxylessNASNets, out_path=""):
  def register(m, prefix):
    fname = '-'.join(prefix)

    def hook(_module, module_in, module_out):
      p = out_path + "/" + fname + "_" + get_shape_str(module_out) + ".npy"
      np.save(p, module_out.to("cpu").data.numpy())

    m.register_forward_hook(hook)

  traverse_all_leave(module, register, [])

def read_from_config(config_path):
    net_config = config_path + "/net.config"
    weights = config_path + "/init"

    model = torch.load(weights)
    state_dict = model["state_dict"]

    with open(net_config, "r") as fp:
        config = dict(json.load(fp))
        net = ProxylessNASNets.build_from_config(config)
        net.load_state_dict(state_dict)
        net.eval()
        return net


def to_onnx(net, path):
    net.eval()
    print(net)
    x = torch.randn(1, 3, 224, 224)
    torch_out = net(x)

    # Export the model
    torch.onnx.export(net,               # model being run
                      # model input (or a tuple for multiple inputs)
                      x,
                      # where to save the model (can be a file or file-like object)
                      path,
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                      #                 'output' : {0 : 'batch_size'}
                      # }
                      )


def data_provider():
    from data_providers.imagenet import ImagenetDataProvider
    data_provider = ImagenetDataProvider(valid_size=50000)
    return data_provider


def validate(dp, net):
    data = dp.valid
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.eval()
    net.to("cuda")

    # noinspection PyUnresolvedReferences
    with torch.no_grad():
        for i, (images, labels) in enumerate(data):
            images, labels = images.to("cuda"), labels.to("cuda")
            # compute output
            output = net(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # if i == 50:
            #   break
            # measure elapsed time
    return top1.avg, top5.avg


def get_input_output(dp, net, path):
    validate_data = dp.valid
    net = net.to("cuda")
    # register_hook_save_output(net, path)
    # net = net.to("cuda")
    for i, (images, labels) in enumerate(validate_data):
        image = images[10:11]
        image = image.to("cuda")
        output = net(image)
        return image, output

dp = data_provider()


def dump_models(net, out_path, dev_config=None):
    net.to_json(out_path, dev_config)
    i, o = get_input_output(dp, net, out_path)

    i = i.squeeze(dim=0)
    print(i.shape)
    inshape = [str(x) for x in i.shape]
    outshape = [str(x) for x in o.shape]
    pin = f"{out_path}/input_{'-'.join(inshape)}.npy"
    pout = f"{out_path}/output_{'-'.join(outshape)}.npy"
    np.save(pin, i.to("cpu").data.numpy())
    np.save(pout, o.to("cpu").data.numpy())


cpu_gpu_net = read_from_config("search_arch/learned_net_2cpu_gpu")
dump_models(cpu_gpu_net, "searched_nets/pipe_2cpu_gpu", cpu_gpu_config([10, 15]))

cpu_net = read_from_config("search_arch/learned_net_2cpu")
dump_models(cpu_net, "searched_nets/pipe_2cpu", cpu_config([15]))

target_platform = "proxyless_mobile" # proxyless_gpu, proxyless_mobile, proxyless_mobile14 are also avaliable.
model = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=True)
net_proxyless_cpu = torch.hub.load('mit-han-lab/ProxylessNAS', "proxyless_cpu", pretrained=True)
n = ProxylessNASNets(model.first_conv, model.blocks, model.feature_mix_layer, model.classifier)
dump_models(n, "searched_nets/proxyless_mobile")

# # i, o = get_input_output(dp, model)


# print(i.shape)
# print(o.shape)

# # print(validate(dp, gpu_et))
# # print(validate(dp, net))
# # print(validate(dp, model))
# # print(validate(dp, net_proxyless_cpu))
