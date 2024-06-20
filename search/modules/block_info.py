from modules.layers import *
from modules.mix_op import *
from models.normal_nets.proxyless_nets import MobileInvertedResidualBlock
from pprint import pprint as pp

__layer_types = [
    "Conv",
    "Conv_1",
    "Logits",
    "expanded_conv"
]

class BlockInfo(object):
    def __init__(self, ltype: str, inp, output, expand=None, kernel=None, stride=None, idskip=None):
        self.ltype = ltype
        self.input = inp
        self.output = output
        self.expand = expand
        self.kernel = kernel
        self.stride = stride
        self.idskip = idskip
        pass

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    @staticmethod
    def get_pad(in_size, out_size, stride, kernel):
        padded_in_size = (out_size - 1) * stride + kernel
        assert padded_in_size >= in_size
        p = padded_in_size - in_size
        assert int(((in_size + p) - kernel)/stride + 1) == out_size
        l_pad = int((padded_in_size - in_size)/2)
        r_pad = p - l_pad
        return l_pad, r_pad

    def get_key(self):
        infos = [self.ltype, 'input:%s' % self.repr_shape(self.input), 'output:%s' % self.repr_shape(self.output), ]

        if self.ltype in ('expanded_conv',):
            assert None not in (self.expand, self.kernel, self.stride, self.idskip)
            infos += ['expand:%d' % self.expand, 'kernel:%d' % self.kernel, 'stride:%d' % self.stride, 'idskip:%d' % self.idskip]
        key = '-'.join(infos)
        return key

    def get_dict_for_armcl(self):
        dict_str = {
            "type" : self.ltype,
            "input" : self.input,
            "output" : self.output,
        }
        if self.ltype == "expanded_conv":
            pad_l, pad_r = self.get_pad(self.input[0], self.output[0], self.stride, self.kernel)
            dict_str.update({
                "expand": self.expand,
                "kernel": self.kernel,
                "stride": self.stride,
                "idskip": self.idskip,
                "pad_l": pad_l,
                "pad_r": pad_r
            })
        elif self.ltype == "Conv":
            pad_l, pad_r = self.get_pad(self.input[0], self.output[0], self.stride, self.kernel)
            dict_str.update({
                "kernel": self.kernel,
                "stride": self.stride,
                "pad_l": pad_l,
                "pad_r": pad_r
            })

        return {self.get_key(): dict_str}

def from_module(mod, in_h, in_w, out_h, out_w):
    print("==== process ")
    print(str(mod))
    if isinstance(mod, ConvLayer):
        if mod.kernel_size == 1:
            type = "Conv_1"
        else:
            type = "Conv"
        b = BlockInfo(type, [in_h, in_w, mod.in_channels],
                  [out_h, out_h, mod.out_channels], kernel=mod.kernel_size, stride=mod.stride)
        return b.get_dict_for_armcl()
    elif isinstance(mod, MBInvertedConvLayer):
        dicts = {}
        b_skip = BlockInfo("expanded_conv", [in_h, in_w, mod.in_channels], 
          [out_h, out_h, mod.out_channels],
          mod.expand_ratio,
          mod.kernel_size,
          mod.stride, 0)
        dicts.update(b_skip.get_dict_for_armcl())

        if mod.in_channels == mod.out_channels and in_h == out_h and in_w == out_w:
            b_skip.idskip = 1
            dicts.update(b_skip.get_dict_for_armcl())
        return dicts
    elif isinstance(mod, MixedEdge):
        print("get info from mixed edge")
        dicts = {}
        for cand_op in mod.candidate_ops:
            if isinstance(cand_op, IdentityLayer):
                continue
            elif isinstance(cand_op, MBInvertedConvLayer):
                b_skip = BlockInfo("expanded_conv", [in_h, in_w, cand_op.in_channels], 
                  [out_h, out_h, cand_op.out_channels],
                  cand_op.expand_ratio,
                  cand_op.kernel_size,
                  cand_op.stride, 0)
                dicts.update(b_skip.get_dict_for_armcl())
                if cand_op.in_channels == cand_op.out_channels and in_h == out_h and in_w == out_w:
                    b_skip.idskip = 1
                    dicts.update(b_skip.get_dict_for_armcl())
        return dicts
    elif isinstance(mod, LinearLayer):
        b = BlockInfo("Logits", [in_h, in_w, mod.in_features], [mod.out_features])
        return b.get_dict_for_armcl()
    else:
        print("None for ")
        print(mod)
        return None

if __name__ == "__main__":
    b = BlockInfo("Conv", [32, 32, 3], [16, 16, 16])
    print(b.get_dict_for_armcl())
    b = BlockInfo("expanded_conv", [32, 32, 3], [16, 16, 16], 3, 3, 2, 1)
    print(b.get_dict_for_armcl())

    mod = MixedEdge(build_candidate_ops(["3x3_MBConv2"], 32, 64, 2, "weight_bn_act"))
    b = BlockInfo.from_module(mod, 32, 32, 16, 16)
    pp(b)
    