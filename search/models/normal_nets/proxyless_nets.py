# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from modules.layers import *
from modules.mix_op import *
import json
import numpy as np


def proxyless_base(net_config=None, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0):
    assert net_config is not None, 'Please input a network config'
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))

    net_config_json['classifier']['out_features'] = n_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config, dropout=0):
        c = config['mobile_inverted_conv']
        if dropout > 0:
            c['dropout_rate'] = dropout
        mobile_inverted_conv = set_layer_from_config(c)
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)

class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config, dropout=0):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop

        delta_flop, x = self.feature_mix_layer.get_flops(x)
        flop += delta_flop

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x


    def net_latency(self, device_type, config, latency_estimator):
        if device_type == "soc":
            predicted_latency = {k : 0 for k in config.split_points}
            try:
                logger.info("enter soc net latency")
                block_idx = 0
                # first conv
                predicted_latency[config.get_stage(block_idx)] += latency_estimator.predict(
                    config.get_proc_of_block(block_idx),
                    'Conv', [224, 224, 3], [112, 112, self.first_conv.out_channels]
                )
                block_idx += 1
                # blocks
                fsize = 112
                for block in self.blocks:
                    mb_conv = block.mobile_inverted_conv
                    shortcut = block.shortcut
                    if isinstance(mb_conv, MixedEdge):
                        mb_conv = mb_conv.active_op
                    if isinstance(shortcut, MixedEdge):
                        shortcut = shortcut.active_op

                    if mb_conv.is_zero_layer():
                        continue
                    if shortcut is None or shortcut.is_zero_layer():
                        idskip = 0
                    else:
                        idskip = 1
                    out_fz = fsize // mb_conv.stride
                    block_latency = latency_estimator.predict(
                        config.get_proc_of_block(block_idx),
                        'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
                        expand=mb_conv.expand_ratio, kernel=mb_conv.kernel_size, stride=mb_conv.stride, idskip=idskip
                    )
                    predicted_latency[config.get_stage(block_idx)] += block_latency
                    block_idx += 1
                    fsize = out_fz

                # feature mix layer
                predicted_latency[config.get_stage(block_idx)] += latency_estimator.predict(
                    config.get_proc_of_block(block_idx),
                    'Conv_1', [7, 7, self.feature_mix_layer.in_channels], [7, 7, self.feature_mix_layer.out_channels]
                )
                block_idx += 1
                # classifier
                predicted_latency[config.get_stage(block_idx)] += latency_estimator.predict(
                    config.get_proc_of_block(block_idx),
                    'Logits', [7, 7, self.classifier.in_features], [self.classifier.out_features]  # 1000
                )
                block_idx += 1

                print(f"{predicted_latency=}")
                return max(predicted_latency.values())
                # assert block_idx == len(net.modules)
            except Exception:
                predicted_latency = 200
                print('fail to predict the mobile latency')
            return predicted_latency, None
        pass


    def to_json(self, dump_path, config = None):
        self.eval()
        # first conv

        def dump_weights(index, block):
            weights = {}
            for name, param in block.state_dict().items():
                path = dump_path + "/" + str(index) + "-" + name + '.npy'
                if len(param.size()) != 0:
                    t = param.data.numpy()
                    # reshape to remove 1 if the layer is depthwise conv
                    if "depth_conv.conv.weight" in name:
                        s = t.shape
                        assert len(s) == 4 and s[1] == 1
                        new_shape = (s[0], s[2], s[3])
                        t = t.reshape(new_shape)

                    if "linear.weight" in name:
                        s = t.shape
                        assert len(s) == 2
                        new_shape = (s[0], s[1], 1, 1)
                        t = t.reshape(new_shape)

                    np.save(path, t)
                    weights[name] = path

            return weights

        def get_first_conv(index, block, hw_size):
            content = {
                "index" : index,
                "type": "Conv",
                "input": [hw_size, hw_size, block.in_channels],
                "output": [hw_size/2, hw_size/2, block.out_channels]
            }
            weights = dump_weights(index, block)
            content["params"] = weights
            return content

        def get_pad(in_size, out_size, stride, kernel):
            padded_in_size = (out_size - 1) * stride + kernel
            assert padded_in_size >= in_size
            p = padded_in_size - in_size
            assert int(((in_size + p) - kernel)/stride + 1) == out_size
            l_pad = int((padded_in_size - in_size)/2)
            r_pad = p - l_pad
            return int(l_pad), int(r_pad)


        def get_expanded_block(index, block, hw_size):
            mb_conv = block.mobile_inverted_conv
            shortcut = block.shortcut

            out_fz = int(hw_size // mb_conv.stride)
            l_pad, r_pad = get_pad(hw_size, out_fz, mb_conv.stride, mb_conv.kernel_size)

            if shortcut is None or shortcut.is_zero_layer():
                idskip = 0
            else:
                idskip = 1

            content = {
                "index" : index,
                "type": "expanded_conv",
                "input": [hw_size, hw_size, mb_conv.in_channels],
                "output": [out_fz, out_fz, mb_conv.out_channels],
                "expand": mb_conv.expand_ratio,
                "kernel": mb_conv.kernel_size,
                "stride": mb_conv.stride,
                "idskip" : idskip,
                "pad_l": l_pad,
                "pad_r": r_pad
            }

            w = dump_weights(index, mb_conv)
            content["params"] = w
            return content

        def get_last_conv(index, block, hw_size):
            content = {
                "index" : index,
                "type": "Conv_1",
                "input": [hw_size, hw_size, block.in_channels],
                "output": [hw_size, hw_size, block.out_channels]
            }
            w = dump_weights(index, block)
            content["params"] = w
            return content

        def get_linear(index, block, hw_size):
            content = {
                "index" : index,
                "type": "Logits",
                "input": [hw_size, hw_size, block.in_features],
                "output": [block.out_features]
            }
            w = dump_weights(index, block)
            content["params"] = w
            return content

        def get_zero(index):
            return {
                    "index": index,
                    "type": "zerolayer"
                }

        def add_proc_assign(json_c, config : PipelineConfig):
            for k, val in json_c.items():
                proc = config.get_proc_of_block(int(k))
                json_c[k]["proc"] = proc

        json_content = {}

        index = 0
        hw_size = 224
        json_content[index] = get_first_conv(index, self.first_conv, hw_size)
        
        hw_size = int(hw_size // 2)
        index += 1

        for block in self.blocks:
            mb_conv = block.mobile_inverted_conv
            shortcut = block.shortcut

            if mb_conv.is_zero_layer():
                json_content[index] = get_zero(index)
                index += 1
                continue

            json_content[index] = get_expanded_block(index, block, hw_size)
            hw_size = int(hw_size // mb_conv.stride)
            index += 1

        json_content[index] = get_last_conv(index, self.feature_mix_layer, hw_size)
        index += 1
        json_content[index] = get_linear(index, self.classifier, hw_size)

        if config is not None:
            add_proc_assign(json_content, config)

        json_file = dump_path + "/net.json"
        with open(json_file, "w+") as fp:
            string = json.dumps(json_content, indent=4)
            fp.write(string)
        pass

