#!/usr/bin/env python

import net224x224 as models


net = models.proxyless_cifar(pretrained=False)
print(net)
