#!/usr/bin/python
# encoding: utf-8

import torch.nn as nn
import torch.nn.parallel


def data_parallel(model, input, ngpu):
    if ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
        output = nn.parallel.data_parallel(model, input, range(ngpu))
    else:
        output = model(input)
    return output
