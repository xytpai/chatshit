import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return torch.nn.functional.gelu(x, approximate=True)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "tanh": torch.tanh,  "relu": F.relu, "swish": swish}
class LinearActivation(nn.Module):
    __constants__ = ['bias']
    def __init__(self, in_features, out_features, act='gelu', bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None
        assert act in ACT2FN, "Activation function is not found in activation dictionary."
        self.act_fn = ACT2FN[act]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
