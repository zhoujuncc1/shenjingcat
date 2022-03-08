import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=100):
        ctx.constant = constant
        return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output), None 

Quantization_ = Quantization.apply

class Quantize(nn.Module):
    def __init__(self, factor):
        super(Clamp, self).__init__()
        self.factor = factor

    def forward(self, x):
        x = torch.div(torch.round(torch.mul(x, self.factor)), self.factor)
        return x

def quantize(x, factor):
    return torch.div(torch.round(torch.mul(x, factor)), factor)

def quantize_to_bit(x, nbit):
    if nbit == 32:
        return x
    return torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))

class Clamp(nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)

class Clamp_q(nn.Module):
    def __init__(self, min=0.0, max=1.0,q_level = 100):
        super(Clamp_q, self).__init__()
        self.min = min
        self.max = max
        self.q_level = q_level

    def forward(self, x):
        x = torch.clamp(x, min=self.min, max=self.max)
        x = Quantization_(x, self.q_level)
        return x

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', quantize_bit=32):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        self.quantize_bit=quantize_bit
    
    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            quantize_to_bit(self.weight, self.quantize_bit), self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, quantize_to_bit(self.weight, self.quantize_bit), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
