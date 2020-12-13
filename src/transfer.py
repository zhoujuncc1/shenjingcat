import torch
import torch.nn as nn
import torch.nn.functional as F
from .cat import _spikeLayer, _poolLayer, spikeLayer
from .cq import quantize_to_bit
def transfer_model(src, dst, quantize_bit=32):
    src_dict = src.state_dict()
    dst_dict = dst.state_dict()
    reshape_dict = {}
    for (k, v) in src_dict.items():
        if k in dst_dict.keys():
            reshape_dict[k] = nn.Parameter(quantize_to_bit(v.reshape(dst_dict[k].shape), quantize_bit))
    dst.load_state_dict(reshape_dict, strict=False)

def load_model(src, dst, quantize_bit=32):
    src_dict = src
    dst_dict = dst.state_dict()
    reshape_dict = {}
    for (k, v) in src_dict.items():
        if k in dst_dict.keys():
            reshape_dict[k] = nn.Parameter(quantize_to_bit(v.reshape(dst_dict[k].shape), quantize_bit))
    dst.load_state_dict(reshape_dict, strict=False)

def normalize_weight(model, threshold_scale = 1.0, quantize_bit=32):
    factor=0
    for m in model.modules():
        if isinstance(m, nn.Conv3d) and not isinstance(m, _poolLayer):
            print("Normalize: " + str(m))
            factor = torch.max(torch.abs(m.weight))
            
            if m.bias is not None and torch.max(torch.abs(m.bias)) > factor:
                factor = torch.max(torch.abs(m.bias))

            m.weight /= factor
            m.weight = nn.Parameter(quantize_to_bit(m.weight, quantize_bit))
            if m.bias is not None:
                m.bias/=factor
                m.bias = nn.Parameter(quantize_to_bit(m.bias, quantize_bit))
                
        elif isinstance(m, _spikeLayer):
            m.theta = m.theta/factor*threshold_scale

def max_weight(model):
    factor=0
    for m in model.modules():
        if isinstance(m, nn.Conv3d) and not isinstance(m, _poolLayer):
            factor = torch.max(torch.abs(m.weight))
            if m.bias is not None and torch.abs(m.bias) > factor:
                factor = torch.abs(m.bias)
            print("Layer: " + str(m) + " MAX:" + str(factor))
      
class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)

def fuse_bn_sequential(block):
    if not isinstance(block, nn.Sequential):
        return block
    stack = []
    for m in block.children():
        if isinstance(m, nn.BatchNorm2d):
            if isinstance(stack[-1], nn.Conv2d):
                bn_st_dict = m.state_dict()
                conv_st_dict = stack[-1].state_dict()

                eps = m.eps
                mu = bn_st_dict['running_mean']
                var = bn_st_dict['running_var']
                if 'weight' in bn_st_dict:
                    gamma = bn_st_dict['weight']
                else:
                    gamma = torch.ones(mu.size(0)).float().to(mu.device)

                if 'bias' in bn_st_dict:
                    beta = bn_st_dict['bias']
                else:
                    beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                # Conv params
                W = conv_st_dict['weight']
                if 'bias' in conv_st_dict:
                    bias = conv_st_dict['bias']
                else:
                    bias = torch.zeros(W.size(0)).float().to(gamma.device)

                denom = torch.sqrt(var + eps)
                b = beta - gamma.mul(mu).div(denom)
                A = gamma.div(denom)
                bias *= A
                A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                W.mul_(A)
                bias.add_(b)

                stack[-1].weight.data.copy_(W)
                if stack[-1].bias is None:
                    stack[-1].bias = torch.nn.Parameter(bias)
                else:
                    stack[-1].bias.data.copy_(bias)

        else:
            stack.append(m)

    if len(stack) > 1:
        return nn.Sequential(*stack)
    else:
        return stack[0]


def fuse_bn_recursively(model):
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_sequential(model._modules[module_name])
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])

    return model


class CatNet(nn.Module):
    def __init__(self, src_model, T, shape_to_dense=None):
        super(CatNet, self).__init__()
        self.snn = spikeLayer(T)
        self.T = T
        self.shape_to_dense = shape_to_dense

        self.net = self._make_layers(src_model)

    def forward(self, x):
        for m in self.net:
            x = m(x)
        x = x.sum(dim=(2,3,4))
        return x

    def load_weight(self, model):
        i = 0
        for m in model.modules():
            pooling = False
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.net[i].weight.data.copy_(m.weight.reshape(self.net[i].weight.shape))
                if m.bias is not None:
                    self.net[i].bias.data.copy_(m.bias.reshape(self.net[i].bias.shape))
                i+=2
            elif isinstance(m, nn.AvgPool2d):
                i+=2

    def _make_layers(self, model):
        layers = []
        first_dense = True
        for m in model.modules():
            pooling = False
            if isinstance(m, nn.Conv2d):
                layer = self.snn.conv(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dialation, grouping=m.grouping, bias = (m.bias is not None))
            elif isinstance(m, nn.Linear):
                if first_dense and self.shape_to_dense is not None:
                    layer = self.snn.dense(self.shape_to_dense, m.out_features, bias=(m.bias is not None))
                    first_dense=False
                else:
                    layer = self.snn.dense(m.in_features, m.out_features, bias=(m.bias is not None))
            elif isinstance(m, nn.AvgPool2d):
                layer = self.snn.pool(m.kernel_size, padding=m.padding, bias=(m.bias is not None))
                pooling=True
            else:
                continue
            if not pooling:
                layer.weight.data.copy_(m.weight.reshape(layer.weight.shape))
                if m.bias is not None:
                    layer.bias.data.copy_(m.bias.reshape(layer.bias.shape))
            layers += [layer, self.snn.spikeLayer()]
        return nn.Sequential(*layers)
