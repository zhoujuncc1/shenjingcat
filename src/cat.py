import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import catCuda
import catCpp

class spikeLayer(torch.nn.Module):

    def __init__(self, T):
        super(spikeLayer, self).__init__()
        self.T = T

def dense(inFeatures, outFeatures, weightScale=1, bias=False):   # default weight scaling of 10
    '''
    Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
    It behaves similar to ``torch.nn.Linear`` applied for each time instance.

    Arguments:
        * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
            dimension of input features (Width, Height, Channel) that represents the number of input neurons.
        * ``outFeatures`` (``int``): number of output neurons.
        * ``weightScale``: sale factor of default initialized weights. Default: 10

    Usage:
    
    >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
    >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
    >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
    '''
    return _denseLayer(inFeatures, outFeatures, weightScale, bias)

def conv(inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1, bias=False):    # default weight scaling of 100
    '''
    Returns a function that can be called to apply conv layer mapping to input tensor per time instance.
    It behaves same as ``torch.nn.conv2d`` applied for each time instance.

    Arguments:
        * ``inChannels`` (``int``): number of channels in input
        * ``outChannels`` (``int``): number of channls produced by convoluion
        * ``kernelSize`` (``int`` or tuple of two ints): size of the convolving kernel
        * ``stride`` (``int`` or tuple of two ints): stride of the convolution. Default: 1
        * ``padding`` (``int`` or tuple of two ints):   zero-padding added to both sides of the input. Default: 0
        * ``dilation`` (``int`` or tuple of two ints): spacing between kernel elements. Default: 1
        * ``groups`` (``int`` or tuple of two ints): number of blocked connections from input channels to output channels. Default: 1
        * ``weightScale``: sale factor of default initialized weights. Default: 100

    The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

    - a single ``int`` -- in which case the same value is used for the height and width dimension
    - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
        and the second `int` for the width dimension

    Usage:

    >>> conv = snnLayer.conv(2, 32, 5) # 32C5 flter
    >>> output = conv(input)           # must have 2 channels
    '''
    return _convLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale, bias) 

def pool(kernelSize, stride=None, padding=0, dilation=1, weight=None, theta=1.0):
    '''
    Returns a function that can be called to apply pool layer mapping to input tensor per time instance.
    It behaves same as ``torch.nn.``:sum pooling applied for each time instance.

    Arguments:
        * ``kernelSize`` (``int`` or tuple of two ints): the size of the window to pool over
        * ``stride`` (``int`` or tuple of two ints): stride of the window. Default: `kernelSize`
        * ``padding`` (``int`` or tuple of two ints): implicit zero padding to be added on both sides. Default: 0
        * ``dilation`` (``int`` or tuple of two ints): a parameter that controls the stride of elements in the window. Default: 1
        
    The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

    - a single ``int`` -- in which case the same value is used for the height and width dimension
    - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
        and the second `int` for the width dimension

    Usage:

    >>> pool = snnLayer.pool(4) # 4x4 pooling
    >>> output = pool(input)
    '''
    return _poolLayer(theta, kernelSize, stride, padding, dilation, weight)

def dropout(p=0.5, inplace=False):
    '''
    Returns a function that can be called to apply dropout layer to the input tensor.
    It behaves similar to ``torch.nn.Dropout``.
    However, dropout over time dimension is preserved, i.e.
    if a neuron is dropped, it remains dropped for entire time duration.

    Arguments:
        * ``p``: dropout probability.
        * ``inplace`` (``bool``): inplace opeartion flag.

    Usage:

    >>> drop = snnLayer.dropout(0.2)
    >>> output = drop(input)
    '''
    return _dropoutLayer(p, inplace)


def spike(membranePotential, theta=1.0):
    '''
    Applies spike function and refractory response.
    The output tensor dimension is same as input.
    ``membranePotential`` will reflect spike and refractory behaviour as well.

    Arguments:
        * ``membranePotential``: subthreshold membrane potential.

    Usage:

    >>> outSpike = snnLayer.spike(membranePotential)
    '''
    return _spikeFunction.apply(membranePotential, theta)

def spikeLayer(theta=1.0):
    return _spikeLayer(theta)


def sum_spikes(x):
    result = torch.sum(x, dim=[i for i in range(2, len(x.shape))])
    return result
def sum_spikes_layer():
    return _sumSpikeLayer()

class _denseLayer(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, weightScale=1, bias=False):
        '''
        '''
        # extract information for kernel and inChannels
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures 
        elif len(inFeatures) == 2:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = 1
        elif len(inFeatures) == 3:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = inFeatures[2]
        else:
            raise Exception('inFeatures should not be more than 3 dimension. It was: {}'.format(inFeatures.shape))
        # print('Kernel Dimension:', kernel)
        # print('Input Channels  :', inChannels)
        
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)
        
        super(_denseLayer, self).__init__(inChannels, outChannels, kernel, bias=bias)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)

    
    def forward(self, input):
        '''
        '''
        return F.conv3d(input, 
                        self.weight, self.bias, 
                        self.stride, self.padding, self.dilation, self.groups)


class _convLayer(nn.Conv3d):
    '''
    '''
    def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1, bias=False):
        inChannels = inFeatures
        outChannels = outFeatures
        
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        # groups
        # no need to check for groups. It can only be int

        # print('inChannels :', inChannels)
        # print('outChannels:', outChannels)
        # print('kernel     :', kernel, kernelSize)
        # print('stride     :', stride)
        # print('padding    :', padding)
        # print('dilation   :', dilation)
        # print('groups     :', groups)

        super(_convLayer, self).__init__(inChannels, outChannels, kernel, stride, padding, dilation, groups, bias=bias)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In conv, using weightScale of', weightScale)

    def foward(self, input):
        '''
        '''
        return F.conv3d(input, 
                        self.weight, self.bias, 
                        self.stride, self.padding, self.dilation, self.groups)

class _poolLayer(nn.Conv3d):
    '''
    '''
    def __init__(self, theta, kernelSize, stride=None, padding=0, dilation=1, weight=None):
        total_size = 1
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
            total_size=kernelSize*kernelSize
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
            total_size = kernelSize[0] * kernelSize[1]
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))
        
        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        # print('theta      :', theta)
        # print('kernel     :', kernel, kernelSize)
        # print('stride     :', stride)
        # print('padding    :', padding)
        # print('dilation   :', dilation)
        
        super(_poolLayer, self).__init__(1, 1, kernel, stride, padding, dilation, bias=False)   

        # set the weights to 1.1*theta and requires_grad = False
        if weight == None:
            weight = 1.0/np.prod(self.weight.shape)
        self.weight = torch.nn.Parameter(torch.FloatTensor(weight * theta * np.ones((self.weight.shape))).to(self.weight.device), requires_grad = False)
        # print('In pool layer, weight =', self.weight.cpu().data.numpy().flatten(), theta)


    def forward(self, input):
        '''
        '''
        device = input.device
        dtype  = input.dtype
        
        # add necessary padding for odd spatial dimension
        # if input.shape[2]%2 != 0:
            # input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], 1, input.shape[3], input.shape[4]), dtype=dtype).to(device)), 2)
        # if input.shape[3]%2 != 0:
            # input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2], 1, input.shape[4]), dtype=dtype).to(device)), 3)
        if input.shape[2]%self.weight.shape[2] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2]%self.weight.shape[2], input.shape[3], input.shape[4]), dtype=dtype).to(device)), 2)
        if input.shape[3]%self.weight.shape[3] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3]%self.weight.shape[3], input.shape[4]), dtype=dtype).to(device)), 3)

        dataShape = input.shape

        result = F.conv3d(input.reshape((dataShape[0], 1, dataShape[1] * dataShape[2], dataShape[3], dataShape[4])), 
                          self.weight, self.bias, 
                          self.stride, self.padding, self.dilation)
        # print(result.shape)
        return _spikeFunction.apply(result.reshape((result.shape[0], dataShape[1], -1, result.shape[3], result.shape[4])), 1.0)

class _dropoutLayer(nn.Dropout3d):
    '''
    '''
    # def __init__(self, p=0.5, inplace=False):
    #   super(_dropoutLayer, self)(p, inplace)

    '''
    '''
    def forward(self, input):
        inputShape = input.shape
        return F.dropout3d(input.reshape((inputShape[0], -1, 1, 1, inputShape[-1])),
                           self.p, self.training, self.inplace).reshape(inputShape)


class _spikeLayer(torch.nn.Module):
    def __init__(self, theta=1.0):
        super(_spikeLayer, self).__init__()
        self.theta=theta
    def forward(self, input):
        return _spikeFunction.apply(input, self.theta)

class _sumSpikeLayer(torch.nn.Module):
    def __init__(self):
        super(_sumSpikeLayer, self).__init__()
    def forward(self, input):
        result = torch.sum(input, dim=[i for i in range(2, len(input.shape))])
        return result

    
class _spikeFunction(torch.autograd.Function):
    '''
    '''

    @staticmethod
    def forward(ctx, membranePotential, theta):
        '''
        '''
        device = membranePotential.device
        dtype  = membranePotential.dtype
        oldDevice = torch.cuda.current_device()

        # if device != oldDevice: torch.cuda.set_device(device)
        # torch.cuda.device(3)

        # spikeTensor = torch.empty_like(membranePotential)

        # print('membranePotential  :', membranePotential .device)
        # print('spikeTensor        :', spikeTensor       .device)


        spikes = catCuda.getSpikes(membranePotential.contiguous(), theta)
        return spikes
        



class SpikeDataset(torch.utils.data.Dataset):
    # type - random, spike, float, ttfs
    def __init__(self, dataset, T, type='spike', theta = 1.0):
        self.dataset = dataset
        self.T = T
        self.type=type
        self.theta=theta
    def __getitem__(self, index):
        data, label = self.dataset.__getitem__(index)
        if self.type == 'random':
            spikes_data = [torch.rand(data.shape) < data for _ in range(self.T)]
        else: # float and spike
            spikes_data = [data for _ in range(self.T)]
        out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor)  #float
        if self.type == 'spike':
            out = catCpp.getSpikes(out, self.theta)
        return out, label

    def __len__(self):
        return len(self.dataset)
