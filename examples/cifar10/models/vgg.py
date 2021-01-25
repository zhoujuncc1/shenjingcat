
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import catSNN
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_': [64, 64, (64,64), 128, 128, (128,128), 256, 256, 256, 256, (256,256), 512, 512, 512, 512, (512,512), 512, 512, 512, 512, (512,512)],
    'Mynetwork':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,512,512, 'M']
}


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class VGG(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max = 1.0, bias=False, quantize_bit=32):
        super(VGG, self).__init__()
        self.quantize_factor=quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name], quantize_bit=quantize_bit)
        self.classifier = nn.Linear(512, 10, bias=bias)
        self.features.apply(initialize_weights)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, quantize_bit=32):
        layers = []
        in_channels = 3
        for x in cfg:
            # [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            if x == 'M':
                #Train the model with dropout
                #layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),nn.Dropout2d(0.3)]
            else:
                # ReLU()-->Clamp()-->Clamp_q-->fuse bn and dropout
                #layers += [catSNN.QuantizedConv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias, quantize_bit=quantize_bit),
                           #catSNN.Clamp(max = self.clamp_max)]
                #layers += [catSNN.QuantizedConv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias, quantize_bit=quantize_bit),nn.BatchNorm2d(x),
                            #catSNN.Clamp(max = self.clamp_max),nn.Dropout2d(0.3)]
                layers += [catSNN.QuantizedConv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias, quantize_bit=quantize_bit),nn.BatchNorm2d(x),nn.ReLU(),nn.Dropout2d(0.2)]
                if self.quantize_factor!=-1:
                    layers += [catSNN.Quantize(self.quantize_factor)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max = 1.0, bias=False, quantize_bit=32):
        super(VGG_, self).__init__()
        self.quantize_factor=quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name], quantize_bit=quantize_bit)
        self.classifier = nn.Linear(512, 10, bias=bias)
        self.features.apply(initialize_weights)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, quantize_bit=32):
        layers = []
        in_channels = 3
        for x in cfg:
            # [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            if type(x) is tuple:
                layers += [nn.Conv2d(x[0], x[1], kernel_size=2, stride = 2, bias=self.bias, groups = x[1]),
                           catSNN.Clamp(max = self.clamp_max)]
                #layers += [nn.Conv2d(x[0], x[1], kernel_size=2, stride = 2, bias=self.bias, groups = x[1]),catSNN.Clamp(max = self.clamp_max),nn.Dropout2d(0.2)]
            else:
                layers += [catSNN.QuantizedConv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias, quantize_bit=quantize_bit),
                           catSNN.Clamp(max = self.clamp_max)]
                #layers += [catSNN.QuantizedConv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias, quantize_bit=quantize_bit),catSNN.Clamp(max = self.clamp_max),nn.Dropout2d(0.2)]
                if self.quantize_factor!=-1:
                    layers += [catSNN.Quantize(self.quantize_factor)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    


class CatVGG(nn.Module):
    def __init__(self, vgg_name, T, bias=False):
        super(CatVGG, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.bias=bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self.snn.dense((1,1,512),10)
  

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        out = self.snn.sum_spikes(out)/self.T
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                #layers += [self.snn.pool(2)]
                layers += [self.snn.pool(2),nn.Dropout2d(0)]
            else:
                #layers += [self.snn.conv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                #        self.snn.spikeLayer()]
                layers += [self.snn.conv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                        self.snn.spikeLayer(),nn.Dropout2d(0)]
                in_channels = x
        return nn.Sequential(*layers)
 
class CatVGG_(nn.Module):
    def __init__(self, vgg_name, T, bias=False):
        super(CatVGG_, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.bias=bias

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self.snn.dense((1,1,512),10)
  

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        out = self.snn.sum_spikes(out)/self.T
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if type(x) is tuple:
                layers += [self.snn.conv(x[0], x[1], kernelSize=2, stride = 2, bias=self.bias, groups = x[1]),self.snn.spikeLayer()]
                #layers += [self.snn.conv(x[0], x[1], kernelSize=2, stride = 2, bias=self.bias, groups = x[1]),self.snn.spikeLayer(),nn.Dropout2d(0)]
            else:
                layers += [self.snn.conv(in_channels, x, kernelSize=3, padding=1, bias=self.bias), self.snn.spikeLayer()]
                #layers += [self.snn.conv(in_channels, x, kernelSize=3, padding=1, bias=self.bias), self.snn.spikeLayer(),nn.Dropout2d(0)]
                in_channels = x
        return nn.Sequential(*layers)
