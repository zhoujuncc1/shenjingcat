
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import catSNN

cfg = {
    'o' : [128,128,'M',256,256,'M',512,512,'M',(1024,0),'M'],
}


def initialize_weights(self):
    for m in self.modules():
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



class VGG_(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=False):
        super(VGG_, self).__init__()
        #self.quantize_factor = quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(1024, 100, bias=True)
        self.features.apply(initialize_weights)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),nn.Dropout2d(0.15)]
            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=True),nn.BatchNorm2d(out_channels), catSNN.Clamp(max = self.clamp_max),nn.Dropout2d(0.15)]
                in_channels = out_channels


        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class CatVGG(nn.Module):
    def __init__(self, vgg_name, T, is_noise=False, bias=True):
        super(CatVGG, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.is_noise = is_noise
        self.bias = bias

        self.features = self._make_layers(cfg[vgg_name], is_noise)
        self.classifier = self.snn.dense((1, 1, 1024), 100,bias = True)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        out = self.snn.sum_spikes(out) / self.T
        return out

    def _make_layers(self, cfg, is_noise=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [self.snn.pool(2),nn.Dropout2d(0)]
            else:
                if is_noise:
                    layers += [self.snn.mcConv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                               self.snn.spikeLayer(1),nn.Dropout2d(0)]
                    in_channels = x
                else:
                    padding = x[1] if isinstance(x, tuple) else 1
                    out_channels = x[0] if isinstance(x, tuple) else x
                    layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                               self.snn.spikeLayer(1),nn.Dropout2d(0)]
                    in_channels = out_channels
        return nn.Sequential(*layers)
