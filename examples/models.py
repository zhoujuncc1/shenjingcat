import torch
import torch.nn as nn
import catSNN

example_mlp_config = [(784, 512), (512, 256), (256, 10)]

class MLP(nn.Module):
    def __init__(self, mlp_config, activation_quantize_factor=-1, clamp_max = 1.0, bias=False, weight_quantize_bit=32, dropout=0, activation='relu'):
        super(MLP, self).__init__()
        assert activation in ['relu', 'clamp']
        self.activation=activation
        self.dropout=dropout
        self.activation_quantize_factor=activation_quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.weight_quantize_bit=weight_quantize_bit
        self.cfg = mlp_config
        self.mlp = self._make_layers()


    def forward(self, x):
        out = self.mlp(x)
        return out

    def _make_layers(self):
        layers = []
        for x in self.cfg:
            toadd = [catSNN.QuantizedLinear(x[0], x[1], bias=self.bias, quantize_bit=self.weight_quantize_bit)]
            if self.activation=='relu':
                toadd.append(nn.ReLU())
            elif self.activation == 'clamp':
                toadd.append(catSNN.Clamp(max = self.clamp_max))
            toadd.append(nn.Dropout2d(self.dropout))
            if self.activation_quantize_factor!=-1:
                toadd.append(catSNN.Quantize(self.activation_quantize_factor))
            else:
                toadd.append(nn.Identity())
            layers += toadd
        return nn.Sequential(*layers)

class CatMLP(nn.Module):
    def __init__(self, mlp_config, activation_quantize_factor=-1, clamp_max = 1.0, bias=False, weight_quantize_bit=32, dropout=0, activation='relu'):
        super(CatMLP, self).__init__()
        assert activation in ['relu', 'clamp']
        self.activation=activation
        self.dropout=dropout
        self.activation_quantize_factor=activation_quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.weight_quantize_bit=weight_quantize_bit
        self.cfg = mlp_config
        self.mlp = self._make_layers()
    def __init__(self, ann_model):
        super(CatMLP, self).__init__()
        self.ann_model=ann_model
        self.activation=ann_model.activation
        self.dropout=ann_model.dropout
        self.activation_quantize_factor=ann_model.activation_quantize_factor
        self.clamp_max = ann_model.clamp_max
        self.bias = ann_model.bias
        self.weight_quantize_bit=ann_model.weight_quantize_bit
        self.cfg = ann_model.cfg
        self.mlp = self._make_layers()

    def forward(self, x):
        out = self.mlp(x)
        return out

    def _make_layers(self):
        layers = []
        for x in self.cfg:
            print(x)
            layers += [catSNN.dense(x[0], x[1], bias=self.bias), catSNN.spikeLayer(), nn.Identity(), nn.Identity()]
        return nn.Sequential(*layers)