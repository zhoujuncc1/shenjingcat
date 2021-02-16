import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def transfer(model, file):
    """For transfer of ANN to TTFS SNN"""
    dst_dict = model.state_dict()
    src_dict = torch.load(file)
    reshape_dict = {}
    for (k, v) in src_dict.items():
        if k in dst_dict.keys():
            if v.shape != dst_dict[k].shape:
                reshape_dict[k] = v.transpose(0,1)
            else:
                reshape_dict[k] = v
    model.load_state_dict(reshape_dict, strict=False)

def norm_error(loss):
    loss = F.normalize(loss, p=2, dim=-1)
    return loss

def norm_error_old(loss):
    for i in range(len(loss)):
        norm = torch.norm(loss[i])
        if norm != 0:
            loss[i] = loss[i] / norm
    return loss

class relative_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, labels, tmax, gamma):
        minFiring = out.min(axis=-1, keepdim=True).values
        target = out.clone()
        toChange = (out - minFiring) < gamma
        #TODO: fix for batchsize>1
        #target[toChange] = torch.minimum(minFiring + gamma, tmax)
        target = torch.where(toChange, torch.minimum(minFiring + gamma, tmax), target)
        target.index_copy_(1,labels, minFiring)
        target = torch.where(minFiring<tmax, target, tmax - gamma)
        loss = (target - out)/tmax
        loss = norm_error(loss)
        ctx.save_for_backward(loss)
        return loss
    
    @staticmethod
    def backward(ctx, gradOutput):
        (loss,) = ctx.saved_tensors
        #import pdb
        #pdb.set_trace()
        return loss, None, None, None

class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias=False, T: int = 64, theta:float = 1.0, lamda=0):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float32))
        self.T=T
        self.t_series = torch.nn.Parameter(torch.from_numpy(np.linspace(0, self.T-1, self.T)), requires_grad=False)
        self.theta=theta
        self.lamda=lamda
    def forward(self, input):
        return _dense.apply(input, self.weight, self.t_series, self.T, self.theta, self.lamda)

class Dense_Voltage(Dense):
    def __init__(self, in_features, out_features, bias=False, T: int = 64, theta:float = 1.0, lamda=0):
        super(Dense_Voltage, self).__init__(in_features, out_features, bias, T, theta, lamda)
    def forward(self, input):
        return _dense_v.apply(input, self.weight, self.t_series, self.T, self.theta, self.lamda)


class DenseDelta(Dense):
    def __init__(self, in_features, out_features, bias=False, T: int = 64, theta:float = 1.0, lamda=0):
        super(DenseDelta, self).__init__(in_features, out_features, bias, T, theta, lamda)
    def forward(self, input):
        return _dense_delta.apply(input, self.weight,self.T, self.theta, self.lamda)

class _dense(torch.autograd.Function):
    '''
    Input of shape (N, X, T) and output (N, Y, T)
    Weight of shape (Y, X)
    theta: threshold
    alpha, beta: parameters for back propagation approximation
    '''
    @staticmethod
    def forward(ctx, input, weight, t_series, T: int = 64, theta:float = 1.0, lamda=0):
        device = input.device
        dtype  = input.dtype
        actual_input = (t_series[:, None, None] >= input[None, :, :]).type(torch.float32)
        Vs = torch.matmul(actual_input, weight[None,:,:]) #[T, N, feature]
        V = torch.cat([Vs, torch.ones_like(Vs[0:1])*theta+1])
        output = torch.argmax((V>theta).type(torch.float), axis=0)+1
        output = torch.minimum(output, torch.tensor(T, device=output.device)).type(torch.float32)

        ctx.save_for_backward(input, output, weight)

        return output

    @staticmethod
    def backward(ctx, gradOutput):
        #import pdb
        #pdb.set_trace()
        (input, output, weight) = ctx.saved_tensors
        actual_input =  (output[:, None, :] > input[:,:,None]).type(torch.float32)
  
        delta_w = (gradOutput[:, None, :] * actual_input).mean(axis=0)
        delta_x = (gradOutput[:, None, :] * actual_input* weight[None,:,:]).sum(axis=-1)
        
        delta_x = norm_error(delta_x)
        return delta_x, delta_w, None, None, None, None, 

class _dense_v(torch.autograd.Function):
    '''
    Input of shape (N, X, T) and output (N, Y, T)
    Weight of shape (Y, X)
    theta: threshold
    alpha, beta: parameters for back propagation approximation
    '''
    @staticmethod
    def forward(ctx, input, weight, t_series, T: int = 64, theta:float = 1.0, lamda=0):
        device = input.device
        dtype  = input.dtype
        actual_input = (t_series[:, None, None] >= input[None, :, :]).type(torch.float32)
        Vs = torch.matmul(actual_input, weight[None,:,:]) #[T, N, feature]
        V = torch.cat([Vs, torch.ones_like(Vs[0:1])*theta+1])
        output = torch.argmax((V>theta).type(torch.float), axis=0)+1
        output = torch.minimum(output, torch.tensor(T, device=output.device)).type(torch.float32)

        ctx.save_for_backward(input, output, weight)

        return output, V
        
    @staticmethod
    def backward(ctx, gradOutput1, gradOutput2):
        #import pdb
        #pdb.set_trace()
        (input, output, weight) = ctx.saved_tensors
        actual_input =  (output[:, None, :] > input[:,:,None]).type(torch.float32)
  
        delta_w = (gradOutput1[:, None, :] * actual_input).mean(axis=0)
        delta_x = (gradOutput1[:, None, :] * actual_input* weight[None,:,:]).sum(axis=-1)
        
        delta_x = norm_error(delta_x)
        return delta_x, delta_w, None, None, None, None, 



class _dense_delta(torch.autograd.Function):
    '''
    Input of shape (N, X, T) and output (N, Y, T)
    Weight of shape (Y, X)
    theta: threshold
    alpha, beta: parameters for back propagation approximation
    '''
    @staticmethod
    def forward(ctx, input, weight, t_series, T: int = 64, theta:float = 1.0, lamda=0):
        device = input.device
        dtype  = input.dtype
        actual_input = torch.relu(t_series[:, None, None] - input[None, :, :]).type(torch.float32)
        Vs = torch.matmul(actual_input, weight[None,:,:]) #[T, N, feature]
        V = torch.cat([Vs, torch.ones_like(Vs[0:1])*theta+1])
        output = torch.argmax((V>theta).type(torch.float), axis=-1)+1
        output = torch.minimum(output, torch.tensor(T, device=output.device)).type(torch.float32)
        ctx.save_for_backward(input, output, weight)

        return output
        
    @staticmethod
    def backward(ctx, gradOutput):
        #import pdb
        #pdb.set_trace()
        (input, output, weight) = ctx.saved_tensors
        actual_input =  torch.relu(output[:, None, :] -input[:,:,None])
  
        delta_w = (gradOutput[:, None, :] * actual_input).mean(axis=0)
        delta_x = (gradOutput[:, None, :] * actual_input* weight[None,:,:]).sum(axis=-1)
        
        delta_x = norm_error(delta_x)
        return delta_x, delta_w, None, None, None, 

