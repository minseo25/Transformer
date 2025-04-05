import torch
from transformer.abstract import Optimizer
from utils.util import unfold_dict

class SGD(Optimizer):
    def __init__(self, model, lr=0.01, wd=0.01, **kwargs):
        super(SGD, self).__init__(model, wd, **kwargs)
        self.lr = lr

    def update_dict(self):
        grad_dict = unfold_dict(self.model.grad_dict())
        grad_dict = self.wd_grad(grad_dict)
        _update_dict = {}
        for k in self.param_dict.keys():
            _update_dict[k] = self.lr * grad_dict[k]
        
        return _update_dict
    
class Momentum(Optimizer):
    def __init__(self, model, lr=0.01, beta=0.9, wd=0.001, **kwargs):
        super(Momentum, self).__init__(model, wd, **kwargs)
        self.lr = lr
        self.beta = beta
        self.momentum_dict = {k: torch.zeros_like(v) for k, v in self.param_dict.items()}

    def update_dict(self):
        grad_dict = unfold_dict(self.model.grad_dict())
        grad_dict = self.wd_grad(grad_dict)
        for k in self.param_dict.keys():
            self.momentum_dict[k] = self.beta * self.momentum_dict[k] + self.lr * grad_dict[k]
        
        return self.momentum_dict

class RMSProp(Optimizer):
    def __init__(self, model, lr=0.01, beta=0.9, wd=0.001, **kwargs):
        super(RMSProp, self).__init__(model, wd, **kwargs)
        self.lr = lr
        self.beta = beta
        self.eps = 1e-6
        self.G_dict = {k: torch.zeros_like(v) for k, v in self.param_dict.items()}

    def update_dict(self):
        grad_dict = unfold_dict(self.model.grad_dict())
        grad_dict = self.wd_grad(grad_dict)
        _update_dict = {}
        for k in self.param_dict.keys():
            self.G_dict[k] = self.beta * self.G_dict[k] + (1 - self.beta) * (grad_dict[k].pow(2))
            _update_dict[k] = self.lr / torch.sqrt(self.G_dict[k] + self.eps) * grad_dict[k]
        
        return _update_dict
    
class Adam(Optimizer):
    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.9, wd=0.001, **kwargs):
        super(Adam, self).__init__(model, wd, **kwargs)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-6
        self.m_dict = {k: torch.zeros_like(v) for k, v in self.param_dict.items()}
        self.v_dict = {k: torch.zeros_like(v) for k, v in self.param_dict.items()}
        self.t = 1

    def update_dict(self):
        grad_dict = unfold_dict(self.model.grad_dict())
        grad_dict = self.wd_grad(grad_dict)
        _update_dict = {}
        for k in self.param_dict.keys():
            self.m_dict[k] = self.beta1 * self.m_dict[k] + (1 - self.beta1) * grad_dict[k]
            self.v_dict[k] = self.beta2 * self.v_dict[k] + (1 - self.beta2) * grad_dict[k].pow(2)
            m_hat = self.m_dict[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v_dict[k] / (1 - self.beta2 ** self.t)
            _update_dict[k] = self.lr * m_hat / torch.sqrt(v_hat + self.eps)
        self.t += 1

        return _update_dict
