from utils.util import unfold_dict

class Module():
    """
    this is the abstract class for the modules.
    it has forward and backward methods:
    - forward method computes the output from the input.
    - backward method computes the gradient of the input from the gradient of the output.
    """
    def __init__(self, **kwargs):
        self.sub_modules = []

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, *args):
        raise NotImplementedError
    
    # the methods below are for the testing
    def state_dict(self):
        _state_dict = {}
        for key in self.sub_modules:
            sub_module = getattr(self, key)
            if isinstance(sub_module, Module):
                _state_dict[key] = sub_module.state_dict()
            else:
                _state_dict[key] = sub_module
        return _state_dict

    def grad_dict(self):
        _grad_dict = {}
        for key in self.sub_modules:
            sub_module = getattr(self, key)
            if isinstance(sub_module, Module):
                _grad_dict[key] = sub_module.grad_dict()
            else:
                _grad_dict[key] = getattr(self, "grad_" + key)
        return _grad_dict

    def load_state_dict(self, state_dict):
        for key in self.sub_modules:
            sub_module = getattr(self, key)
            if not key in state_dict:
                raise ValueError(f"Missing key {key} in state_dict")
            value = state_dict[key]
            if isinstance(sub_module, Module):
                sub_module.load_state_dict(value)
            else:
                setattr(self, key, value)

    def to(self, dest):
        for key in self.sub_modules:
            sub_module = getattr(self, key)
            setattr(self, key, sub_module.to(dest))

        return self

class Optimizer():
    def __init__(self, model, wd=0.0001, **kwargs):
        self.model = model
        self.param_dict = unfold_dict(self.model.state_dict())
        self.wd = wd

    def step(self):
        # Norm of gradient updates.
        # Usually used for clipping magnitude of updates or logging, but we don't use here.
        grad_norm = 0

        step_dict = self.update_dict()
        for k in self.param_dict.keys():
            self.param_dict[k] -= step_dict[k]
            grad_norm += (step_dict[k].abs()).sum()

        return grad_norm

    def wd_grad(self, grad_dict):
        # weight decay, add normalization term
        for k in grad_dict.keys():
            grad_dict[k] += self.wd * self.param_dict[k]

        return grad_dict

    def update_dict(self):
        """
        Implement this function to define the update rule.
        """
        raise NotImplementedError
