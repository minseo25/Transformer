import numpy as np
import torch
from itertools import product
from copy import deepcopy

def unfold_dict(state_dict):
    """
    input: nested dictionary, whose key is string
    output: flattened dictionary, where the key is string with "." separator 
    """
    new_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            sub_state_dict = unfold_dict(v)
            for sub_k, sub_v in sub_state_dict.items():
                new_dict[k + "." + sub_k] = sub_v
        else:
            new_dict[k] = v

    return new_dict

def fold_dict(state_dict):
    """
    output: flattened dictionary, where the key is string with "." separator 
    input: nested dictionary, whose key is string
    """
    new_dict = {}
    for k, v in state_dict.items():
        split = k.split(".")
        if len(split) == 1:
            new_dict[split[0]] = v
        else:
            key1 = split[0]
            key2 = ".".join(split[1:])
            if key1 not in new_dict:
                new_dict[key1] = {}
            new_dict[key1][key2] = v

    for k in new_dict.keys():
        if isinstance(new_dict[k], dict):
            new_dict[k] = fold_dict(new_dict[k])

    return new_dict

def test_gradient(function_name, forward_gradient_func, base_input, eps=1e-3):
    forward, gradient = forward_gradient_func
    def print_test_result(target, ours):
        is_error = torch.allclose(target, ours, atol=1e-4, rtol=1e-2)
        if is_error:
            print('\033[92m' + "✓" + '\033[0m', function_name, "test passed")
        else:
            print('\033[91m' + "✗" + '\033[0m', function_name, "test failed")
    
    f = []
    grad = []
    for x in base_input:
        grad.append(gradient(x))
        f.append((forward(x + eps) - forward(x - eps)) / (2 * eps))
    f = torch.tensor(f, dtype=torch.float32)
    grad = torch.tensor(grad, dtype=torch.float32)
    print_test_result(f, grad)
    

def test_module(module_name, module_class, module_param, base_inputs, eps=1e-3):
    def print_test_result(name, target, ours):
        is_error = torch.allclose(target, ours, atol=1e-2, rtol=1e-2)
        if is_error:
            print('\033[92m' + "✓" + '\033[0m', module_name, name, "test passed")
        else:
            print('\033[91m' + "✗" + '\033[0m', module_name, name, "test failed")

    def loss_fn(model, inputs, return_grad=False):
        output = model(*inputs)
        if isinstance(output, torch.Tensor):
            output = [output]

        loss = 0
        coeffs = []
        for o in output:
            size = o.numel()
            coeff = torch.arange(1, size + 1) / size
            coeff = coeff.reshape(o.shape).to(torch.float64)
            loss += (o * coeff).sum()
            coeffs.append(coeff)

        if not return_grad:
            return loss, None
        else:
            grad = model.backward(*coeffs)
            if isinstance(grad, torch.Tensor):
                grad = [grad]
            return loss, grad

    ##################################################################

    base_module = module_class(*module_param)
    new_module = module_class(*module_param)

    base_module.to(torch.float64)
    new_module.to(torch.float64)

    state_dict = unfold_dict(base_module.state_dict())
    if isinstance(base_inputs, torch.Tensor):
        base_inputs = [base_inputs]
    base_inputs = [bi.to(dtype=(torch.float64 if bi.dtype != torch.long else torch.long)) for bi in base_inputs]
    base_scalar, input_grad_bs = loss_fn(base_module, base_inputs, return_grad=True)
    weight_grad_b = unfold_dict(base_module.grad_dict())

    # parameter test
    for k, grad_b in weight_grad_b.items():
        grad_n = torch.zeros_like(grad_b)
        lists = [list(range(s)) for s in grad_b.shape]
        for idx in product(*lists):
            step_size = min(eps / ((grad_b[idx]).abs() + 0.001), 1e-3)
            new_dict = deepcopy(state_dict)
            new_dict[k][idx] += step_size
            new_module.load_state_dict(fold_dict(new_dict))
            new_scalar, _ = loss_fn(new_module, base_inputs)
            grad_n[idx] = (new_scalar - base_scalar) / step_size

        print_test_result(k, grad_n, grad_b)

    # input test
    for i in range(len(base_inputs)):
        base_input = base_inputs[i]
        input_grad_b = input_grad_bs[i]
        new_inputs = [bi for bi in base_inputs]

        if base_input.dtype != torch.long:
            grad_n = torch.zeros_like(base_input)
            lists = [list(range(s)) for s in base_input.shape]
            for idx in product(*lists):
                step_size = min(eps / ((input_grad_b[idx]).abs() + 0.001), 1e-3)
                new_inputs[i] = base_input.clone()
                new_inputs[i][idx] += step_size

                new_scalar, _ = loss_fn(base_module, new_inputs)
                grad_n[idx] = (new_scalar - base_scalar) / step_size

            print_test_result(f"input_{i}", grad_n, input_grad_b)

    print()

def data_loader(x, y, batch_size):
    N = len(x)
    indices = np.arange(N)
    np.random.shuffle(indices)
    for i in range(0, N, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield x[batch_indices], y[batch_indices]

def simple_update(model, lr):
    param_dict = unfold_dict(model.state_dict())
    grad_dict = unfold_dict(model.grad_dict())
    for k in param_dict.keys():
        param_dict[k] -= lr * grad_dict[k]

def crossent_loss(logit, label, class_n):
    logit_max, _ = torch.max(logit, dim=-1, keepdim=True)
    exp_logit = torch.exp((logit - logit_max) * 10)
    sum_exp = exp_logit.sum(dim=-1, keepdim=True)
    softmax = exp_logit / sum_exp

    log_prob = torch.log(softmax)
    log_prob = log_prob.reshape(-1, class_n)
    n = log_prob.shape[0]

    loss = -log_prob[torch.arange(n), label.reshape(-1)].mean()

    grad_log_prob = -torch.nn.functional.one_hot(label, num_classes=class_n).to(torch.float32) / label.numel()
    grad_log_prob_sum = grad_log_prob.sum(dim=-1, keepdim=True)
    grad_logit = (grad_log_prob - grad_log_prob_sum * softmax) * 10

    return loss, grad_logit