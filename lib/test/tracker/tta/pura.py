import operator
from collections import OrderedDict
from numbers import Number

import torch
from timm.layers import DropPath
from torch import nn
import torch.nn.functional as F


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def configure_model(model):
    model.box_head.train()
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        if isinstance(m, DropPath):
            m.drop_prob = 0.0
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model


def convert_batchnorm(module, name=None):
    if isinstance(module, nn.BatchNorm2d):
        return PURA(module, name)
    elif isinstance(module, nn.Sequential):
        children = [convert_batchnorm(child) for child in module.children()]
        return nn.Sequential(*children)
    else:
        return module


def replace_batchnorm(model, prefix=''):
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, convert_batchnorm(module, full_name))
        else:
            replace_batchnorm(module, full_name)


class PURA(nn.BatchNorm2d):
    def __init__(self, layer, name):
        super().__init__(layer.num_features, layer.eps, layer.momentum, layer.affine, layer.track_running_stats)
        self.name = name
        self.running_mean = layer.running_mean
        self.running_var = layer.running_var
        self.weight = layer.weight
        self.bias = layer.bias
        self.num_batches_tracked = layer.num_batches_tracked
        self.momentum = layer.momentum

        self.t = 1
        self.N = 3

        self.eta_mu = 1e-4
        self.eta_var = 5e-5
        self.xi = 0.6
        self.m_lambda = 0.95

        self.lambda_bar = 0

        self.pseudo_grad_queue = {
            'mean': [],
            'var': [],
        }

        self.base_weights = {
            'mean': layer.running_mean.clone(),
            'var': layer.running_var.clone(),
        }

        self.updated_weights = {
            'mean': layer.running_mean.clone(),
            'var': layer.running_var.clone(),
        }

    def forward(self, x):

        self.fast_parameter_update(x, self.pseudo_grad_queue)

        if self.t % self.N == 0:
            sum_lambda = self.parameter_decomposition_recovery()
            self.momentum = self.adaptive_lr_scaling(self.momentum, sum_lambda, tao=self.xi, pho=self.m_lambda)

        self.t += 1

        return F.batch_norm(
            x,
            self.running_mean.detach(),
            self.running_var.detach(),
            self.weight,
            self.bias,
            training=False,
            momentum=self.momentum,
            eps=self.eps
        )

    def fast_parameter_update(self, x, queue):
        mean = x.mean([0, 2, 3])
        var = x.var([0, 2, 3], unbiased=False)

        queue['mean'].append(self.base_weights['mean'] - mean.clone())
        queue['var'].append(self.base_weights['var'] - var.clone())

        n = x.numel() / x.size(1)
        self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
        self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var

        return queue

    def parameter_decomposition_recovery(self):
        self.updated_weights = {
            'mean': self.running_mean.clone(),
            'var': self.running_var.clone(),
        }

        sum_lambda = self.parameter_decomposition(self.base_weights, self.updated_weights, self.pseudo_grad_queue)

        self.running_mean = self.running_mean + self.eta_mu * self.base_weights['mean'].grad
        self.running_var = self.running_var + self.eta_var * self.base_weights['var'].grad

        self.base_weights = {
            'mean': self.running_mean.clone(),
            'var': self.running_var.clone(),
        }

        self.pseudo_grad_queue = {
            'mean': [],
            'var': [],
        }

        return sum_lambda

    def adaptive_lr_scaling(self, lr, sum_lambda, tao=0.6, pho=0.95):
        if self.lambda_bar == 0:
            self.lambda_bar = sum_lambda
            return lr

        factor = 1 + tao - abs(self.lambda_bar - sum_lambda)
        self.lambda_bar = pho * self.lambda_bar + (1 - pho) * sum_lambda
        lr_new = torch.clamp(factor, min=0.9, max=1.1) * lr
        return torch.clamp(lr_new, min=0.1, max=0.11)

    def stack(self, weights):
        stack = [[] for _ in range(self.N)]
        for i in range(self.N):
            stack[i] = [weights[ele][i].view(1, -1) for ele in weights.keys()]
            stack[i] = torch.cat(stack[i], dim=1)
        return stack

    def svd(self, centered_weights, stacked_weights):
        cov_weights = centered_weights @ centered_weights.T / (stacked_weights.size(1) - 1)
        return torch.svd(cov_weights)[:2]

    def parameter_decomposition(self, base_weights, updated_weights, params_stack):
        """
        Code adapted from https://github.com/QData/PGrad/blob/main/domainbed/algorithms.py
        """
        base_weights = ParamDict(base_weights)
        updated_weights = ParamDict(updated_weights)

        stacked_weights = self.stack(params_stack)
        stacked_weights = torch.cat(stacked_weights)
        mean_weights = stacked_weights.mean(dim=0)
        centered_weights = stacked_weights - mean_weights
        principal_directions, principal_eigenvalues = self.svd(centered_weights, stacked_weights)

        principal_directions = centered_weights.T @ principal_directions
        principal_directions = principal_directions / (principal_directions.norm(dim=0) + 1e-6)

        weight_difference = base_weights - updated_weights
        delta_theta = sum([ele.pow(2).sum() for ele in weight_difference.values()]).sqrt()
        principal_directions *= delta_theta

        start_index = 0
        gradient_mask = torch.zeros_like(principal_eigenvalues)
        for name, value in self.base_weights.items():
            param_size = value.numel()
            end_index = start_index + param_size
            pra_grad = principal_directions[start_index:end_index, :]
            calibration_direction = weight_difference[name]
            calibration_mask = (calibration_direction.flatten().unsqueeze(1) * pra_grad).sum(0)
            gradient_mask += calibration_mask
            start_index = end_index

        calibration_mask = 2 * (gradient_mask > 0).float() - 1
        pra_grad = calibration_mask * principal_directions
        comb_coef = principal_eigenvalues / principal_eigenvalues.norm()
        pra_grad = (comb_coef * pra_grad).sum(1)

        start_index = 0
        for name, value in self.base_weights.items():
            param_size = value.numel()
            end_index = start_index + param_size
            value_grad = pra_grad[start_index:end_index].view(value.size())
            start_index = end_index
            value.grad = value_grad.clone()

        return principal_eigenvalues.mean()
