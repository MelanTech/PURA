from timm.layers import DropPath
from torch import nn
import torch.nn.functional as F


def collect_params(model):
    params = []
    names = []

    for nm, m in model.named_modules():
        if 'box_head' in nm:
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                names.append(f"{nm}.running_mean")
                params.append(m.running_mean)
                names.append(f"{nm}.running_var")
                params.append(m.running_var)

    return params, names


def configure_model(model):
    # train mode, because tta optimizes the model to minimize entropy
    model.box_head.train()
    # disable grad, to (re-)enable only what fstta updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for nm, m in model.named_modules():
        # if 'backbone.norm' in nm:
        #     m.requires_grad_(True)
        if isinstance(m, DropPath):
            m.drop_prob = 0.0
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        # if isinstance(m, nn.BatchNorm2d):
        #     if 'conv1_ctr' in nm or 'conv2_ctr' in nm or 'conv3_ctr' in nm or 'conv4_ctr' in nm:
        #         m.requires_grad_(True)
        # force use of batch stats in train and eval modes
        # m.track_running_stats = False
        # if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
        #     m.requires_grad_(True)
    return model


def convert_batchnorm(module, name=None):
    if isinstance(module, nn.BatchNorm2d):
        return AdaBN(module, name)
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


class AdaBN(nn.BatchNorm2d):
    def __init__(self, layer, name):
        super().__init__(layer.num_features, layer.eps, layer.momentum, layer.affine, layer.track_running_stats)
        self.name = name
        self.running_mean = layer.running_mean
        self.running_var = layer.running_var
        self.weight = layer.weight
        self.bias = layer.bias
        self.num_batches_tracked = layer.num_batches_tracked

        self.t = 1

        self.momentum = layer.momentum

    def forward(self, x):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        mean = x.mean([0, 2, 3])
        var = x.var([0, 2, 3], unbiased=False)

        d = mean - self.running_mean
        self.running_mean = self.running_mean + d / self.t
        self.running_var = (self.running_var * self.t) / (self.t + 1) + var / (self.t + 1) + (
                d ** 2 * self.t) / (self.t + 1) ** 2

        self.t += 1

        return F.batch_norm(
            x,
            self.running_mean.detach(),
            self.running_var.detach(),
            self.weight,
            self.bias,
            training=False,
            momentum=exponential_average_factor,
            eps=self.eps
        )
