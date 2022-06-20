import math

import torch
import torch.distributed as dist
import torch.nn as nn


def init_bias_focal(module, cls_loss_type='sigmoid', init_prior=0.001, num_classes=1000):
    if cls_loss_type == 'sigmoid':
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # to keep the torch random state
                m.bias.data.normal_(-math.log(1.0 / init_prior - 1.0), init_prior)
                torch.nn.init.constant_(m.bias, -math.log(1.0 / init_prior - 1.0))
    elif cls_loss_type == 'softmax':
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.bias.data.normal_(0, 0.01)
                for i in range(0, m.bias.data.shape[0], num_classes):
                    fg = m.bias.data[i + 1:i + 1 + num_classes - 1]
                    mu = torch.exp(fg).sum()
                    m.bias.data[i] = math.log(mu * (1.0 - init_prior) / init_prior)
    else:
        raise NotImplementedError(f'{cls_loss_type} is not supported')


def all_reduce(*args, **kwargs):
    return dist.all_reduce(*args, **kwargs)


class EqualizedFocalLoss(nn.Module):
    def __init__(self, num_classes, focal_gamma=2.0, focal_alpha=0.25, scale_factor=8.0):
        super(EqualizedFocalLoss, self).__init__()
        activation_type = 'sigmoid'
        self.activation_type = activation_type
        # cfg for focal loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        self.num_classes = num_classes

        # cfg for efl loss
        self.scale_factor = scale_factor
        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes))

        print(
            f"build EqualizedFocalLoss, focal_alpha: {focal_alpha}, focal_gamma: {focal_gamma}, scale_factor: {scale_factor}"
        )

    def forward(self, x, target):
        bs, nc = x.shape
        self.cache_target = target

        pred = torch.sigmoid(x)
        pred_t = pred * target + (1 - pred) * (1 - target)

        map_val = 1 - self.pos_neg.detach()
        dy_gamma = self.focal_gamma + self.scale_factor * map_val
        # focusing factor
        ff = dy_gamma.view(1, -1).expand(bs, nc)
        # weighting factor
        wf = ff / self.focal_gamma

        # ce_loss
        ce_loss = -torch.log(pred_t)
        cls_loss = ce_loss * torch.pow((1 - pred_t), ff.detach()) * wf.detach()

        # to avoid an OOM error
        # torch.cuda.empty_cache()

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
            cls_loss = alpha_t * cls_loss

        return cls_loss.sum() / bs

    def collect_grad(self, grad):
        target = self.cache_target

        grad = torch.abs(grad.detach())
        pos_grad = torch.sum(grad * target, dim=0)
        neg_grad = torch.sum(grad * (1 - target), dim=0)

        all_reduce(pos_grad)
        all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)


def initial_gradient_collector(criterion: EqualizedFocalLoss, fc: nn.Module):
    def backward_hook(module, grad_in, grad_out):
        criterion.collect_grad(grad_out[0])

    init_bias_focal(fc)
    fc.register_backward_hook(backward_hook)


if __name__ == '__main__':
    fc = nn.Linear(2048, 1000)
    criterion = EqualizedFocalLoss(num_classes=1000)
    initial_gradient_collector(criterion, fc)
