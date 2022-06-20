import json

import torch
import torch.nn as nn
import torch.nn.functional as F


def json_loader(path):
    with open(path, "r") as f:
        obj = json.load(f)
    print(f'load {path}')
    return obj


class LogitAdjustment(nn.Module):
    """
    Menon A K, Jayasumana S, Rawat A S, et al. Long-tail learning via logit adjustment. arXiv preprint arXiv:2007.07314, 2020.
    https://arxiv.org/pdf/2007.07314.pdf
    """
    def __init__(self, file='', tau=1.0, beta=0.999999, eps=1e-8):
        """
        file: file to load sample_per_class
        spc: sample_per_class: shape of (n_class,), total number for each label. 
        weight: weight of this loss
        tau: coefficient see paper
        beta: coefficient see paper
        """
        super(LogitAdjustment, self).__init__()
        sample_per_class = json_loader(file)

        spc = torch.tensor([(1 - beta**N) / (1 - beta) for N in sample_per_class])
        spc_norm = spc / spc.sum() + eps
        self.adjustment = tau * spc_norm.log()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        """
        :param logits: shape of (batch_size, n_class)
        :param label: shape of (batch_size, n_class)
        :return: loss: shape of (1, )
        """
        adjustment = self.adjustment.to(x.device)
        x += adjustment

        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class LogitAdjustmentLabelSmoothing(LogitAdjustment):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, **kwargs):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LogitAdjustmentLabelSmoothing, self).__init__(**kwargs)
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        adjustment = self.adjustment.to(x.device)
        x += adjustment

        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
