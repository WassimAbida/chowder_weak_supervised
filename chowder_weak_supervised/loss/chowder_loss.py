"""Chowder loss init module."""

import torch.nn as nn


class LossScore(nn.BCEWithLogitsLoss):
    def __init__(self, reduction: str = "mean") -> None:
        # Implement Binary Cross Entropy loss with logits
        # See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        super(LossScore, self).__init__(
            weight=None, size_average=None, reduce=None, reduction=reduction
        )
