import torch
import torch.nn as nn


class ScaleSignLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        scale_loss = self.loss(x.abs(), y.abs())
        sign_loss = self.loss(x, y)
        return scale_loss + sign_loss
