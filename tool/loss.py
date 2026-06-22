import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y, mask):
        error = (yhat - y) ** 2
        masked_error = error * mask
        mse = masked_error.sum() / (mask.sum() + self.eps)
        loss = torch.sqrt(mse + self.eps)
        return loss


class MaskedMAELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y, mask):
        error = torch.abs(yhat - y)
        masked_error = error * mask
        loss = masked_error.sum() / (mask.sum() + self.eps)
        return loss