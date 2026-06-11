import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y, mask):

        # erro quadrático
        error = (yhat - y) ** 2

        # aplica máscara
        masked_error = error * mask

        # média APENAS onde existe estação
        mse = masked_error.sum() / (mask.sum() + self.eps)

        # raiz
        loss = torch.sqrt(mse + self.eps)

        return loss