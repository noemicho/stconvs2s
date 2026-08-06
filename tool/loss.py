import torch
import torch.nn as nn


PRECIP_THRESHOLDS_MM15 = (1.25, 6.25, 12.5)


def _masked_mean(error, mask, eps):
    return (error * mask).sum() / (mask.sum() + eps)


def _huber_error(yhat, y, delta):
    abs_error = torch.abs(yhat - y)
    delta_tensor = abs_error.new_full(abs_error.shape, delta)
    quadratic = torch.where(abs_error < delta, abs_error, delta_tensor)
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def _precipitation_weights(target_log, class_weights):
    target_mm15 = torch.expm1(target_log)
    weights = target_mm15.new_full(target_mm15.shape, class_weights[0])
    moderate = target_mm15.new_full(target_mm15.shape, class_weights[1])
    strong = target_mm15.new_full(target_mm15.shape, class_weights[2])
    extreme = target_mm15.new_full(target_mm15.shape, class_weights[3])
    weights = torch.where(target_mm15 >= PRECIP_THRESHOLDS_MM15[0], moderate, weights)
    weights = torch.where(target_mm15 >= PRECIP_THRESHOLDS_MM15[1], strong, weights)
    weights = torch.where(target_mm15 >= PRECIP_THRESHOLDS_MM15[2], extreme, weights)
    return weights


def _weighted_masked_mean(error, target, mask, class_weights, eps):
    weights = _precipitation_weights(target, class_weights)
    weighted_mask = weights * mask
    return (error * weighted_mask).sum() / (weighted_mask.sum() + eps)


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
        return _masked_mean(error, mask, self.eps)


class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=0.1, eps=1e-6):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, yhat, y, mask):
        error = _huber_error(yhat, y, self.delta)
        return _masked_mean(error, mask, self.eps)


class WeightedMaskedMAELoss(nn.Module):
    def __init__(self, class_weights=(1.0, 5.0, 10.0, 20.0), eps=1e-6):
        super().__init__()
        if len(class_weights) != 4:
            raise ValueError("class_weights must contain four values")
        self.class_weights = tuple(float(weight) for weight in class_weights)
        self.eps = eps

    def forward(self, yhat, y, mask):
        error = torch.abs(yhat - y)
        return _weighted_masked_mean(error, y, mask, self.class_weights, self.eps)


class WeightedMaskedHuberLoss(nn.Module):
    def __init__(self, delta=0.1, class_weights=(1.0, 5.0, 10.0, 20.0), eps=1e-6):
        super().__init__()
        if len(class_weights) != 4:
            raise ValueError("class_weights must contain four values")
        self.delta = delta
        self.class_weights = tuple(float(weight) for weight in class_weights)
        self.eps = eps

    def forward(self, yhat, y, mask):
        error = _huber_error(yhat, y, self.delta)
        return _weighted_masked_mean(error, y, mask, self.class_weights, self.eps)
