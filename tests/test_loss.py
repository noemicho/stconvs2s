import unittest

import torch

from tool.loss import (
    MaskedHuberLoss,
    MaskedMAELoss,
    WeightedMaskedHuberLoss,
    WeightedMaskedMAELoss,
)


class LossTest(unittest.TestCase):
    def test_masked_mae_ignores_masked_positions(self):
        y = torch.tensor([0.0, 1.0, 2.0])
        yhat = torch.tensor([1.0, 5.0, 4.0])
        mask = torch.tensor([1.0, 0.0, 1.0])

        loss = MaskedMAELoss(eps=0.0)(yhat, y, mask)

        self.assertAlmostEqual(loss.item(), 1.5)

    def test_masked_huber_uses_delta_piecewise(self):
        y = torch.tensor([0.0, 0.0])
        yhat = torch.tensor([0.5, 2.0])
        mask = torch.tensor([1.0, 1.0])

        loss = MaskedHuberLoss(delta=1.0, eps=0.0)(yhat, y, mask)

        self.assertAlmostEqual(loss.item(), 0.8125)

    def test_weighted_mae_uses_target_precipitation_bins(self):
        target_mm15 = torch.tensor([0.5, 2.0, 8.0, 20.0])
        y = torch.log1p(target_mm15)
        yhat = y + torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.ones_like(y)

        loss = WeightedMaskedMAELoss(
            class_weights=(1.0, 2.0, 3.0, 4.0),
            eps=0.0,
        )(yhat, y, mask)

        self.assertAlmostEqual(loss.item(), 3.0)

    def test_weighted_mae_uses_weighted_denominator(self):
        target_mm15 = torch.tensor([0.5, 20.0])
        y = torch.log1p(target_mm15)
        yhat = y + torch.tensor([1.0, 3.0])
        mask = torch.ones_like(y)

        loss = WeightedMaskedMAELoss(
            class_weights=(1.0, 10.0, 10.0, 10.0),
            eps=0.0,
        )(yhat, y, mask)

        self.assertAlmostEqual(loss.item(), 31.0 / 11.0, places=6)

    def test_weighted_huber_combines_bins_and_delta(self):
        target_mm15 = torch.tensor([0.5, 20.0])
        y = torch.log1p(target_mm15)
        yhat = y + torch.tensor([0.5, 2.0])
        mask = torch.ones_like(y)

        loss = WeightedMaskedHuberLoss(
            delta=1.0,
            class_weights=(1.0, 10.0, 10.0, 10.0),
            eps=0.0,
        )(yhat, y, mask)

        self.assertAlmostEqual(loss.item(), 15.125 / 11.0, places=6)


if __name__ == "__main__":
    unittest.main()
