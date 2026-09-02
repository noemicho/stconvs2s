import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.radar_station_memmap_dataset import RadarStationMemmapDataset
from tool.train_evaluate import (
    create_metric_stats,
    metric_rows_from_stats,
    update_metric_stats,
)


class Week3SamplerMetricsTest(unittest.TestCase):
    def test_memmap_dataset_classifies_samples_by_max_target_precipitation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._create_memmap_dataset(tmpdir)

            classes = dataset.get_sample_precipitation_classes()
            weights, counts = dataset.get_balanced_sample_weights()

        np.testing.assert_array_equal(classes, np.array([0, 1, 2, 3, 0]))
        np.testing.assert_array_equal(counts, np.array([2, 1, 1, 1]))
        np.testing.assert_allclose(weights, np.array([0.5, 1.0, 1.0, 1.0, 0.5]))

    def test_metric_stats_accumulate_by_precipitation_bin_and_horizon(self):
        stats = create_metric_stats(step=2)

        target = torch.log1p(torch.tensor([[[[[0.5, 2.0]], [[8.0, 20.0]]]]]))
        output = torch.expm1(target) + torch.tensor([[[[[1.0, -1.0]], [[2.0, -2.0]]]]])
        target_mm15 = torch.expm1(target)
        mask = torch.ones_like(target_mm15)

        update_metric_stats(
            stats,
            output[:, :, 0, :, :],
            target_mm15[:, :, 0, :, :],
            mask[:, :, 0, :, :],
            horizon=0,
        )
        update_metric_stats(
            stats,
            output[:, :, 1, :, :],
            target_mm15[:, :, 1, :, :],
            mask[:, :, 1, :, :],
            horizon=1,
        )

        rows = metric_rows_from_stats(stats)
        rows_by_key = {
            (row["precipitation_bin"].split(" ")[0], row["horizon"]): row
            for row in rows
        }

        self.assertEqual(rows_by_key[("Fraca", 1)]["n"], 1)
        self.assertAlmostEqual(rows_by_key[("Fraca", 1)]["bias"], 1.0)
        self.assertEqual(rows_by_key[("Moderada", 1)]["n"], 1)
        self.assertAlmostEqual(rows_by_key[("Moderada", 1)]["mae"], 1.0)
        self.assertEqual(rows_by_key[("Forte", 2)]["n"], 1)
        self.assertAlmostEqual(rows_by_key[("Forte", 2)]["rmse"], 2.0)
        self.assertEqual(rows_by_key[("Extrema", 2)]["n"], 1)
        self.assertAlmostEqual(rows_by_key[("Extrema", 2)]["bias"], -2.0)
        self.assertEqual(rows_by_key[("Extrema", 1)]["n"], 0)
        self.assertEqual(rows_by_key[("Extrema", 1)]["rmse"], "")

    def _create_memmap_dataset(self, tmpdir):
        root = Path(tmpdir)
        year_dir = root / "year=2024"
        year_dir.mkdir()

        frame_shape = (12, 2, 2, 3)
        target_shape = (12, 2, 2, 1)

        with open(year_dir / "metadata.json", "w", encoding="utf-8") as file:
            json.dump({"shape": frame_shape}, file)

        with open(year_dir / "targets_metadata.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "shape": target_shape,
                    "Y_file": "Y_all.dat",
                    "M_file": "M_all.dat",
                },
                file,
            )

        frames = np.memmap(
            year_dir / "radar_frames.dat",
            dtype=np.uint8,
            mode="w+",
            shape=frame_shape,
        )
        frames[:] = 0
        frames.flush()

        target_mm15 = np.zeros(target_shape, dtype=np.float32)
        target_mm15[2:4] = 0.5
        target_mm15[4:6] = 2.0
        target_mm15[6:8] = 8.0
        target_mm15[8:10] = 20.0

        y_all = np.memmap(
            year_dir / "Y_all.dat",
            dtype=np.float32,
            mode="w+",
            shape=target_shape,
        )
        y_all[:] = np.log1p(target_mm15)
        y_all.flush()

        m_all = np.memmap(
            year_dir / "M_all.dat",
            dtype=np.uint8,
            mode="w+",
            shape=target_shape,
        )
        m_all[:] = 1
        m_all[10:12] = 0
        m_all.flush()

        return RadarStationMemmapDataset(
            root,
            years=[2024],
            t_in=2,
            t_out=2,
            stride=2,
            split="train",
            train_ratio=1.0,
            val_ratio=0.0,
            target_source="websirene",
        )


if __name__ == "__main__":
    unittest.main()
