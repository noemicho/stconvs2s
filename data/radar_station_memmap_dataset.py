from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class RadarStationMemmapDataset(Dataset):
    """
    Dataset PyTorch para radar + estações usando memmap.

    Estrutura esperada:

    radar_root/
        year=2022/
            radar_frames.dat
            radar_timestamps.npy
            metadata.json
            Y_all.dat
            M_all.dat
            targets_metadata.json
        year=2023/
            ...
        year=2024/
            ...

    Retorna:
        x: [C, T_in, H, W]
        y: [C, T_out, H, W]
        m: [C, T_out, H, W]
    """

    def __init__(
        self,
        radar_root,
        years,
        t_in=5,
        t_out=5,
        stride=5,
        split="train",
        train_ratio=0.6,
        val_ratio=0.2,
    ):
        self.radar_root = Path(radar_root)
        self.years = years
        self.t_in = t_in
        self.t_out = t_out
        self.stride = stride

        self.year_data = {}
        self.samples = []
        self._sample_class_cache = {}

        for year in years:
            year_dir = self.radar_root / f"year={year}"

            metadata_path = year_dir / "metadata.json"
            frames_path = year_dir / "radar_frames.dat"

            targets_metadata_path = year_dir / "targets_metadata.json"
            y_path = year_dir / "Y_all.dat"
            m_path = year_dir / "M_all.dat"

            if not metadata_path.exists():
                print(f"[AVISO] metadata não encontrado para {year}. Pulando.")
                continue

            if not frames_path.exists():
                print(f"[AVISO] radar_frames.dat não encontrado para {year}. Pulando.")
                continue


            if not targets_metadata_path.exists():
                print(f"[AVISO] targets_metadata.json não encontrado para {year}. Pulando.")
                continue

            if not y_path.exists():
                print(f"[AVISO] Y_all.dat não encontrado para {year}. Pulando.")
                continue

            if not m_path.exists():
                print(f"[AVISO] M_all.dat não encontrado para {year}. Pulando.")
                continue

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            shape = tuple(metadata["shape"])

            frames = np.memmap(
                frames_path,
                dtype=np.uint8,
                mode="r",
                shape=shape,
            )


            with open(targets_metadata_path, "r", encoding="utf-8") as f:
                targets_metadata = json.load(f)

            target_shape = tuple(targets_metadata["shape"])

            Y_all = np.memmap(
                y_path,
                dtype=np.float32,
                mode="r",
                shape=target_shape,
            )

            M_all = np.memmap(
                m_path,
                dtype=np.uint8,
                mode="r",
                shape=target_shape,
            )

            if len(frames) != len(Y_all) or len(frames) != len(M_all):
                raise ValueError(
                    f"Ano {year}: radar/Y/M têm tamanhos diferentes: "
                    f"frames={len(frames)}, Y={len(Y_all)}, M={len(M_all)}"
                )

            self.year_data[year] = {
                "frames": frames,
                "Y_all": Y_all,
                "M_all": M_all,
            }

            n_frames = len(frames)
            n_possible = n_frames - (t_in + t_out) + 1

            print(
                f"[{split}] Ano {year} carregado | "
                f"frames={n_frames} | "
                f"shape={shape} | "
                f"amostras possíveis={max(n_possible, 0)}",
                flush=True
            )

            if n_possible <= 0:
                continue

            for start_idx in range(0, n_possible, stride):
                self.samples.append((year, start_idx))

        n = len(self.samples)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        elif split == "test":
            self.samples = self.samples[val_end:]
        else:
            raise ValueError("split deve ser 'train', 'val' ou 'test'")

        print(f"[{split}] Total de amostras:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def get_sample_precipitation_classes(self, thresholds=(1.25, 6.25, 12.5)):
        """
        Classifica cada amostra pela maior precipitacao observada no horizonte de saida.

        Classes:
            0: sem observacao valida ou chuva fraca
            1: chuva moderada
            2: chuva forte
            3: chuva extrema
        """
        thresholds = tuple(float(value) for value in thresholds)

        if thresholds in self._sample_class_cache:
            return self._sample_class_cache[thresholds].copy()

        classes = np.zeros(len(self.samples), dtype=np.int64)

        for idx, (year, start_idx) in enumerate(self.samples):
            classes[idx] = self._classify_sample(year, start_idx, thresholds)

        self._sample_class_cache[thresholds] = classes
        return classes.copy()

    def get_balanced_sample_weights(self, thresholds=(1.25, 6.25, 12.5)):
        classes = self.get_sample_precipitation_classes(thresholds)
        counts = np.bincount(classes, minlength=4).astype(np.float64)
        weights_by_class = np.zeros_like(counts, dtype=np.float64)
        nonzero = counts > 0
        weights_by_class[nonzero] = 1.0 / counts[nonzero]
        return weights_by_class[classes].astype(np.float64), counts.astype(np.int64)

    def _classify_sample(self, year, start_idx, thresholds):
        data = self.year_data[year]

        y_start = start_idx + self.t_in
        y_end = y_start + self.t_out

        target_log = np.array(data["Y_all"][y_start:y_end], dtype=np.float32)
        mask = np.array(data["M_all"][y_start:y_end], dtype=np.float32)

        valid = mask > 0
        if not np.any(valid):
            return 0

        max_precip = np.expm1(target_log[valid]).max()
        return int(np.searchsorted(thresholds, max_precip, side="right"))

    def __getitem__(self, idx):
        year, start_idx = self.samples[idx]

        data = self.year_data[year]

        frames = data["frames"]
        Y_all = data["Y_all"]
        M_all = data["M_all"]

        x_start = start_idx
        x_end = start_idx + self.t_in

        y_start = x_end
        y_end = x_end + self.t_out

        # X: [T_in, H, W, 3]
        x = np.array(frames[x_start:x_end], dtype=np.float32) / 255.0

        # Y/M: [T_out, H, W, 1]
        y = np.array(Y_all[y_start:y_end], dtype=np.float32)
        m = np.array(M_all[y_start:y_end], dtype=np.float32)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        m = torch.from_numpy(m).float()

        # X: [T, H, W, C] -> [C, T, H, W]
        x = x.permute(3, 0, 1, 2)

        # Y/M: [T, H, W, C] -> [C, T, H, W]
        y = y.permute(3, 0, 1, 2)
        m = m.permute(3, 0, 1, 2)

        return x, y, m
