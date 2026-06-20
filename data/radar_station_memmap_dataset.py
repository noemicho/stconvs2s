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