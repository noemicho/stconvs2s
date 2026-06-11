import numpy as np
import torch
from torch.utils.data import Dataset


class NPZSequenceDataset(Dataset):
    """
    Loader genérico para datasets .npz com chaves:
      - X
      - Y
      - M opcional

    X: [N, T_in, H, W, C]
    Y: [N, T_out, H, W, C]
    M: [N, T_out, H, W, C]
    """

    def __init__(self, path, split="train", train_ratio=0.6, val_ratio=0.2):
        data = np.load(path, allow_pickle=True)

        self.X = data["X"]
        self.y = data["Y"]
        self.m = data["M"] if "M" in data.files else None

        n = len(self.X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == "train":
            self.X = self.X[:train_end]
            self.y = self.y[:train_end]
            if self.m is not None:
                self.m = self.m[:train_end]

        elif split == "val":
            self.X = self.X[train_end:val_end]
            self.y = self.y[train_end:val_end]
            if self.m is not None:
                self.m = self.m[train_end:val_end]

        elif split == "test":
            self.X = self.X[val_end:]
            self.y = self.y[val_end:]
            if self.m is not None:
                self.m = self.m[val_end:]

        else:
            raise ValueError("split deve ser 'train', 'val' ou 'test'")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32) / 255.0

        # Agora Y é chuva em mm, então NÃO divide por 255.
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        # X: [T, H, W, C] -> [C, T, H, W]
        if x.ndim == 4:
            x = x.permute(3, 0, 1, 2)

        # Y: [H, W, C] -> [C, H, W]
        if y.ndim == 3:
            y = y.permute(2, 0, 1)

        # Y: [T_out, H, W, C] -> [C, T_out, H, W]
        elif y.ndim == 4:
            y = y.permute(3, 0, 1, 2)

        if self.m is not None:
            m = torch.tensor(self.m[idx], dtype=torch.float32)

            # M: [H, W, C] -> [C, H, W]
            if m.ndim == 3:
                m = m.permute(2, 0, 1)

            # M: [T_out, H, W, C] -> [C, T_out, H, W]
            elif m.ndim == 4:
                m = m.permute(3, 0, 1, 2)

            return x, y, m

        return x, y