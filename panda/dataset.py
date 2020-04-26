from pathlib import Path

import jpeg4py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PandaDataset(Dataset):
    def __init__(
            self, root: Path,
            df: pd.DataFrame,
            patch_size: int,
            n_patches: int,
            pseudorandom: bool = False,
            ):
        self.df = df
        self.root = root
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.pseudorandom = pseudorandom

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = jpeg4py.JPEG(self.root / f'{item.image_id}_2.jpeg').decode()
        image_xs, image_ys = (image.max(2) != 255).nonzero()
        if len(image_xs) == 0:
            image_xs, image_ys = [0], [0]
        patches = []
        ids = []
        ys = []
        state = np.random.RandomState(seed=idx if self.pseudorandom else None)
        for _ in range(self.n_patches):
            idx = state.randint(0, len(image_xs))
            patches.append(
                to_torch(cut_patch(
                    image,
                    xc=image_xs[idx],
                    yc=image_ys[idx],
                    patch_size=self.patch_size,
                )))
            ids.append(item.image_id)
            ys.append(item.isup_grade)
        xs = torch.stack(patches)
        return ids, xs, torch.tensor(ys)

    @staticmethod
    def collate_fn(batch):
        ids = sum([ids for ids, _, _ in batch], [])
        xs = torch.cat([xs for _ , xs, _ in batch])
        ys = torch.cat([ys for _ , _, ys in batch])
        return ids, xs, ys


to_torch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def cut_patch(
        image: np.ndarray, xc: int, yc: int, patch_size: int) -> np.ndarray:
    h, w, _ = image.shape
    s2 = patch_size // 2
    if xc + s2 >= w:
        xc = w - s2
    if yc + s2 >= h:
        yc = h - s2
    x0 = max(0, xc - s2)
    y0 = max(0, yc - s2)
    patch = image[y0: y0 + patch_size, x0: x0 + patch_size]
    expected_shape = (patch_size, patch_size, 3)
    if patch.shape != expected_shape:
        ph, pw, _ = patch.shape
        patch = np.pad(
            patch,
            pad_width=((0, patch_size - ph), (0, patch_size - pw), (0, 0)),
            constant_values=255,
        )
    return patch
