import random
from pathlib import Path

import cv2
try:
    import jpeg4py
except ImportError:
    pass  # not needed on kaggle
import pandas as pd
import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import crop_white


class PandaDataset(Dataset):
    def __init__(
            self, root: Path,
            df: pd.DataFrame,
            patch_size: int,
            n_patches: int,
            scale: float,
            training: bool,
            ):
        self.df = df
        self.root = root
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.scale = scale
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        jpeg_path = self.root / f'{item.image_id}_2.jpeg'
        if jpeg_path.exists():
            image = jpeg4py.JPEG(jpeg_path).decode()
        else:
            image = crop_white(skimage.io.MultiImage(
                str(self.root / f'{item.image_id}.tiff'))[2])
        image_xs, image_ys = (image.max(2) != 255).nonzero()
        if len(image_xs) == 0:
            image_xs, image_ys = [0], [0]
        xs = []
        ps = self.patch_size
        state = np.random.RandomState(seed=None if self.training else idx)
        for _ in range(self.n_patches):
            idx = state.randint(0, len(image_xs))
            x = cut_patch(
                image,
                xc=image_xs[idx],
                yc=image_ys[idx],
                patch_size=int(ps / self.scale),
            )
            if self.scale != 1:
                x = cv2.resize(x, (ps, ps), interpolation=cv2.cv2.INTER_AREA)
            if self.training:
                x = self.augment_patch(x)
            x = to_torch(x)
            assert x.shape == (3, ps, ps)
            assert x.dtype == torch.float32
            xs.append(x)
        return item.image_id, torch.stack(xs), item.isup_grade

    def augment_patch(self, x):
        # TODO do that properly
        k = random.randint(0, 3)
        if k != 0:
            x = np.rot90(x, k=k)
        if random.random() < 0.5:
            x = np.fliplr(x)
        return x.copy()


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
to_torch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def one_from_torch(x):
    assert len(x.shape) == 3
    assert x.shape[0] == 3
    x = x.cpu()
    x = torch.stack([x[i] * STD[i] + MEAN[i] for i in range(3)])
    x = x.numpy()
    x = np.rollaxis(x, 0, 3)
    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return x


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
