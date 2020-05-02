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

from .utils import crop_white, rotate_image


class PandaDataset(Dataset):
    def __init__(
            self, root: Path,
            df: pd.DataFrame,
            patch_size: int,
            n_patches: int,
            scale: float,
            level: int,
            training: bool,
            ):
        self.df = df
        self.root = root
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.scale = scale
        self.level = level
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        jpeg_path = self.root / f'{item.image_id}_{self.level}.jpeg'
        if jpeg_path.exists():
            image = jpeg4py.JPEG(jpeg_path).decode()
        else:
            image = crop_white(skimage.io.MultiImage(
                str(self.root / f'{item.image_id}.tiff'))[self.level])
        if self.scale != 1:
            image = cv2.resize(
                image, (int(image.shape[1] * self.scale),
                        int(image.shape[0] * self.scale)),
                interpolation=cv2.INTER_AREA)
        if self.training:
            image = random_flip(image)
            # image = random_rotate(image)
            image = random_pad(image, self.patch_size)
        patches = make_patches(image, n=self.n_patches, size=self.patch_size)
        xs = torch.stack([to_torch(x) for x in patches])
        assert xs.shape == (self.n_patches, 3, self.patch_size, self.patch_size)
        assert xs.dtype == torch.float32
        return item.image_id, xs, item.isup_grade


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


def random_flip(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        image = np.fliplr(image)
    return image


def random_rotate(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        return rotate_image(image, angle=random.uniform(-10, 10))
    return image


def random_pad(image: np.ndarray, size: int) -> np.ndarray:
    pad0 = random.randint(0, size)
    pad1 = random.randint(0, size)
    return np.pad(
        image,
        [[pad0, size - pad0], [pad1, size - pad1], [0, 0]],
        constant_values=255)


def make_patches(image, n: int, size: int) -> np.ndarray:
    """ Based on https://www.kaggle.com/iafoss/panda-16x128x128-tiles
    """
    pad0 = (size - image.shape[0] % size) % size
    pad1 = (size - image.shape[1] % size) % size
    pad0_0 = pad0 // 2
    pad1_0 = pad1 // 2
    image = np.pad(
        image,
        [[pad0_0, pad0 - pad0_0],
         [pad1_0, pad1 - pad1_0],
         [0, 0]],
        constant_values=255)
    image = image.reshape(
        image.shape[0] // size, size, image.shape[1] // size, size, 3)
    image = image.transpose(0, 2, 1, 3, 4).reshape(-1, size, size, 3)
    if len(image) < n:
        image = np.pad(
            image,
            [[0, n - len(image)], [0, 0], [0, 0], [0, 0]],
            constant_values=255)
    idxs = np.argsort(image.reshape(image.shape[0], -1).sum(-1))[:n]
    return image[idxs]
