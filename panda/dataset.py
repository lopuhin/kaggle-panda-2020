import io
import random
from pathlib import Path

import cv2
try:
    import jpeg4py
except ImportError:
    pass  # not needed on kaggle
from PIL import Image
import pandas as pd
import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import crop_white, rotate_image


N_CLASSES = 6


class PandaDataset(Dataset):
    def __init__(
            self, root: Path,
            df: pd.DataFrame,
            patch_size: int,
            n_patches: int,
            level: int,
            training: bool,
            tta: bool,
            ):
        self.df = df
        self.root = root
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.level = level
        self.training = training
        self.tta = tta

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
            if self.level != 0:
                # use PIL as jpeg4py is not available on kaggle
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format='jpeg', quality=90)
                image = np.array(Image.open(buffer))
        s = self.patch_size
        if self.training or self.tta:
            image = random_flip(image)
            image = random_rot90(image)
            # if self.training: image = random_rotate(image)
            image = random_pad(image, s)
        patches = make_patches(
            image, n=self.n_patches, size=s * 2, randomize=self.training)
        patches_low = [
            cv2.resize(patch, (s, s), interpolation=cv2.INTER_AREA)
            for patch in patches]
        patches_high = []
        for patch in patches:
            patches_high.extend([
                patch[:s, :s],
                patch[:s, s:],
                patch[s:, :s],
                patch[s:, s:],
            ])
        xs_low = torch.stack([to_torch(x) for x in patches_low])
        xs_high = torch.stack([to_torch(x) for x in patches_high])
        assert xs_low.shape == (self.n_patches, 3, s, s)
        assert xs_low.dtype == torch.float32
        assert xs_high.shape == (4 * self.n_patches, 3, s, s)
        assert xs_high.dtype == torch.float32
        ys = torch.tensor(item.isup_grade, dtype=torch.float32)
        return item.image_id, xs_low, xs_high, ys


MEAN = [0.894, 0.789, 0.857]
STD = [0.140, 0.256, 0.173]
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


def random_rot90(image: np.ndarray) -> np.ndarray:
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k)
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


def make_patches(
        image: np.ndarray, n: int, size: int, randomize: bool = False,
        ) -> np.ndarray:
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
    counts = image.reshape(image.shape[0], -1).sum(-1)
    idxs = np.argsort(counts)[:n]
    if randomize:
        max_value = 3 * size**2 * 255
        probs = (1 - counts / max_value)
        if (probs > 0).sum() > n:
            probs /= probs.sum()
            probs = probs**2
            probs /= probs.sum()
            assert probs.shape == (len(image),)
            idxs = np.random.choice(
                range(len(image)), size=n, p=probs, replace=False)
    return image[idxs]
