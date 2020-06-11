import random
from pathlib import Path

from albumentations.augmentations.functional import(
    brightness_contrast_adjust, shift_hsv)
import cv2
import pandas as pd
import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import turbojpeg

from .utils import crop_white, rotate_image


N_CLASSES = 6


class PandaDataset(Dataset):
    def __init__(
            self, root: Path,
            df: pd.DataFrame,
            patch_size: int,
            n_patches: int,
            scale: float,
            level: int,
            training: bool,
            tta: bool,
            ):
        self.df = df
        self.root = root
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.scale = scale
        self.level = level
        self.training = training
        self.tta = tta
        self._jpeg = None

    def __len__(self):
        return len(self.df)

    @property
    def jpeg(self):
        if self._jpeg is None:
            self._jpeg = turbojpeg.TurboJPEG()
        return self._jpeg

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        jpeg_path = self.root / f'{item.image_id}_{self.level}.jpeg'
        if jpeg_path.exists():
            image = self.jpeg.decode(
                jpeg_path.read_bytes(), pixel_format=turbojpeg.TJPF_RGB)
        else:
            image = skimage.io.MultiImage(
                str(self.root / f'{item.image_id}.tiff'))
            if self.level == 5:
                image = image[0]
                image = crop_white(image)
                image = cv2.resize(
                    image, (image.shape[1] // 2, image.shape[0] // 2),
                    interpolation=cv2.INTER_AREA)
            else:
                image = image[self.level]
                image = crop_white(image)
            if self.level != 0:
                image = self.jpeg.decode(
                    self.jpeg.encode(
                        image, quality=90, pixel_format=turbojpeg.TJPF_RGB),
                    pixel_format=turbojpeg.TJPF_RGB)
        if self.scale != 1:
            image = cv2.resize(
                image, (int(image.shape[1] * self.scale),
                        int(image.shape[0] * self.scale)),
                interpolation=cv2.INTER_AREA)
        if self.training or self.tta:
            if self.training:
                image = random_rotate(image)
            image = random_pad(image, self.patch_size)
        patches = make_patches(
            image, n=self.n_patches, size=self.patch_size,
            randomize=self.training)
        if self.training or self.tta:
            patches = list(map(random_flip, patches))
            patches = list(map(random_rot90, patches))
            patches = list(map(color_aug, patches))
            patches = [p.copy() for p in patches]
        xs = torch.stack([to_torch(x) for x in patches])
        assert xs.shape == (self.n_patches, 3, self.patch_size, self.patch_size)
        assert xs.dtype == torch.float32
        ys = torch.tensor(item.isup_grade, dtype=torch.float32)
        return item.image_id, xs, ys


MEAN = [0.894, 0.789, 0.857]
STD = [0.140, 0.256, 0.173]
WHITE_THRESHOLD = float(
    np.mean([(0.90 - m) / s for m, s in zip(MEAN, STD)]))
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


def color_aug(image: np.ndarray) -> np.ndarray:
    if random.random() > 0.5:
        image = brightness_contrast_adjust(
            image,
            1 + random.uniform(-0.03, 0.03),
            random.uniform(-0.05, 0.05),
            True)
    if random.random() > 0.5:
        image = shift_hsv(
            image, random.uniform(-5, 5), 0, random.uniform(-1, 1))
    return image


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
