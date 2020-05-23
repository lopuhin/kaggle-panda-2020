import random
from pathlib import Path
from typing import Tuple

import cv2
import pandas as pd
import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import turbojpeg

from .utils import rotate_image


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
        self.jpeg = turbojpeg.TurboJPEG()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        jpeg_path = self.root / f'{item.image_id}_{self.level}.jpeg'
        if jpeg_path.exists():
            image = self.jpeg.decode(
                jpeg_path.read_bytes(), pixel_format=turbojpeg.TJPF_RGB)
            image_l2 = scale_to_l2 = None  # image above is cropped
        else:
            multi_image = skimage.io.MultiImage(
                str(self.root / f'{item.image_id}.tiff'))
            if self.level == 5:
                image = multi_image[0]
                image = cv2.resize(
                    image, (image.shape[1] // 2, image.shape[0] // 2),
                    interpolation=cv2.INTER_AREA)
            else:
                image = multi_image[self.level]
            if self.level != 0:
                image = self.jpeg.decode(
                    self.jpeg.encode(
                        image, quality=90, pixel_format=turbojpeg.TJPF_RGB),
                    pixel_format=turbojpeg.TJPF_RGB)
            image_l2 = multi_image[2]
            scale_to_l2 = [16, 4, 1, None, None, 8][self.level] * self.scale
        if self.scale != 1:
            image = cv2.resize(
                image, (int(image.shape[1] * self.scale),
                        int(image.shape[0] * self.scale)),
                interpolation=cv2.INTER_AREA)
        if self.training or self.tta:
            # if self.training: image = random_rotate(image)
            pad, image = random_pad(image, self.patch_size)
        else:
            pad = (0, 0)
        patches = make_patches(
            image, n=self.n_patches, size=self.patch_size,
            randomize=self.training,
            image_l2=image_l2, previous_pad=pad, scale_to_l2=scale_to_l2)
        if self.training or self.tta:
            patches = list(map(random_flip, patches))
            patches = list(map(random_rot90, patches))
            patches = [p.copy() for p in patches]
        xs = torch.stack([to_torch(x) for x in patches])
        assert xs.shape == (self.n_patches, 3, self.patch_size, self.patch_size), xs.shape
        assert xs.dtype == torch.float32, xs.dtype
        ys = torch.tensor(item.isup_grade, dtype=torch.float32)
        return item.image_id, xs, ys


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


def random_pad(
        image: np.ndarray, size: int) -> Tuple[Tuple[int, int], np.ndarray]:
    pad0 = random.randint(0, size)
    pad1 = random.randint(0, size)
    return (pad0, pad1), np.pad(
        image,
        [[pad0, size - pad0], [pad1, size - pad1], [0, 0]],
        constant_values=255)


def make_patches(
        image: np.ndarray, n: int, size: int, randomize: bool = False,
        image_l2: np.ndarray = None,
        previous_pad: Tuple[int, int] = None,
        scale_to_l2: float = None,
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
    image = image.transpose(0, 2, 1, 3, 4)
    image = image.reshape(-1, size, size, 3)
    if len(image) < n:  # TODO get rid of this
        image = np.pad(
            image,
            [[0, n - len(image)], [0, 0], [0, 0], [0, 0]],
            constant_values=255)
    max_value = 3 * size**2 * 255
    if image_l2 is None:
        counts = image.reshape(image.shape[0], -1).sum(-1)
    else:
        # optimization: use image_l1 to compute intensity
        pad0, pad1 = previous_pad
        pad0 += pad0_0
        pad1 += pad1_0
        pad0_l2 = int(round(pad0 / scale_to_l2))
        pad1_l2 = int(round(pad1 / scale_to_l2))
        size_l2 = int(size / scale_to_l2)
        image_l2 = np.pad(
            image_l2,
            [[pad0_l2, (size_l2 - (image_l2.shape[0] + pad0_l2) % size_l2) % size_l2],
             [pad1_l2, (size_l2 - (image_l2.shape[1] + pad1_l2) % size_l2) % size_l2],
             [0, 0]],
            constant_values=255)
        image_l2 = image_l2.reshape(
            image_l2.shape[0] // size_l2, size_l2,
            image_l2.shape[1] // size_l2, size_l2, 3)
        image_l2 = image_l2.transpose(0, 2, 1, 3, 4)
        image_l2 = image_l2.reshape(-1, size_l2, size_l2, 3)
        counts_l2 = image_l2.reshape(image_l2.shape[0], -1).sum(-1)
        counts_l2 = counts_l2 * scale_to_l2 ** 2
        diff = image.shape[0] - len(counts_l2)
        if diff > 0:
            counts_l2 = np.append(counts_l2, [max_value] * diff)
        elif diff != 0:
            counts_l2 = counts_l2[:image.shape[0]]
        assert counts_l2.shape == image.shape[:1]
        counts = counts_l2
    idxs = np.argsort(counts)[:n]
    if randomize:
        probs = (1 - counts / max_value)
        if (probs > 0).sum() > n:
            probs /= probs.sum()
            probs = probs**2
            probs /= probs.sum()
            assert probs.shape == (len(image),)
            idxs = np.random.choice(
                range(len(image)), size=n, p=probs, replace=False)
    return image[idxs]
