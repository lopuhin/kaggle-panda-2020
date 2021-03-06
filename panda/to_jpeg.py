#!/usr/bin/env python3
from pathlib import Path
import multiprocessing

import cv2
import turbojpeg
import numpy as np
import skimage.io
import tqdm

from .utils import crop_white


_jpeg = None


def to_jpeg(path: Path):
    image = skimage.io.MultiImage(str(path))
    image_to_jpeg(path, '_1', image[1])
    image_to_jpeg(path, '_2', image[2])
    image_0 = crop_white(image[0])
    image_5 = cv2.resize(
        image_0, (image_0.shape[1] // 2, image_0.shape[0] // 2),
        interpolation=cv2.INTER_AREA)
    image_to_jpeg(path, '_5', image_5)


def image_to_jpeg(path: Path, suffix: str, image: np.ndarray):
    global _jpeg
    if _jpeg is None:
        _jpeg = turbojpeg.TurboJPEG()
    jpeg_path = path.parent / f'{path.stem}{suffix}.jpeg'
    if jpeg_path.exists():
        try:
            _jpeg.decode(jpeg_path.read_bytes())
        except Exception as e:
            print(e)
        else:
            return
    image = crop_white(image)
    jpeg_path.write_bytes(
        _jpeg.encode(image, quality=90, pixel_format=turbojpeg.TJPF_RGB))


def main():
    paths = list(Path('data/train_images').glob('*.tiff'))
    with multiprocessing.Pool() as pool:
        for _ in tqdm.tqdm(pool.imap(to_jpeg, paths), total=len(paths)):
            pass


if __name__ == '__main__':
    main()
