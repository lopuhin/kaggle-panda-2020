#!/usr/bin/env python3
from pathlib import Path
import multiprocessing

import jpeg4py
from PIL import Image
import skimage.io
import tqdm

from utils import crop_white


def to_jpeg(path: Path):
    jpeg_path = path.parent / f'{path.stem}.jpeg'
    if jpeg_path.exists():
        try:
            jpeg4py.JPEG(jpeg_path).decode()
        except Exception as e:
            print(e)
        else:
            return
    image = skimage.io.MultiImage(str(path))[1]
    image = crop_white(image)
    image = Image.fromarray(image)
    image.save(jpeg_path, quality=90)


def main():
    paths = list(Path('data/train_images').glob('*.tiff'))
    with multiprocessing.Pool() as pool:
        for _ in tqdm.tqdm(pool.imap(to_jpeg, paths), total=len(paths)):
            pass


if __name__ == '__main__':
    main()
