import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PandaDataset
from . import models
from .utils import load_weights


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model_path')
    arg('data_root')
    arg('--workers', type=int, default=4)
    arg('--device', type=str, default='cuda')
    arg('--batch-size', type=int)
    args = parser.parse_args()

    root = Path(args.data_root)
    df = pd.read_csv(root / 'sample_submission.csv')
    image_root = root / 'test_images'
    if not image_root.exists():
        df.to_csv('submission.csv', index=False)
        return

    state = torch.load(args.model_path, map_location='cpu')
    params = state['params']
    if args.batch_size:
        params['batch_size'] = args.batch_size

    dataset = PandaDataset(
        root=image_root,
        df=df,
        patch_size=params['patch_size'],
        n_patches=params['n_patches'],
        scale=params['scale'],
        level=params['level'],
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=args.workers,
    )

    device = torch.device(args.device)
    model = getattr(models, params['model'])(
        head_name=params['head'], pretrained=False)
    model.to(device)
    load_weights(model, state)
    model.eval()

    predictions = []
    image_ids = []
    with torch.no_grad():
        for ids, xs, ys in loader:
            xs = xs.to(device)
            ys = ys.to(device)
            output = model(xs).cpu().numpy()
            predictions.extend(np.digitize(output, state['bins']))
            image_ids.extend(ids)

    by_image_id = dict(zip(image_ids, predictions))
    df['isup_grade'] = df['image_id'].apply(lambda x: by_image_id[x])
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
