import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import tqdm

from .dataset import PandaDataset
from . import models as panda_models
from .utils import load_weights, train_valid_df


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('data_root')
    arg('models', nargs='+')
    arg('--workers', type=int, default=4)
    arg('--device', type=str, default='cuda')
    arg('--batch-size', type=int)
    arg('--tta', type=int)
    arg('--output')
    arg('--fold', type=int)
    arg('--n-folds', type=int, default=5)
    args = parser.parse_args()

    root = Path(args.data_root)
    is_train = args.fold is not None
    if is_train:
        image_root = root / 'train_images'
        _, df = train_valid_df(args.fold, args.n_folds)
    else:
        df = pd.read_csv(root / 'sample_submission.csv')
        image_root = root / 'test_images'
        if not image_root.exists():
            df.to_csv('submission.csv', index=False)
            return

    states = [torch.load(m, map_location='cpu') for m in args.models]
    params = states[0]['params']
    if args.batch_size:
        params['batch_size'] = args.batch_size

    device = torch.device(args.device)
    models = [
        getattr(panda_models, s['params']['model'])(
            head_name=s['params']['head'], pretrained=False)
        for s in states]
    for m, s in zip(models, states):
        m.white_mask = s['params']['white_mask']
        m.to(device)
        load_weights(m, s)
        m.eval()

    predictions = [[] for _ in models]
    image_ids = []

    n_tta = args.tta or 1
    dataset = PandaDataset(
        root=image_root,
        df=df,
        patch_size=params['patch_size'],
        n_patches=params['n_test_patches'] or params['n_patches'],
        scale=params['scale'],
        level=params['level'],
        training=False,
        n_tta=n_tta,
    )
    loader = DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=args.workers,
    )
    with torch.no_grad():
        if is_train:
            loader = tqdm.tqdm(loader)
        for ids, xs, ys in loader:
            xs = xs.to(device)
            ys = ys.to(device)
            image_ids.extend(ids)
            for i, m in enumerate(models):
                predictions[i].extend(m(xs).cpu().numpy())

    image_ids = image_ids[::n_tta]
    predictions = np.mean([
        np.array(x).reshape((-1, n_tta)).mean(1) for x in predictions], 0)

    bins = np.mean([s['bins'] for s in states], 0)
    binned_predictions = np.digitize(predictions, bins)
    by_image_id = dict(zip(image_ids, binned_predictions))
    df['isup_grade'] = df['image_id'].apply(lambda x: by_image_id[x])
    df.to_csv('submission.csv', index=False)

    if args.output:
        output = {
            'image_ids': image_ids,
            'predictions': list(map(float, predictions)),
            'bins': list(map(float, bins)),
            'params': [s['params'] for s in states],
            'metrics': [s['metrics'] for s in states],
        }
        Path(args.output).write_text(
            json.dumps(output, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
