import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import tqdm

from .dataset import PandaDataset, N_CLASSES
from . import models
from .utils import load_weights, train_valid_df


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model_path')
    arg('data_root')
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

    # TODO multiple models
    state = torch.load(args.model_path, map_location='cpu')
    params = state['params']
    if args.batch_size:
        params['batch_size'] = args.batch_size

    device = torch.device(args.device)
    model = getattr(models, params['model'])(
        head_name=params['head'], pretrained=False)
    model.to(device)
    load_weights(model, state)
    model.eval()

    predictions = []
    image_ids = []

    for n_tta in range(args.tta or 1):
        dataset = PandaDataset(
            root=image_root,
            df=df,
            patch_size=params['patch_size'],
            n_patches=params['n_test_patches'] or params['n_patches'],
            scale=params['scale'],
            level=params['level'],
            training=False,
            tta=bool(n_tta),
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
                output = model(xs).cpu().numpy()
                predictions.extend(output)
                if n_tta == 0:
                    image_ids.extend(ids)
    if args.tta:
        predictions = (np.array(predictions)
                       .reshape((args.tta, -1, N_CLASSES)).mean(0))

    isup_predictions = np.argmax(predictions, -1)
    by_image_id = dict(zip(image_ids, isup_predictions))
    df['isup_grade'] = df['image_id'].apply(lambda x: by_image_id[x])
    df.to_csv('submission.csv', index=False)

    if args.output:
        output = {
            'image_ids': image_ids,
            'predictions': [
                list(map(float, logits)) for logits in predictions],
            'params': state['params'],
            'metrics': state['metrics'],
        }
        Path(args.output).write_text(
            json.dumps(output, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
