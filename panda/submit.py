import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

from .dataset import PandaDataset
from . import models


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

    model_path = Path(args.model_path)
    params = json.loads((model_path / 'params.json').read_text())
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
    model.load_state_dict(
        torch.load(model_path / 'model.pt', map_location='cpu'))
    model.eval()

    predictions = []
    image_ids = []
    with torch.no_grad():
        for ids, xs, ys in loader:
            xs = xs.to(device)
            ys = ys.to(device)
            with amp.autocast(enabled=bool(params['amp'])):
                output = model(xs).cpu().numpy()
            predictions.extend(output.argmax(1))
            image_ids.extend(ids)

    by_image_id = dict(zip(image_ids, predictions))
    df['isup_grade'] = df['image_id'].apply(lambda x: by_image_id[x])
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
