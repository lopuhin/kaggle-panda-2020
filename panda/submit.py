import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PandaDataset
from . import models


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model_path')
    arg('data_root')
    arg('--model', type=str, default='resnet34')
    arg('--batch-size', type=int, default=32)
    arg('--n-patches', type=int, default=4)
    arg('--patch_size', type=int, default=256)
    arg('--workers', type=int, default=4)
    arg('--device', type=str, default='cuda')
    args = parser.parse_args()

    root = Path(args.data_root)
    df = pd.read_csv(root / 'sample_submission.csv')
    image_root = root / 'test_images'
    if not image_root.exists():
        df.to_csv('submission.csv', index=False)
        return

    dataset = PandaDataset(
        root=image_root,
        df=df,
        patch_size=args.patch_size,
        n_patches=args.n_patches,
        pseudorandom=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=PandaDataset.collate_fn,
    )

    device = torch.device(args.device)
    model = getattr(models, args.model)(pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    predictions = []
    image_ids = []
    with torch.no_grad():
        for ids, xs, ys in loader:
            xs = xs.to(device)
            ys = ys.to(device)
            output = model(xs).cpu().numpy()
            predictions.extend(
                output.argmax(1).reshape((-1, args.n_patches)).max(1))
            image_ids.extend(ids[::args.n_patches])

    by_image_id = dict(zip(image_ids, predictions))
    df['isup_grade'] = df['image_id'].apply(lambda x: by_image_id[x])
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
