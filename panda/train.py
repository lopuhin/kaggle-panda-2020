#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import tqdm

from .dataset import PandaDataset


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)
    arg('--batch-size', type=int, default=32)
    arg('--workers', type=int, default=4)
    arg('--epochs', type=int, default=100)
    arg('--n-patches', type=int, default=4)
    arg('--patch_size', type=int, default=256)

    args = parser.parse_args()

    df = pd.read_csv('data/train.csv')
    kfold = KFold(args.n_folds, shuffle=True, random_state=42)
    for i, (train_ids, valid_ids) in enumerate(kfold.split(df)):
        if i == args.fold:
            df_train = df.iloc[train_ids]
            # df_valid = df.iloc[valid_ids]

    root = Path('data/train_images')

    def make_loader(df):
        dataset = PandaDataset(
            root=root,
            df=df,
            patch_size=args.patch_size,
            n_patches=args.n_patches,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=PandaDataset.collate_fn,
        )

    train_loader = make_loader(df_train)

    for epoch in tqdm.trange(args.epochs, dynamic_ncols=True):
        for ids, xs, ys in tqdm.tqdm(train_loader, dynamic_ncols=True):
            pass # print(xs.shape, ys.shape)


if __name__ == '__main__':
    main()
