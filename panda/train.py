#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm

from .dataset import PandaDataset
from . import models


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)
    arg('--lr', type=float, default=1e-4)
    arg('--batch-size', type=int, default=32)
    arg('--n-patches', type=int, default=4)
    arg('--patch_size', type=int, default=256)
    arg('--epochs', type=int, default=100)
    arg('--workers', type=int, default=4)
    arg('--model', type=str, default='resnet34')
    arg('--device', type=str, default='cuda')

    args = parser.parse_args()

    df = pd.read_csv('data/train.csv')
    kfold = KFold(args.n_folds, shuffle=True, random_state=42)
    for i, (train_ids, valid_ids) in enumerate(kfold.split(df)):
        if i == args.fold:
            df_train = df.iloc[train_ids]
            df_valid = df.iloc[valid_ids]

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
    valid_loader = make_loader(df_valid)

    device = torch.device(args.device)
    model = getattr(models, args.model)()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    def forward(xs, ys):
        xs = xs.to(device)
        ys = ys.to(device)
        output = model(xs)
        loss = criterion(output, ys)
        return output, loss

    def train_epoch():
        model.train()
        for ids, xs, ys in tqdm.tqdm(
                train_loader, dynamic_ncols=True, desc='train'):
            optimizer.zero_grad()
            _, loss = forward(xs, ys)
            loss.backward()
            optimizer.step()
            print('train loss', float(loss))

    @torch.no_grad()
    def validate():
        model.eval()
        losses = []
        for ids, xs, ys in tqdm.tqdm(
                valid_loader, dynamic_ncols=True, desc='valid'):
            output, loss = forward(xs, ys)
            losses.append(float(loss))
        print('valid loss', np.mean(losses))

    for epoch in tqdm.trange(args.epochs, dynamic_ncols=True):
        train_epoch()
        validate()


if __name__ == '__main__':
    main()
