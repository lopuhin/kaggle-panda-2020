#!/usr/bin/env python3
import argparse
from pathlib import Path

import json_log_plots
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

    arg('run_root')
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)
    arg('--lr', type=float, default=1e-5)
    arg('--batch-size', type=int, default=32)
    arg('--n-patches', type=int, default=4)
    arg('--patch_size', type=int, default=256)
    arg('--epochs', type=int, default=100)
    arg('--workers', type=int, default=4)
    arg('--model', type=str, default='resnet34')
    arg('--device', type=str, default='cuda')

    args = parser.parse_args()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    to_clean = ['json-log-plots.log']
    for name in to_clean:
        path = run_root / name
        if path.exists():
            path.unlink()

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
    step = 0

    def forward(xs, ys):
        xs = xs.to(device)
        ys = ys.to(device)
        output = model(xs)
        loss = criterion(output, ys)
        return output, loss

    def train_epoch():
        nonlocal step
        model.train()
        report_freq = 5
        running_losses = []
        pbar = tqdm.tqdm(train_loader, dynamic_ncols=True, desc='train')
        for i, (ids, xs, ys) in enumerate(pbar):
            step += len(ids)
            optimizer.zero_grad()
            _, loss = forward(xs, ys)
            loss.backward()
            optimizer.step()
            running_losses.append(float(loss))
            if i and i % report_freq == 0:
                mean_loss = np.mean(running_losses)
                running_losses.clear()
                pbar.set_postfix({'loss': f'{mean_loss:.4f}'})
                json_log_plots.write_event(run_root, step, loss=mean_loss)
        pbar.close()

    @torch.no_grad()
    def validate():
        model.eval()
        losses = []
        for ids, xs, ys in valid_loader:
            output, loss = forward(xs, ys)
            losses.append(float(loss))
        return {'valid_loss': np.mean(losses)}

    model_path = run_root / 'model.pt'
    epoch_pbar = tqdm.trange(args.epochs, dynamic_ncols=True)
    for epoch in epoch_pbar:
        train_epoch()
        valid_metrics = validate()
        epoch_pbar.set_postfix(
            {k: f'{v:.4f}' for k, v in valid_metrics.items()})
        json_log_plots.write_event(run_root, step, **valid_metrics)
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
