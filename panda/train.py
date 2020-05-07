import argparse
import json
from pathlib import Path
import random

import json_log_plots
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda import amp
import tqdm

from .dataset import PandaDataset, one_from_torch, N_CLASSES
from . import models
from .utils import OptimizedRounder


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('run_root')
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)
    arg('--lr', type=float, default=3e-5)
    arg('--batch-size', type=int, default=32)
    arg('--grad-acc', type=int, default=1)
    arg('--n-patches', type=int, default=12)
    arg('--patch-size', type=int, default=128)
    arg('--scale', type=float, default=1.0)
    arg('--level', type=int, default=2)
    arg('--epochs', type=int, default=10)
    arg('--workers', type=int, default=4)
    arg('--model', default='resnet34')
    arg('--head', default='HeadFC2')
    arg('--device', default='cuda')
    arg('--validation', action='store_true')
    arg('--save-patches', action='store_true')
    arg('--lr-scheduler', default='cosine')
    arg('--amp', type=int, default=1)

    args = parser.parse_args()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    params = vars(args)
    if not args.validation:
        to_clean = ['json-log-plots.log']
        for name in to_clean:
            path = run_root / name
            if path.exists():
                path.unlink()
        (run_root / 'params.json').write_text(
            json.dumps(params, indent=4, sort_keys=True))

    df = pd.read_csv('data/train.csv')
    kfold = StratifiedKFold(args.n_folds, shuffle=True, random_state=42)
    for i, (train_ids, valid_ids) in enumerate(kfold.split(df, df.isup_grade)):
        if i == args.fold:
            df_train = df.iloc[train_ids]
            df_valid = df.iloc[valid_ids]

    root = Path('data/train_images')

    def make_loader(df, training):
        dataset = PandaDataset(
            root=root,
            df=df,
            patch_size=args.patch_size,
            n_patches=args.n_patches,
            scale=args.scale,
            level=args.level,
            training=training,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=training,
            num_workers=args.workers,
        )

    train_loader = make_loader(df_train, training=True)
    valid_loader = make_loader(df_valid, training=False)

    device = torch.device(args.device)
    model = getattr(models, args.model)(head_name=args.head)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    amp_enabled = bool(args.amp)
    scaler = amp.GradScaler(enabled=amp_enabled)
    step = 0

    lr_scheduler = None
    if args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr / 100)
    elif args.lr_scheduler:
        parser.error(f'unexpected schedule {args.schedule}')

    if args.save_patches:
        for p in run_root.glob('patch-*.jpeg'):
            p.unlink()

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
        optimizer.zero_grad()
        for i, (ids, xs, ys) in enumerate(pbar):
            step += len(ids)
            save_patches(xs)
            with amp.autocast(enabled=amp_enabled):
                _, loss = forward(xs, ys)
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_acc == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_losses.append(float(loss))
            if i and i % report_freq == 0:
                mean_loss = np.mean(running_losses)
                running_losses.clear()
                pbar.set_postfix({'loss': f'{mean_loss:.4f}'})
                json_log_plots.write_event(run_root, step, loss=mean_loss)
        pbar.close()
        if lr_scheduler is not None:
            lr_scheduler.step()

    def save_patches(xs):
        if args.save_patches:
            for i in random.sample(range(len(xs)), 1):
                j = random.randint(0, args.n_patches - 1)
                patch = Image.fromarray(one_from_torch(xs[i, j]))
                patch.save(run_root / f'patch-{i}.jpeg')

    @torch.no_grad()
    def validate():
        model.eval()
        losses = []
        predictions = []
        targets = []
        image_ids = []
        for ids, xs, ys in valid_loader:
            with amp.autocast(enabled=amp_enabled):
                output, loss = forward(xs, ys)
            losses.append(float(loss))
            predictions.extend(output.cpu().numpy())
            targets.extend(ys.cpu().numpy())
            image_ids.extend(ids)
        predictions = np.array(predictions)
        targets = np.array(targets)
        kfold = StratifiedKFold(args.n_folds, shuffle=True, random_state=42)
        oof_predictions, oof_targets = [], []
        for train_ids, valid_ids in kfold.split(targets, targets):
            rounder = OptimizedRounder(n_classes=N_CLASSES)
            rounder.fit(predictions[train_ids], targets[train_ids])
            oof_predictions.extend(rounder.predict(predictions[valid_ids]))
            oof_targets.extend(targets[valid_ids])
        metrics = {
            'valid_loss': np.mean(losses),
            'kappa': cohen_kappa_score(
                oof_targets, oof_predictions, weights='quadratic')
        }
        rounder = OptimizedRounder(n_classes=N_CLASSES)
        rounder.fit(predictions, targets)
        bins = rounder.coef_
        return metrics, bins

    model_path = run_root / 'model.pt'
    if args.validation:
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state['weights'])
        valid_metrics, bins = validate()
        for k, v in sorted(valid_metrics.items()):
            print(f'{k:<20} {v:.4f}')
        print('bins', bins)
        return

    epoch_pbar = tqdm.trange(args.epochs, dynamic_ncols=True)
    best_kappa = 0
    for epoch in epoch_pbar:
        train_epoch()
        valid_metrics, bins = validate()
        epoch_pbar.set_postfix(
            {k: f'{v:.4f}' for k, v in valid_metrics.items()})
        json_log_plots.write_event(run_root, step, **valid_metrics)
        if valid_metrics['kappa'] > best_kappa:
            best_kappa = valid_metrics['kappa']
            state = {
                'weights': model.state_dict(),
                'bins': bins,
                'metrics': valid_metrics,
                'params': params,
            }
            torch.save(state, model_path)


if __name__ == '__main__':
    main()
