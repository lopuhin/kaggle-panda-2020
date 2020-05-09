import argparse
import json
import os
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
from torch.utils.data.distributed import DistributedSampler
from torch.cuda import amp
from torch.backends import cudnn
import torch.multiprocessing as mp
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
    arg('--lr', type=float, default=6e-5)
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
    arg('--frozen', type=int, default=0)
    arg('--ddp', type=int, default=0, help='number of devices to use with ddp')
    arg('--benchmark', type=int, default=1)
    args = parser.parse_args()

    if args.ddp:
        mp.spawn(run_main, (args,), args.ddp)
    else:
        run_main(device_id=None, args=args)


def run_main(device_id, args):
    is_main = device_id in {0, None}
    n_devices = max(1, args.ddp)
    ddp_rank = device_id

    params = vars(args)
    run_root = Path(args.run_root)
    if is_main:
        run_root.mkdir(parents=True, exist_ok=True)
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

    def make_loader(df, batch_size, training):
        dataset = PandaDataset(
            root=root,
            df=df,
            patch_size=args.patch_size,
            n_patches=args.n_patches,
            scale=args.scale,
            level=args.level,
            training=training,
        )
        # TODO validate on all devices
        if training and args.ddp:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training and not sampler,
            sampler=sampler,
            num_workers=args.workers,
        )

    valid_loader = make_loader(df_valid, args.batch_size, training=False)

    if args.benchmark:
        cudnn.benchmark = True
    device = torch.device(args.device, index=device_id)
    model = getattr(models, args.model)(head_name=args.head)
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
        raise ValueError(f'unexpected schedule {args.schedule}')

    if args.ddp:
        print(f'device {device} initializing process group')
        os.environ['MASTER_PORT'] = '40390'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        torch.distributed.init_process_group(
            backend='nccl', rank=ddp_rank, world_size=n_devices)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device_id], output_device=device_id,
            find_unused_parameters=True)  # not sure why it's needed?
        print(f'process group for {device} initialized')
    else:
        model.to(device)

    if args.save_patches:
        for p in run_root.glob('patch-*.jpeg'):
            p.unlink()

    def forward(xs, ys):
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        output = model(xs)
        loss = criterion(output, ys)
        output = output.detach().cpu().numpy()
        return output, loss

    grad_acc = args.grad_acc
    batch_size = args.batch_size

    def train_epoch():
        nonlocal step
        model.train()
        report_freq = 5
        running_losses = []
        train_loader = make_loader(df_train, batch_size, training=True)
        pbar = tqdm.tqdm(train_loader, dynamic_ncols=True, desc='train',
                         disable=not is_main)
        optimizer.zero_grad()
        for i, (ids, xs, ys) in enumerate(pbar):
            step += len(ids) * n_devices
            save_patches(xs)
            with amp.autocast(enabled=amp_enabled):
                _, loss = forward(xs, ys)
            scaler.scale(loss).backward()
            if (i + 1) % grad_acc == 0:
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
            predictions.extend(output)
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
        if args.frozen:
            batch_size = args.batch_size
            grad_acc = args.grad_acc
            model.frozen = True
            if epoch == 0:
                model.frozen = False
                batch_size //= 2
                grad_acc *= 2
        train_epoch()
        if is_main:
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
