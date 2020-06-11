import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import json_log_plots
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
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
from .utils import OptimizedRounder, load_weights, train_valid_df


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
    arg('--n-test-patches', type=int)
    arg('--patch-size', type=int, default=128)
    arg('--scale', type=float, default=1.0)
    arg('--level', type=int, default=2)
    arg('--epochs', type=int, default=10)
    arg('--workers', type=int, default=8)
    arg('--model', default='resnet34')
    arg('--head', default='HeadFC2')
    arg('--device', default='cuda')
    arg('--validation', action='store_true')
    arg('--validation-save')
    arg('--tta', type=int)
    arg('--save-patches', action='store_true')
    arg('--lr-scheduler', default='cosine')
    arg('--amp', type=int, default=1)
    arg('--frozen', action='store_true')
    arg('--white-mask', type=int, default=0)
    arg('--resume')
    arg('--ddp', type=int, default=0, help='number of devices to use with ddp')
    arg('--benchmark', type=int, default=1)
    arg('--optimizer', default='adam')
    arg('--wd', type=float, default=0)
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

    df_train, df_valid = train_valid_df(args.fold, args.n_folds)
    root = Path('data/train_images')

    def make_loader(df, batch_size, training, tta, **kwargs):
        dataset = PandaDataset(
            root=root,
            df=df,
            patch_size=args.patch_size,
            n_patches=args.n_patches if training else (
                args.n_test_patches or args.n_patches),
            scale=args.scale,
            level=args.level,
            training=training,
            tta=tta,
            **kwargs,
        )
        sampler = None
        if args.ddp:
            sampler = DistributedSampler(dataset, shuffle=training)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training and not sampler,
            sampler=sampler,
            num_workers=args.workers,
        )

    if args.benchmark:
        cudnn.benchmark = True
    device = torch.device(args.device, index=device_id)
    model = getattr(models, args.model)(head_name=args.head)
    model.frozen = args.frozen
    model.white_mask = args.white_mask
    if args.optimizer == 'adam':
        assert args.wd == 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        raise ValueError(f'unknown optimizer {args.optimizer}')
    criterion = nn.SmoothL1Loss()
    amp_enabled = bool(args.amp)
    scaler = amp.GradScaler(enabled=amp_enabled)
    step = 0

    lr_scheduler = None
    lr_scheduler_per_step = False
    if args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr / 100)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [int(args.epochs * 0.6), int(args.epochs * 0.8)])
    elif args.lr_scheduler == '1cycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(make_loader(
                df_train, args.batch_size, training=True, tta=False)),
        )
        lr_scheduler_per_step = True
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
        save_patches(xs)
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        output = model(xs)
        loss = criterion(output, ys)
        return output, loss

    def train_epoch(epoch):
        nonlocal step
        model.train()
        report_freq = 5
        running_losses = []
        use_hard_aug = (epoch + 1) < 0.8 * args.epochs
        train_loader = make_loader(
            df_train, args.batch_size, training=True, tta=False,
            hard_aug=use_hard_aug)
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        pbar = tqdm.tqdm(train_loader, dynamic_ncols=True, desc='train',
                         disable=not is_main)
        optimizer.zero_grad()
        for i, (ids, xs, ys) in enumerate(pbar):
            step += len(ids) * n_devices
            with amp.autocast(enabled=amp_enabled):
                _, loss = forward(xs, ys)
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_acc == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_losses.append(float(loss))
            if lr_scheduler_per_step:
                try:
                    lr_scheduler.step()
                except ValueError as e:
                    print(e)
            if i and i % report_freq == 0:
                mean_loss = np.mean(running_losses)
                running_losses.clear()
                pbar.set_postfix({'loss': f'{mean_loss:.4f}'})
                json_log_plots.write_event(run_root, step, loss=mean_loss)
        pbar.close()
        if lr_scheduler is not None and not lr_scheduler_per_step:
            lr_scheduler.step()

    def save_patches(xs):
        if args.save_patches:
            if args.white_mask:
                batch_size, n_patches, *patch_shape = xs.shape
                xm = xs.reshape((batch_size * n_patches, *patch_shape))
                white_mask = model.get_white_mask(xm)
                white_mask = white_mask.reshape((batch_size, n_patches) + white_mask.shape[2:])
                white_mask = white_mask / (1e-2 + white_mask.max())
            for i in range(min(4, len(xs))):
                for j in range(args.n_patches):
                    patch = Image.fromarray(one_from_torch(xs[i, j]))
                    patch.save(run_root / f'patch-{i}-{j}.jpeg')
                    if args.white_mask:
                        mask = white_mask[i, j].numpy()
                        mask = np.rollaxis(np.stack([mask, mask, mask]), 0, 3)
                        mask = (mask * 255).astype(np.uint8)
                        mask = Image.fromarray(mask)
                        mask = mask.resize(xs[i, j].shape[1:][::-1])
                        mask.save(run_root / f'patch-{i}-{j}-mask.png')

    @torch.no_grad()
    def validate():
        model.eval()

        prediction_results = defaultdict(list)
        for n_tta in range(args.tta or 1):
            valid_loader = make_loader(
                df_valid, args.batch_size, training=False, tta=bool(n_tta))
            for ids, xs, ys in tqdm.tqdm(
                    valid_loader, desc='validation', dynamic_ncols=True):
                with amp.autocast(enabled=amp_enabled):
                    output, loss = forward(xs, ys)
                if n_tta == 0:
                    prediction_results['image_ids'].extend(ids)
                    prediction_results['targets'].extend(ys.cpu().numpy())
                    prediction_results['losses'].append(float(loss))
                prediction_results['predictions'].extend(
                    output.cpu().float().numpy())
        if args.tta:
            prediction_results['predictions'] = list(
                np.array(prediction_results['predictions'])
                .reshape((args.tta, -1)).mean(0))
        if args.ddp:
            paths = [run_root / f'.val_{i}.pth' for i in range(args.ddp)]
            if not is_main:
                torch.save(prediction_results, paths[ddp_rank])
            torch.distributed.barrier()
            if not is_main:
                return None, None, None
            for p in paths[1:]:
                worker_results = torch.load(p)
                for key in list(prediction_results):
                    prediction_results[key].extend(worker_results[key])
                p.unlink()

        provider_by_id = dict(
            zip(df_valid['image_id'], df_valid['data_provider']))
        predictions = np.array(prediction_results['predictions'])
        targets = np.array(prediction_results['targets'])
        image_ids = np.array(prediction_results['image_ids'])

        kfold = StratifiedKFold(args.n_folds, shuffle=True, random_state=42)
        oof_predictions, oof_targets, oof_image_ids = [], [], []
        for train_ids, valid_ids in kfold.split(targets, targets):
            rounder = OptimizedRounder(n_classes=N_CLASSES)
            rounder.fit(predictions[train_ids], targets[train_ids])
            oof_predictions.extend(rounder.predict(predictions[valid_ids]))
            oof_targets.extend(targets[valid_ids])
            oof_image_ids.extend(image_ids[valid_ids])
        oof_predictions = np.array(oof_predictions)
        oof_targets = np.array(oof_targets)
        metrics = {
            'valid_loss': np.mean(prediction_results['losses']),
            'kappa': cohen_kappa_score(
                oof_targets, oof_predictions, weights='quadratic')
        }
        oof_providers = np.array([provider_by_id[x] for x in oof_image_ids])
        for provider in set(oof_providers):
            mask = oof_providers == provider
            metrics[f'kappa_{provider}'] = cohen_kappa_score(
                oof_targets[mask], oof_predictions[mask], weights='quadratic')
        rounder = OptimizedRounder(n_classes=N_CLASSES)
        rounder.fit(predictions, targets)
        bins = rounder.coef_
        predictions_df = pd.DataFrame({
            'target': oof_targets,
            'prediction': oof_predictions,
            'image_id': oof_image_ids,
        })
        return metrics, bins, predictions_df

    def load_weights_by_path(path):
        state = torch.load(path, map_location='cpu')
        if args.ddp:
            load_weights(model.module, state)
        else:
            load_weights(model, state)

    model_path = run_root / 'model.pt'
    if args.validation:
        load_weights_by_path(model_path)
        valid_metrics, bins, predictions_df = validate()
        if is_main:
            for k, v in sorted(valid_metrics.items()):
                print(f'{k:<20} {v:.4f}')
            print('bins', bins)
            if args.validation_save:
                predictions_df.to_csv(args.validation_save, index=None)
        return

    epoch_pbar = tqdm.trange(args.epochs, dynamic_ncols=True)
    best_kappa = 0

    def _validate():
        nonlocal best_kappa
        valid_metrics, bins, _ = validate()
        if is_main:
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

    if args.resume:
        load_weights_by_path(args.resume)
        _validate()

    for epoch in epoch_pbar:
        train_epoch(epoch)
        _validate()


if __name__ == '__main__':
    main()
