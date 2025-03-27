# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import timm.optim.optim_factory as optim_factory

import utils.mcc.misc as misc
import models.depth_estimate.MCC.MCC_model as mcc_model
from utils.mcc.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.mcc.hypersim_dataset import HyperSimDataset, hypersim_collate_fn
from utils.mcc.co3d_dataset import CO3DV2Dataset, co3dv2_collate_fn
from models.depth_estimate.MCC.MCC_engine import train_one_epoch, run_viz, eval_one_epoch
from utils.mcc.co3d_utils import get_all_dataset_maps
from config.mcc import get_args_parser



def build_loader(args, num_tasks, global_rank, is_train, dataset_type, collate_fn, dataset_maps):
    '''Build data loader'''
    dataset = dataset_type(args, is_train=is_train, dataset_maps=dataset_maps)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size if is_train else args.eval_batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers if is_train else args.num_eval_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )
    return data_loader


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # define the model
    model = mcc_model.get_mcc_model(
        rgb_weight=args.rgb_weight,
        occupancy_weight=args.occupancy_weight,
        args=args,
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 512

    print("base lr: %.2e" % (args.blr))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.use_hypersim:
        dataset_type = HyperSimDataset
        collate_fn = hypersim_collate_fn
        dataset_maps = None
    else:
        dataset_type = CO3DV2Dataset
        collate_fn = co3dv2_collate_fn
        dataset_maps = get_all_dataset_maps(
            args.co3d_path, args.holdout_categories,
        )

    dataset_viz = dataset_type(args, is_train=False, is_viz=True, dataset_maps=dataset_maps)
    sampler_viz = torch.utils.data.DistributedSampler(
        dataset_viz, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    data_loader_viz = torch.utils.data.DataLoader(
        dataset_viz, batch_size=1,
        sampler=sampler_viz,
        num_workers=args.num_eval_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )

    if args.run_viz:
        run_viz(
            model, data_loader_viz,
            device, args=args, epoch=0,
        )
        exit()

    data_loader_train, data_loader_val = [
        build_loader(
            args, num_tasks, global_rank,
            is_train=is_train,
            dataset_type=dataset_type, collate_fn=collate_fn, dataset_maps=dataset_maps
        ) for is_train in [True, False]
    ]

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch {epoch}:')
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args,
        )

        val_stats = {}
        if (epoch % 5 == 4 or epoch + 1 == args.epochs) or args.debug:
            val_stats = eval_one_epoch(
                model, data_loader_val,
                device, args=args,
            )

        if ((epoch % 10 == 9 or epoch + 1 == args.epochs) or args.debug):
            run_viz(
                model, data_loader_viz,
                device, args=args, epoch=epoch,
            )

        if args.output_dir and (epoch % 10 == 9 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    run_viz(
        model, data_loader_viz,
        device, args=args, epoch=-1,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)