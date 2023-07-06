# basic libraries
import numpy as np
import random
import os
import time
import datetime
import argparse
from pprint import pprint
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

# tadlib
from core import load_config
from modeling import make_head
from train import (
    train_one_epoch,
    save_checkpoint,
    make_optimizer,
    make_scheduler,
    fix_random_seed,
    ModelEma
)

"""Main program for training"""
parser = argparse.ArgumentParser(
    description='A Python program based on Mlp-like for action detection'
)
parser.add_argument('config', metavar='DIR',
                    help='config path')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    help='frequency (default: 20 iterations)')
parser.add_argument('-c', '--ckpt-freq', default=10, type=int,
                    help='checkpoint frequency (default: every 10 epochs)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to a checkpoint (default: none)')
parser.add_argument('--output', default='', type=str,
                    help='output folder (default: none)')
args = parser.parse_args()

###################################################################
def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

datasets = {}
def make_dataset(name, is_training, split, **kwargs):
   """
       A simple dataset builder
   """
   dataset = datasets[name](is_training, split, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True
    )
    return loader
def main(args):
    """Main function for training / inference"""

    """1. Parameters"""
    # Parse arguments
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # Prepare output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts)
        )
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output)
        )
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # Tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # Fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # Rescale learning rate / number of workers based on the number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. Create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # Update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # Data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader']
    )

    """3. Create model, optimizer, and scheduler"""
    # Model
    model = make_head(cfg['model_name'], **cfg['model'])
    # Not ideal for multi-GPU training, but okay for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # Optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # Scheduler
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # Enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # Resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # Load checkpoint, reset epoch / best RMSE
            checkpoint = torch.load(
                args.resume,
                map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
            )
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # Also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> Loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))
            return

    # Save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. Training / Validation Loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # Start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    for epoch in range(args.start_epoch, max_epochs):
        # Train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        # Save checkpoint once in a while
        if (
            (epoch == max_epochs - 1) or
            (
                (args.ckpt_freq > 0) and
                (epoch % args.ckpt_freq == 0) and
                (epoch > 0)
            )
        ):
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

    # Wrap up
    tb_writer.close()
    print("All done!")
    return

## Train ##
if __name__ == '__main__':
    main(args)
