#!/usr/bin/env python3
"""
Training and testing script for LEAF, EfficientLEAF and a fixed mel filterbank.
Run with --help for command line options.
See experiments.sh for a listing of all experiments published in the
EfficientLEAF paper.

Author: Gerald Gutenbrunner, Jan Schlüter
"""
# imports
## builtin
import argparse
import time
import os
from pathlib import Path
from collections import OrderedDict

## processing
import numpy as np
from tqdm import tqdm

## torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

## efficentnet pytorch implementation
# !pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

## internal functions
from model import AudioClassifier
from model.leaf import Leaf, PCEN, Log_Compress
from engine import train
from utils import optimizer_to, scheduler_to

# Default LEAF parameters
n_filters = 40
sample_rate = 16000
window_len = 25.0
window_stride = 10.0
min_freq = 60.0
max_freq = 7800.0


# Arg Parser Function
def get_args_parser():
    parser = argparse.ArgumentParser('Leaf training and evaluation script', add_help=False)

    # General options (if no --ret-network or --frontend-benchmark, trains the model)
    parser.add_argument('--ret-network', action='store_true',
                        help='Returns the network and if --data-set is not "None" also returns all dataloaders')
    parser.add_argument('--frontend-benchmark', action='store_true',
                        help='Perform a benchmark on a model (can use --resume). Saves a benchmark.txt in the models folder.')
    parser.add_argument('--benchmark-runs', default=100, type=int,
                        help='Number of run to perform the benchmark --frontend-benchmark (default: 100)')

    # General network settings
    parser.add_argument('--seed', default=0, type=int,
                        help='if seed is 0, a random seed is taken')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume',
                        help='Path of the saved network/optimizer to resume from, ignored for new networks')
    parser.add_argument('--cudnn-benchmark', action='store_true')
    parser.add_argument('--no-cudnn-benchmark', action='store_false', dest='cudnn_benchmark')
    parser.set_defaults(cudnn_benchmark=True)

    # Training parameters
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--batch-size-eval', default=0, type=int,
                        help='Batch size used for evaluation (default: same as --batch-size')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Maximum number of epochs (default: 100, you will want to reduce that if running with --no-scheduler)')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_every', default=5, type=int,
                        help='Model will be saved per number of epochs')
    parser.add_argument('--test-every-epoch', action='store_true',
                        help='If given, compute test error every epoch, otherwise only at the end')

    # Optimizer parameters
    parser.add_argument('--lr', default=1e-4, type=float, metavar='LR',
                        help='Learning rate for Adam (default: 1e-3)')
    parser.add_argument('--frontend-lr-factor', default=1, type=float, metavar='FACTOR',
                        help='Learning rate factor for the frontend (default: %(default)s)')
    parser.add_argument('--adam-eps', default=1e-8, type=float, metavar='EPS',
                        help='Epsilon for Adam (default: 1e-8)')
    parser.add_argument('--warmup-steps', default=0, type=int, metavar='STEPS',
                        help='Number of update steps for a linear learning rate warmup (default: %(default)s)')

    # Scheduler parameters
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--no-scheduler', action='store_false', dest='schedule')
    parser.set_defaults(scheduler=True)
    parser.add_argument('--scheduler-mode', default='loss', choices=['acc', 'loss'],
                        type=str, help='Should the scheduler focus on maximizing the acc or minimizing the loss')
    parser.add_argument('--scheduler-factor', default=0.1, type=float,
                        help='Only needed if a scheduler is used. Factor that the LR will be reduced.')
    parser.add_argument('--patience', default=20, type=int, metavar='EPOCHS',
                        help='If a scheduler is used, reduce learning rate after this many epochs without improvement (default: %(default)s)')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='Only needed if a scheduler is used (default: 1e-5)')

    # Dataset parameters
    parser.add_argument('--data-path', default='/media/unsw/172E-A21B/IS2023/leaf_dataset', type=str,
                        help='path below which to store the datasets (defaults to current directory)')
    parser.add_argument('--data-set', default='CREMAD_SEN_90', choices=['CREMAD', 'CREMAD_90', 'CREMAD_SPEAKERS','CREMAD_SEN_90','None'],
                        type=str, help='Which dataset to use. "None" does not load any dataset; used with --ret-network to return a network without dataloaders.')
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--eval-pad', default='drop', type=str, choices=('zero', 'drop', 'overlap'),
                        help='If and how to deal with incomplete evaluation chunks: "zero" for zero-padding, "drop" for omitting them (default), "overlap" for overlapping with the previous chunk')
    parser.add_argument('--eval-overlap', default=0, type=float,
                        help='Amount or fraction of overlap between consecutive evaluation chunks (default: %(default)s)')

    # Saving parameters
    parser.add_argument('--save-best-model', default='loss', choices=['acc', 'loss'],
                        type=str,
                        help='Which metric ("acc" or "loss" ) for saving the best performing model based on of validation set ("net_best_model.pth"')
    parser.add_argument('--overwrite-save', action='store_true',
                        help='Overwrites the saved run file for every --save-every as the current run with "net_checkpoint.pth"')
    parser.add_argument('--no-overwrite-save', action='store_false',
                        help='Save the best run and all the runs between "net_e(--save-every)checkpoint.pth"',
                        dest='overwrite_save')
    parser.set_defaults(overwrite_save=True)
    parser.add_argument('--save-every', default=1, type=int, help='Interval of epochs between saving the model')

    # General frontend parameters
    parser.add_argument('--frontend', default='Leaf', type=str,
                        choices=('Leaf', 'EfficientLeaf', 'Mel'),
                        help='Frontend type (Leaf, EfficientLeaf or Mel)')
    parser.add_argument('--input-size', default=sample_rate*3, type=int,
                        help='How long the input excerpts are in samples (default: %(default)s)')
    parser.add_argument('--output-dir', default='/media/unsw/172E-A21B/IS2023/noise_outputs',
                        help='Path where the network and tensorboard logs are saved')

    # Compression parameters
    parser.add_argument('--compression', default='PCEN', type=str,
                        help='Which compression method should be used PCEN (original Leaf) or Log. (default PCEN)')
    parser.add_argument('--pcen-learn-logs', action='store_true',
                        help="If given, learns logarithms of PCEN parameters as in PCEN paper, otherwise learns parameters directly as in Leaf paper.")
    # Model Name
    parser.add_argument('--model-name', default='speaker-independent-model-fixed-noisy',
                        help='run name for subdirectory in outputs/models and outputs/runs (may contain slashes)')

    # trainable layers
    parser.add_argument('--fixed_filter', default=False, type=bool,
                        help='Set the filtering and pooling layer not learnable')
    parser.add_argument('--fixed_PCEN', default=False, type=bool,
                        help='Set the PCEN compression layer not learnable')
    parser.add_argument('--fixed_all_PCEN', default=False, type=bool,
                        help='Set the PCEN compression layer not learnable')
    parser.add_argument('--fixed_backend', default=False, type=bool,
                        help='Set the backend classifier not learnable')
    parser.add_argument('--target_speaker', default=None, type=int,
                         help='input the target speaker model you want to train')
    parser.add_argument('--noise', default=False, type=bool,
                         help='add gaussian to all train, validation and test data')
    parser.add_argument('--babble', default=False, type=bool,
                         help='add reverbration to all train. validation and test data')
    parser.add_argument('--noise_test',default=False, type=bool,
                        help='Using the noisy data for testing')
    parser.add_argument('--babble_test',default=True, type=bool,
                        help='Using the noisy data for testing')

    parser.add_argument('--tune',default=False, type=bool,
                        help='tuning the noise adaptation model using noise/reverb data for one sentences')
    # adaptaion parameters
    parser.add_argument('--adaptation', default=False, type=bool,
                        help='Speaker ID for adaptation')

    parser.add_argument('--level', default=0, type=int,
                        help='noise level for noise experiment (dB)')

    return parser


def main(args):
    device = torch.device(args.device)

    ## fix the seed for reproducibility
    if args.seed:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)

    cudnn.benchmark = args.cudnn_benchmark

    ## init dataset and dataloader
    if args.data_set == 'CREMAD':
        from datasets.crema_d import build_dataset
    if args.data_set == 'CREMAD_90':
        from datasets.crema_d import build_dataset_90 as build_dataset
    if args.data_set == 'CREMAD_SPEAKERS':
        from datasets.crema_d import build_dataset_speakers as build_dataset
    if args.data_set == 'CREMAD_SEN_90':
        from datasets.crema_d import build_dataset_90_sentences as build_dataset

    if args.data_set != 'None':
        train_loader, val_loader, test_loader, args.nb_classes = build_dataset(args=args)

    ## init encoder
    frontend_channels = 1
    encoder = EfficientNet.from_name("efficientnet-b0", num_classes=args.nb_classes, include_top=False,
                                     in_channels=frontend_channels)
    encoder._avg_pooling = torch.nn.Identity()

    ## init compression layer
    if args.compression == 'PCEN':
        compression_fn = PCEN(num_bands=n_filters,
                              s=0.04,
                              alpha=0.96,
                              delta=2.0,
                              r=0.5,
                              eps=1e-12,
                              learn_logs=args.pcen_learn_logs,
                              clamp=1e-5)

    ## init frontend
    if args.frontend == 'Leaf':
        frontend = Leaf(n_filters=n_filters,
                        min_freq=min_freq,
                        max_freq=max_freq,
                        sample_rate=sample_rate,
                        window_len=window_len,
                        window_stride=window_stride,
                        compression=compression_fn)


    ## init classifier
    network = AudioClassifier(
        num_outputs=args.nb_classes,
        frontend=frontend,
        encoder=encoder)

    def criterion_and_optimizer(args, network):
        ## init criterion, optimizer and set scheduler
        criterion = nn.CrossEntropyLoss(reduction='none')
        if args.frontend_lr_factor == 1:
            params = network.parameters()
            if args.fixed_filter == True:
                for param in network._frontend.filterbank.parameters():
                    param.requires_grad = False
            if args.fixed_PCEN == True:
                for name, param in network._frontend.compression.named_parameters():
                    if name == 'delta' or name == 's':
                        param.requires_grad = False
            if args.fixed_all_PCEN == True:
                for param in network._frontend.compression.parameters():
                    param.requires_grad = False
            if args.fixed_backend == True:
                for param in network._encoder.parameters():
                    param.requires_grad = False

        else:
            frontend_params = list(network._frontend.parameters())
            frontend_paramids = set(id(p) for p in frontend_params)
            params = [p for p in network.parameters()
                      if id(p) not in frontend_paramids]
        optimizer = torch.optim.Adam(params, lr=args.lr, eps=args.adam_eps)
        if args.frontend_lr_factor != 1:
            optimizer.add_param_group(dict(
                params=frontend_params,
                lr=args.lr * args.frontend_lr_factor))

        ## lr scheduler
        if args.scheduler:
            args.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max' if args.scheduler_mode == 'acc' else 'min',
                factor=args.scheduler_factor,
                patience=args.patience)
        else:
            args.scheduler = None

        return args, criterion, optimizer

    if not args.resume or not os.path.exists(args.resume):
        args, criterion, optimizer = criterion_and_optimizer(args, network)

    ## load previous run
    if args.resume and not os.path.exists(args.resume):
        print("resume file %s does not exist; ignoring" % args.resume)

    if args.resume and os.path.exists(args.resume):
        saved_dict = torch.load(args.resume, map_location=torch.device('cpu'))
        network.load_state_dict(saved_dict['network'])
        args, criterion, optimizer = criterion_and_optimizer(args, network)
        args.start_epoch = 1
        optimizer.load_state_dict(saved_dict['optimizer'])
        if args.scheduler is not None and saved_dict['scheduler'] is not None:
            args.scheduler.load_state_dict(saved_dict['scheduler'])
        del saved_dict

    ## move network and optim to device
    network.to(device)
    torch.cuda.empty_cache()
    optimizer_to(optimizer, device)
    scheduler_to(args.scheduler, device)

    # return network if wanted
    if args.ret_network:
        if args.data_set != "None":
            return network, train_loader, val_loader, test_loader
        else:
            return network

    ## save args for this experiment
    model_dir = os.path.join(args.output_dir, 'models', args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
        f.writelines('%s=%s\n' % (k, v) for k, v in vars(args).items())

    if args.frontend_benchmark and getattr(network, '_frontend', None):
        print("train: %d, valid: %d, test: %d" % (len(train_loader.dataset), len(val_loader.dataset.dataset), len(test_loader.dataset.dataset)))
        print("Performing frontend training benchmark...")
        batch, *_ = next(iter(val_loader))
        batch_size = batch.shape[0]
        batch = batch.to(device)
        network.train()
        #warmup
        for i in tqdm(range(args.benchmark_runs // 10), 'Warmup'):
            network._frontend(batch).sum().backward()
        #perform benchmark
        torch.cuda.synchronize()
        start = time.time()
        for i in tqdm(range(args.benchmark_runs), 'Benchmark'):
            network._frontend(batch).sum().backward()
        torch.cuda.synchronize()
        end = time.time()

        #calculate stats
        time_elpsd = end - start
        sap_per_sec = args.benchmark_runs * batch_size / time_elpsd
        print('Time Elapsed:', time_elpsd)
        print('Samples/Sec:', sap_per_sec)

        #save into model_path\benchmark_time.txt
        model_path = os.path.join(args.output_dir, 'models', args.model_name)
        if not os.path.isdir(os.path.join(args.output_dir, 'models')): os.mkdir(os.path.join(args.output_dir, 'models'))
        if not os.path.isdir(model_path): os.mkdir(model_path)
        with open(os.path.join(model_path, "benchmark_time.txt"), 'w') as f:
            f.write('time={}\nsamples_per_sec={}'.format(time_elpsd, sap_per_sec))
    else:
        train(network=network, loader_train=train_loader, loader_val=val_loader, loader_test=test_loader,
              path=args.output_dir, criterion=criterion, optimizer=optimizer, num_epochs=args.epochs, tqdm_on=True,
              overwrite_save=args.overwrite_save, save_every=args.save_every, starting_epoch=args.start_epoch,
              test_every_epoch=args.test_every_epoch,
              scheduler=None, scheduler_item=args.scheduler_mode, scheduler_min_lr=args.min_lr,
              warmup_steps=args.warmup_steps,
              save_best_model=args.save_best_model, model_name=args.model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Torch Leaf Trainings Script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
