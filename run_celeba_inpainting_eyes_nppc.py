#!python
import os
import argparse
import socket

import torch
import torch.distributed as distrib
import torch.multiprocessing as mp

import nppc

def main(args, rank=0, world_size=None):
    device = args.device.split(',')[rank]

    print('Running ...')
    print(f'Hostname: {socket.gethostname()}-{device}')
    print(f'Process ID: {os.getgid()}')

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    ## Define model
    ## ------------
    model = nppc.NPPCModel(
        restoration_model_folder='./results/celeba_inpainting_eyes/restoration/',
        net_type='res_unet',
        n_dirs=5,
        lr=3e-5,
        device=device,
    )

    ## Train
    ## -----
    trainer = nppc.NPPCTrainer(
        model=model,
        batch_size=16,
        max_chunk_size=8,
        output_folder='./results/celeba_inpainting_eyes/nppc/',
        max_benchmark_samples=256,
    )
    trainer.train(
        n_steps=50000,
        log_every=20,
        benchmark_every=None,
    )

def wrapper_func(rank, n_processes, args):
    distrib.init_process_group('nccl', rank=rank, world_size=n_processes)
    main(args, rank=rank, world_size=n_processes)
    distrib.destroy_process_group()

def main_ddp_wrapper(args, n_processes=4):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_IB_DISABLE'] = '1'

    mp.spawn(
        wrapper_func,
        args=(n_processes, args),
        nprocs=n_processes,
        join=True,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_group = parser.add_argument_group('model')
    parser_group.add_argument(f'--device', default='cuda:0', type=str)

    args = parser.parse_args()

    n_processes = len(args.device.split(','))
    if n_processes == 1:
        main(args)
    else:
        main_ddp_wrapper(args, n_processes=n_processes)
