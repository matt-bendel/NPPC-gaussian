#!python
import os
import argparse
import socket

import torch

import nppc

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_group = parser.add_argument_group('model')
    parser_group.add_argument(f'--device', default='cuda:0', type=str)
    parser_group.add_argument(f'--d', default=1, type=int)
    args = parser.parse_args()

    d = args.d

    device = args.device

    print('Running ...')
    print(f'Hostname: {socket.gethostname()}-{device}')
    print(f'Process ID: {os.getgid()}')

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    ## Define model
    ## ------------
    model = nppc.RestorationModel(
        dataset='gaussian',
        data_folder=None,
        distortion_type='gaussian_1',
        net_type='linear',
        d=d,
        lr=1e-3,
        device=device,
    )

    ## Train
    ## -----
    trainer = nppc.RestorationTrainer(
        model=model,
        batch_size=256,
        output_folder='./results/gaussian_inpainting/restoration/',
        max_benchmark_samples=256,
    )
    trainer.train(
        n_steps=150000,
        log_every=None,
    )

if __name__ == '__main__':
    main()
