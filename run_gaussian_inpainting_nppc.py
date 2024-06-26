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
    model = nppc.NPPCModel(
        restoration_model_folder='./results/gaussian_inpainting/restoration/',
        net_type='linear',
        n_dirs=d,
        lr=1e-4,
        device=device,
    )

    ## Train
    ## -----
    trainer = nppc.NPPCTrainer(
        model=model,
        batch_size=256,
        output_folder='./results/gaussian_inpainting/nppc/',
    )
    trainer.train(
        n_steps=100000,
        log_every=None,
    )

if __name__ == '__main__':
    main()
