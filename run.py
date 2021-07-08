import argparse
import torch
from unsup3d import setup_runtime, Trainer, Unsup3d_Generator, Unsup3D_Discriminator


## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=2, type=int, help='Specify a random seed')
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)
trainer = Trainer(cfgs, Unsup3d_Generator, Unsup3D_Discriminator)
run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)

## run
if run_train:
    trainer.train()
if run_test:
    trainer.test()
