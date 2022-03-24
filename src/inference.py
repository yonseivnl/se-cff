import os
import argparse

import torch

from manager import DLManager

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--save_root', type=str, required=True)

parser.add_argument('--num_workers', type=int, default=4)

args = parser.parse_args()

args.is_distributed = False
args.is_master = True
args.world_size = 1
args.local_rank = 0

assert os.path.isdir(args.data_root)

exp_manager = DLManager(args)
exp_manager.load(args.checkpoint_path)

exp_manager.test()
