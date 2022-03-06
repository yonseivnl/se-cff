import os
import argparse

from manager import DLManager
from utils.config import get_cfg

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='/root/code/configs/config.yaml')
parser.add_argument('--data_root', type=str, default='/root/data/DSEC')
parser.add_argument('--save_root', type=str, default='/root/code/save')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--save_term', type=int, default=25)

args = parser.parse_args()

args.is_distributed = False
args.is_master = True
args.world_size = 1
args.local_rank = 0

assert os.path.isfile(args.config_path)
assert os.path.isdir(args.data_root)

# Set Config
cfg = get_cfg(args.config_path)

exp_manager = DLManager(args, cfg)
exp_manager.train()
exp_manager.test()
