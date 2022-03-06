import os
import shutil

from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from ..config import get_cfg


def get_time():
    fmt = "%Y-%m-%d %H:%M:%S %Z%z"
    kst_time = datetime.now().strftime(fmt)

    return kst_time


class Log:
    def __init__(self, log_path):
        self._log_path = log_path

    def write(self, log, mode='a', end='\n', is_print=True, add_time=True):
        if add_time:
            log = '%s: %s' % (get_time(), log)
        if is_print:
            print(log, end=end)
        with open(self._log_path, mode=mode) as f:
            f.write(log + end)


class ExpLogger:
    _FILE_NAME = {
        'args': 'args.txt',
        'config': 'config.yaml',
        'model': 'model.txt',
        'optimizer': 'optimizer.txt',
        'train': 'train_log.txt',
        'validation': 'validation_log.txt',
        'test': 'test_log.txt',
    }
    _DIR_NAME = {
        'src': 'src',
        'weight': 'weights',
        'visualize': 'visualize',
    }

    def __init__(self, save_root, mode='train'):
        assert mode in ['train', 'validation', 'test']
        self._save_root = save_root
        self._mode = mode
        self._tensor_log = None

        os.makedirs(self._save_root, exist_ok=True)

    def train(self):
        self._mode = 'train'

    def test(self):
        self._mode = 'test'

    def write(self, log, file_name=None, mode='a', end='\n', is_print=True, add_time=True):
        if file_name is None:
            file_name = self._FILE_NAME[self._mode]
        log_path = os.path.join(self._save_root, file_name)
        logger = Log(log_path)
        logger.write(log=log, end=end, is_print=is_print, add_time=add_time, mode=mode)

    def add_scalar(self, tag, scalar_value, global_step):
        if self._tensor_log is None:
            self._tensor_log = SummaryWriter(self._save_root)
        self._tensor_log.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    def save_args(self, args):
        args_log = ''
        for argument in args.__dict__.keys():
            args_log += '--%s %s \\\n' % (argument, args.__dict__[argument])

        self.write(log=args_log, file_name=self._FILE_NAME['args'], mode='w', add_time=False)

    def save_cfg(self, cfg):
        self.write(log=str(cfg), file_name=self._FILE_NAME['config'], mode='w', add_time=False)

    def load_cfg(self):
        cfg_path = os.path.join(self._save_root, self._FILE_NAME['config'])
        cfg = get_cfg(cfg_path)

        return cfg

    def log_model(self, model):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        self.write(log=str(model), file_name=self._FILE_NAME['model'], mode='w', add_time=False)
        self.write(log='Total number of parameters: %d' % num_params,
                   file_name=self._FILE_NAME['model'], add_time=False)

    def log_optimizer(self, optimizer):
        self.write(log=str(optimizer), file_name=self._FILE_NAME['optimizer'], mode='w', add_time=False)

    def save_src(self, src_root):
        src_save_path = os.path.join(self._save_root, self._DIR_NAME['src'])
        if os.path.exists(src_save_path):
            shutil.rmtree(src_save_path)
        shutil.copytree(src_root, src_save_path)

    def save_file(self, file, file_name):
        torch.save(file, os.path.join(self._save_root, file_name))

    def load_file(self, file_name):
        return torch.load(os.path.join(self._save_root, file_name))

    def save_checkpoint(self, checkpoint, name):
        checkpoint_root = os.path.join(self._save_root, self._DIR_NAME['weight'])
        os.makedirs(checkpoint_root, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_root, name)
        torch.save(checkpoint, checkpoint_path)

        self.write(log='Checkpoint is saved to %s' % checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self.write(log='Checkpoint is Loaded from %s' % checkpoint_path)

        return checkpoint

    def save_visualize(self, image, visual_type, sequence_name, image_name):
        visualize_root = os.path.join(self._save_root, self._DIR_NAME['visualize'], visual_type, sequence_name)
        os.makedirs(visualize_root, exist_ok=True)
        visualize_path = os.path.join(visualize_root, image_name)
        image.save(visualize_path)
