import os
from PIL import Image

import numpy as np

import torch.utils.data


class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'event': 'event',
    }
    _DOMAIN = ['event']
    NO_VALUE = 0.0

    def __init__(self, root):
        self.root = root
        self.timestamps = load_timestamp(os.path.join(root, self._PATH_DICT['timestamp']))

        self.disparity_path_list = {}
        self.timestamp_to_disparity_path = {}
        for domain in self._DOMAIN:
            self.disparity_path_list[domain] = get_path_list(os.path.join(root, self._PATH_DICT[domain]))
            self.timestamp_to_disparity_path[domain] = {timestamp: filepath for timestamp, filepath in
                                                        zip(self.timestamps, self.disparity_path_list[domain])}
        self.timestamp_to_index = {
            timestamp: int(os.path.splitext(os.path.basename(self.timestamp_to_disparity_path['event'][timestamp]))[0])
            for timestamp in self.timestamp_to_disparity_path['event'].keys()
        }

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, timestamp):
        return load_disparity(self.timestamp_to_disparity_path['event'][timestamp])

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch


def load_timestamp(root):
    return np.loadtxt(root, dtype='int64')


def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]


def load_disparity(root):
    disparity = np.array(Image.open(root)).astype(np.float32) / 256.
    return disparity
