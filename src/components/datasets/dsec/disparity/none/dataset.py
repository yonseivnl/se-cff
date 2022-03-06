import torch.utils.data


class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'event': 'event',
        'image': 'image'
    }
    _DOMAIN = ['event', 'image']
    NO_VALUE = 0.0

    def __init__(self, root):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, timestamp):
        return None

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch
