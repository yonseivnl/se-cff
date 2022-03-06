import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    def __init__(self, event_module, disparity_module):
        self.event_transform = event_module.transforms.ToTensor()
        self.disparity_transform = disparity_module.transforms.ToTensor()

    def __call__(self, sample):
        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'])

        if 'disparity' in sample.keys():
            sample['disparity'] = self.disparity_transform(sample['disparity'])

        return sample


class RandomCrop:
    def __init__(self, event_module, disparity_module, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.event_transform = event_module.transforms.Crop(crop_height, crop_width)
        self.disparity_transform = disparity_module.transforms.Crop(crop_height, crop_width)

    def __call__(self, sample):
        if 'event' in sample:
            ori_height, ori_width = sample['event']['left'].shape[:2]
        else:
            raise NotImplementedError

        assert self.crop_height <= ori_height and self.crop_width <= ori_width

        offset_x = np.random.randint(ori_width - self.crop_width + 1)
        offset_y = np.random.randint(ori_height - self.crop_height + 1)

        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'], offset_x, offset_y)

        if 'disparity' in sample.keys():
            sample['disparity'] = self.disparity_transform(sample['disparity'], offset_x, offset_y)

        return sample


class Padding:
    def __init__(self, event_module, disparity_module,
                 img_height, img_width, no_event_value=0, no_disparity_value=0):
        self.img_height = img_height
        self.img_width = img_width
        self.event_transform = event_module.transforms.Padding(img_height, img_width, no_event_value)
        self.disparity_transform = disparity_module.transforms.Padding(img_height, img_width, no_disparity_value)

    def __call__(self, sample):
        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'])

        if 'disparity' in sample.keys():
            sample['disparity'] = self.disparity_transform(sample['disparity'])

        return sample


class RandomVerticalFlip:
    def __init__(self, event_module, disparity_module):
        self.event_transform = event_module.transforms.VerticalFlip()
        self.disparity_transform = disparity_module.transforms.VerticalFlip()

    def __call__(self, sample):
        if np.random.random() < 0.5:
            if 'event' in sample.keys():
                sample['event'] = self.event_transform(sample['event'])

            if 'disparity' in sample.keys():
                sample['disparity'] = self.disparity_transform(sample['disparity'])

        return sample
