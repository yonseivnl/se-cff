import torch
import numpy as np


class ToTensor:
    def __call__(self, sample):
        sample = torch.from_numpy(sample)

        return sample


class Padding:
    def __init__(self, img_height, img_width, no_disparity_value):
        self.img_height = img_height
        self.img_width = img_width
        self.no_disparity_value = no_disparity_value

    def __call__(self, sample):
        ori_height, ori_width = sample.shape[:2]
        top_pad = self.img_height - ori_height
        right_pad = self.img_width - ori_width

        assert top_pad >= 0 and right_pad >= 0

        sample = np.lib.pad(sample,
                            ((top_pad, 0), (0, right_pad)),
                            mode='constant',
                            constant_values=self.no_disparity_value)

        return sample


class Crop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, sample, offset_x, offset_y):
        start_y, end_y = offset_y, offset_y + self.crop_height
        start_x, end_x = offset_x, offset_x + self.crop_width

        sample = sample[start_y:end_y, start_x:end_x]

        return sample


class VerticalFlip:
    def __call__(self, sample):
        sample = np.copy(np.flipud(sample))

        return sample
