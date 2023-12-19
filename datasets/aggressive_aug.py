from typing import Any
import torch
import torch.nn as nn
from torchvision.datasets import *
import torchvision.transforms as T


class GaussianNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noised_sample = tensor + noise
        return noised_sample


class PerCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, crops):
        return [self.transform(crop) for crop in crops]


class DoTransform:
    def __init__(self):
        self.none = None

    @staticmethod
    def do_transform(dataset):
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        if dataset == "cifar10":
            size = 32
        # Define the transformations
        pipeline = T.Compose(
            [
                T.CenterCrop(size),
                T.FiveCrop(size),  # Size of each crop
                PerCropTransform(
                    T.Compose(
                        [
                            T.ColorJitter(),
                            T.RandomVerticalFlip(),
                            T.ToTensor(),
                            GaussianNoise(0.0, 0.1),
                            T.Normalize(mean, std),
                        ]
                    )
                ),
            ]
        )
        return pipeline
