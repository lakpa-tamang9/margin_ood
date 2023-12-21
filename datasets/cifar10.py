import torch
from torchvision.datasets import CIFAR10
import numpy as np
from PIL import Image
from datasets.aggressive_aug import DoTransform
import torchvision.transforms as T
import logging


class CIFAR10Extended(CIFAR10):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        m=100,
        ood_transform=None,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.ood_transform = ood_transform

        # CIFAR-10 class names with 'ood' as the 11th class
        self.classes = self.classes + ["ood"]

        # Create the new class data
        ood_data, ood_labels = self.create_ood_class(m)
        ood_data = ood_data.transpose(0, 2, 3, 1)
        self.data = np.concatenate((self.data, ood_data), axis=0)
        self.targets.extend(ood_labels)
        print(len(self.data))
        print(len(self.targets))

    def create_ood_class(self, m):
        ood_data = []
        ood_labels = []

        # Sample m images from each class and apply ood_transform
        for class_idx in range(len(self.classes) - 1):
            class_data = self.data[np.array(self.targets) == class_idx]
            sampled_data = class_data[
                np.random.choice(len(class_data), m, replace=False)
            ]

            for img in sampled_data:
                transformed_img = self.ood_transform(Image.fromarray(img))
                transformed_labels = [10] * 5
                try:
                    assert isinstance(
                        transformed_img, list
                    )  # The transformed image must be a list
                except Exception as e:
                    logging.info(e)

                ood_data.extend(transformed_img)
                ood_labels.extend(transformed_labels)  # Label for the 'ood' class

        return np.array(ood_data), ood_labels

    def __getitem__(self, index):
        is_ood_class = index >= len(self.data)

        if is_ood_class:
            print("Index exceeded 50000")
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform is not None and index < len(self.data):
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return super().__len__()
