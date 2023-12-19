import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np
from PIL import Image
from aggressive_aug import DoTransform


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
        self.ood_data, self.ood_labels = self.create_ood_class(m)

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
                ood_data.append(np.array(transformed_img))
                ood_labels.append(10)  # Label for the 'ood' class

        return np.array(ood_data), ood_labels

    def __getitem__(self, index):
        if index < len(self.data):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
        else:
            img, target = (
                self.ood_data[index - len(self.data)],
                self.ood_labels[index - len(self.data)],
            )

        if self.transform is not None and index < len(self.data):
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Return image, numeric label, and class name
        class_name = self.classes[target]
        return img, target, class_name

    def __len__(self):
        return super().__len__() + len(self.ood_data)


# Define your original and new transformations
id_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

ood_transform = DoTransform.do_transform(dataset="cifar10")

# Create an instance of your extended dataset
extended_dataset = CIFAR10Extended(
    root="./data",
    train=True,
    download=True,
    transform=id_transform,
    ood_transform=ood_transform,
    m=100,
)

# Use DataLoader to iterate over the dataset
dataloader = torch.utils.data.DataLoader(extended_dataset, batch_size=64, shuffle=True)
