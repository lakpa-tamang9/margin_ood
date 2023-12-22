import torch
from torchvision.datasets import *
import torchvision.transforms as T
import dataset_utils.svhn_loader as svhn


def load_ood_data(dataset, batch_size, mean, std):
    if dataset == "textures":
        ood_data = ImageFolder(
            root="../data/dtd/images",
            transform=T.Compose(
                [T.Resize(32), T.CenterCrop(32), T.ToTensor(), T.Normalize(mean, std)]
            ),
        )
        num_workers = 4
        print("\n\nTexture Detection")

    elif dataset == "svhn":
        ood_data = svhn.SVHN(
            root="../data/svhn/",
            split="test",
            transform=T.Compose(
                [T.ToTensor(), T.Normalize(mean, std)]
            ),  # T.Resize(32),
            download=False,
        )
        num_workers = 2
        print("\n\nSVHN Detection")

    elif dataset == "places365":
        ood_data = ImageFolder(
            root="../data/places365/",
            transform=T.Compose(
                [T.Resize(32), T.CenterCrop(32), T.ToTensor(), T.Normalize(mean, std)]
            ),
        )
        num_workers = 2
        print("\n\nPlaces365 Detection")

    elif dataset == "lsunc":
        ood_data = ImageFolder(
            root="../data/LSUN_C",
            transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]),
        )
        num_workers = 1
        print("\n\nLSUN_C Detection")

    elif dataset == "lsunr":
        ood_data = ImageFolder(
            root="../data/LSUN_resize",
            transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]),
        )
        num_workers = 1
        print("\n\nLSUN_Resize Detection")

    elif dataset == "isun":
        ood_data = ImageFolder(
            root="../data/iSUN",
            transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]),
        )
        num_workers = 1

    ood_loader = torch.utils.data.DataLoader(
        ood_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ood_loader
