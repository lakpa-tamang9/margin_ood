import pandas as pd  # additional dependency, used here for convenience
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from pytorch_ood.dataset.img import *
import torch.nn as nn
from pytorch_ood.detector import *
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed
import os
import csv
import torch.optim as optim
import argparse
import torchvision.transforms as tvt
from pytorch_ood.utils import OODMetrics, ToRGB, ToUnknown
from torchvision.transforms.functional import to_pil_image
import random
import torchvision.datasets as dset
import matplotlib.pyplot as plt
from utils import *
from dataset_utils.randimages import RandomImages
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

fix_random_seed(123)

parser = argparse.ArgumentParser(
    description="Testing",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--exp_name",
    "-en",
    type=str,
    default="test",
    required=False,
)
parser.add_argument(
    "--dataset", "-d", type=str, default="cifar10", choices=["cifar10", "cifar100"]
)
args = parser.parse_args()

# Setup preprocessing
norm_std = WideResNet.norm_std_for("cifar10-pt")

# Setup datasets
if args.dataset == "cifar10":
    trans = WideResNet.transform_for("cifar10-pt")
    dataset_in_test = CIFAR10(root="data", train=False, transform=trans, download=True)
    # fit detectors to training data (some require this, some do not)
    loader_in_train = DataLoader(
        CIFAR10(root="data", train=True, transform=trans),
        batch_size=256,
        num_workers=12,
    )
    num_classes = 10

elif args.dataset == "cifar100":
    trans = WideResNet.transform_for("cifar100-pt")
    dataset_in_test = CIFAR100(root="data", train=False, transform=trans, download=True)
    # fit detectors to training data (some require this, some do not)
    loader_in_train = DataLoader(
        CIFAR100(root="data", train=True, transform=trans),
        batch_size=256,
        num_workers=12,
    )
    num_classes = 100

svhn = dset.ImageFolder(root="data/svhn", transform=trans, target_transform=ToUnknown())
isun = dset.ImageFolder(root="data/iSUN", transform=trans, target_transform=ToUnknown())
# create all OOD datasets
ood_datasets = [
    Textures,
    "svhn",
    "isun",
    LSUNCrop,
    LSUNResize,
    Places365,
]
datasets = {}
for ood_dataset in ood_datasets:
    dataset_out_test = ood_dataset(
        root="data", transform=trans, target_transform=ToUnknown(), download=True
    )
    if ood_dataset == "svhn":
        dataset_out_test = dset.ImageFolder(
            root="data/svhn", transform=trans, target_transform=ToUnknown()
        )
    elif ood_dataset == "isun":
        dataset_out_test = dset.ImageFolder(
            root="data/iSUN", transform=trans, target_transform=ToUnknown()
        )

    test_loader = DataLoader(
        dataset_in_test + dataset_out_test, batch_size=256, num_workers=12
    )
    datasets[ood_dataset.__name__] = test_loader

# Create DNN with pre-trained weights from the Hendrycks baseline paper
model = (
    WideResNet(num_classes=num_classes, pretrained="{}-pt".format(args.dataset))
    .eval()
    .to(device)
)

# Create OOD detector
detectors = {}
# detectors["Entropy"] = Entropy(model)
# detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
# detectors["Mahalanobis+ODIN"] = Mahalanobis(
#     model.features, norm_std=norm_std, eps=0.002
# )
detectors["Mahalanobis"] = Mahalanobis(model.features)
# detectors["KLMatching"] = KLMatching(model)
# detectors["SHE"] = SHE(model.features, model.fc)
# detectors["MSP"] = MaxSoftmax(model)
# detectors["EnergyBased"] = EnergyBased(model)
# detectors["MaxLogit"] = MaxLogit(model)
# detectors["ODIN"] = ODIN(model, norm_std=norm_std, eps=0.002)
# detectors["DICE"] = DICE(
#     model=model.features, w=model.fc.weight, b=model.fc.bias, p=0.65
# )
# detectors["RMD"] = RMD(model.features)

outlier_data = RandomImages(
    transform=tvt.Compose(
        [
            tvt.ToTensor(),
            tvt.ToPILImage(),
            tvt.RandomCrop(32, padding=4),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor(),
            tvt.Normalize(mean, std),
        ]
    ),
)
train_loader_out = DataLoader(
    outlier_data,
    batch_size=256,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
)

for name, detector in detectors.items():
    print(f"--> Fitting {name}")
    detector.fit(loader_in_train, device=device)


def OE_mixup(x_in, x_out, alpha=10.0):
    if x_in.size()[0] != x_out.size()[0]:
        length = min(x_in.size()[0], x_out.size()[0])
        x_in = x_in[:length]
        x_out = x_out[:length]
    lam = np.random.beta(alpha, alpha)
    x_in_m = MixUp(x_in, mix_size=10)
    x_out_m = MixUp(x_out, mix_size=10)
    x_oe = lam * x_in_m + (1 - lam) * x_out_m
    return x_oe


def MixUp(inputs, mix_size):
    batch_size = inputs.size(0)
    index = [torch.randperm(batch_size) for _ in range(mix_size)]

    mixed_input = torch.zeros_like(inputs)
    for i in range(batch_size):
        for j in range(mix_size):
            mixed_input[i] += inputs[index[j][i], :] / mix_size

    return mixed_input


new_trans = tvt.Compose(
    [
        tvt.Resize(size=(32, 32)),
        ToRGB(),
        tvt.ColorJitter(
            brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3
        ),  # Random color jitter
        tvt.RandomAffine((-90, 90), translate=(0.2, 0.2)),
        tvt.ToTensor(),
        tvt.Normalize(std=std, mean=mean),
    ]
)


def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def evaluate():
    with torch.no_grad():
        for detector_name, detector in detectors.items():
            results = []
            print(f"> Evaluating {detector_name}")
            for dataset_name, loader in datasets.items():
                print(f"--> {dataset_name}")
                metrics = OODMetrics()
                for x, y in loader:
                    metrics.update(detector(x.to(device)), y.to(device))

                r = {"Detector": detector_name, "Dataset": dataset_name}
                r.update(metrics.compute())

                results.append(r)
    return results


def test():
    model.eval()
    loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            test_acc = predicted.eq(targets).sum().item() / targets.size(0)
            progress_bar(
                batch_idx,
                len(val_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (loss, test_acc * 100, correct, total),
            )


def train():
    print("\nEpoch: %d" % epoch)
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    original_confidences = []
    mixed_confidences = []

    log_step = 30
    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader, train_loader_out)):
        inputs = in_set[0].to(device)
        targets = in_set[1].to(device)
        out_set_tensor = out_set[0].to(device)
        after_mix = OE_mixup(inputs, out_set_tensor)
        mixed_input = torch.cat((inputs, after_mix), 0)

        # inputs, targets = inputs.to(device), targets.to(device)
        # inputs_mix, targets_mix = inputs_mix.to(device), targets_mix.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        original_prob = torch.softmax(outputs, dim=1).max(1)[0].detach().cpu().numpy()
        original_confidences.append(original_prob)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # mixed_input = MixUp(inputs_mix, mix_size=10)
        for i in range(inputs.size(0)):
            x = random.randint(1, 2)
            if x == 1:
                mixed_input_pil = to_pil_image(unnormalize(mixed_input[i], mean, std))
                mixed_input[i] = new_trans(mixed_input_pil)

        mixed_outputs = model(mixed_input)
        _, mixed_preds = torch.max(mixed_outputs.data, 1)

        mixed_prob = torch.max(torch.softmax(mixed_outputs[0], dim=0)).item()
        mixed_confidences.append(mixed_prob)

        dir_path = "logs/figures/{}".format(args.exp_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if batch_idx % log_step == 0:
            for i in range(12):
                img = mixed_input[i].cpu()
                img = unnormalize(img, mean, std)
                img = img.numpy().transpose((1, 2, 0))
                plt.savefig(f"{dir_path}/mim_img_{i}.png")

        normalized_probs = torch.nn.functional.softmax(mixed_outputs, dim=1)
        max_id, _ = torch.max(normalized_probs[: len(inputs)], dim=1)
        max_ood, _ = torch.max(normalized_probs[len(inputs) :], dim=1)

        batch_size = mixed_input.size(0)
        uniform_labels = (
            torch.ones((batch_size, num_classes), dtype=torch.int64).to(device)
            / num_classes
        )
        uniform_loss = criterion(mixed_outputs, uniform_labels).to(device)

        loss_pre = torch.pow(F.relu(max_id - max_ood), 2).mean()
        margin_loss = torch.clamp(margin - loss_pre, min=0.0)

        total_loss = loss + uniform_loss + margin_loss
        total_loss.backward()

        optimizer.step()

        losses = total_loss / (batch_idx + 1)

        train_acc = predicted.eq(targets).sum().item() / targets.size(0)
        train_loss = total_loss.item()
        progress_bar(
            batch_idx,
            len(train_loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (losses, train_acc * 100, correct, total),
        )


if args.dataset == "cifar10":
    train_data = CIFAR10(root="data", train=True, download=True, transform=trans)
    train_loader = DataLoader(train_data, batch_size=128, num_workers=12)

    val_data = CIFAR10(root="data", train=False, download=True, transform=trans)
    val_loader = DataLoader(val_data, batch_size=128, num_workers=12)

elif args.dataset == "cifar100":
    train_data = CIFAR100(root="data", train=True, download=True, transform=trans)
    train_loader = DataLoader(train_data, batch_size=128, num_workers=12)

    val_data = CIFAR100(root="data", train=False, download=True, transform=trans)
    val_loader = DataLoader(val_data, batch_size=128, num_workers=12)


epochs = 10
for margin in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    for epoch in range(epochs):
        train()
        test()

        do_eval = epoch == epochs - 1
        if do_eval:
            results = evaluate()

    # calculate mean scores over all datasets, use percent
    df = pd.DataFrame(results)
    mean_auroc = df.groupby("Detector")["AUROC"].mean() * 100
    mean_aupr_in = df.groupby("Detector")["AUPR-IN"].mean() * 100
    mean_aupr_out = df.groupby("Detector")["AUPR-OUT"].mean() * 100
    mean_fpr95 = df.groupby("Detector")["FPR95TPR"].mean() * 100

    df.loc[len(df.index)] = [
        "MSP",
        "Mean Values",
        mean_auroc.values[0],
        mean_aupr_in.values[0],
        mean_aupr_out.values[0],
        mean_fpr95.values[0],
    ]
    df.to_csv(
        "logs/pytorch_ood/fine_tuning_results/{}_{}_margin_{}.csv".format(
            args.exp_name, args.dataset, margin
        )
    )
