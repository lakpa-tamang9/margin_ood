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
from PIL import Image
import logging
import dataset_utils.svhn_loader as svhn
from dataset_utils.resized_imagenet_loader import ImageNetDownSample

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
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Total number of trials to average the result.",
)
parser.add_argument(
    "--detectors",
    type=str,
    default="MSP",
    choices=["msp", "mahalanobis", "maxlogit", "energy"],
    help="Name of detector to use.",
)
parser.add_argument(
    "--save_img",
    type=bool,
    default=False,
    help="Save the input and outlier images.",
)
parser.add_argument(
    "--outlier_name",
    "-on",
    type=str,
    default="300k",
    choices=["300k", "imgnet32", "tinyimagenet"],
    help="Choose the outlier data",
)
parser.add_argument(
    "--plot_tsne",
    type=bool,
    default=False,
    help="Plot the feature representation of the penultimate layer.",
)
parser.add_argument(
    "--num_plot_samples",
    "-nps",
    type=int,
    default=1000,
    help="Total number of samples for T-SNE plot",
)

parser.add_argument(
    "--save_array",
    type=bool,
    default=False,
    help="Save the input and outlier npy arry for plots",
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

# create all OOD datasets
ood_datasets = [
    Textures,
    "svhn",
    "isun",
    LSUNCrop,
    Places365,
]
datasets = {}
for ood_dataset in ood_datasets:
    if ood_dataset == "svhn":
        dataset_out_test = dset.ImageFolder(
            root="data/svhn",
            transform=tvt.Compose(
                [
                    tvt.Resize(32),
                    tvt.CenterCrop(32),
                    tvt.ToTensor(),
                    tvt.Normalize(mean, std),
                ]
            ),
            target_transform=ToUnknown(),
        )
    elif ood_dataset == "isun":
        dataset_out_test = dset.ImageFolder(
            root="data/iSUN", transform=trans, target_transform=ToUnknown()
        )
    elif ood_dataset == "places_365":
        dataset_out_test = dset.ImageFolder(
            root="data/places365_standard",
            transform=tvt.Compose(
                [
                    tvt.Resize(32),
                    tvt.CenterCrop(32),
                    tvt.ToTensor(),
                    tvt.Normalize(mean, std),
                ]
            ),
            target_transform=ToUnknown(),
        )
    else:
        dataset_out_test = ood_dataset(
            root="data", transform=trans, target_transform=ToUnknown(), download=True
        )

    test_loader = DataLoader(
        dataset_in_test + dataset_out_test, batch_size=256, num_workers=12
    )
    if ood_dataset in ["svhn", "isun", "places_365"]:
        datasets[ood_dataset] = test_loader
    else:
        datasets[ood_dataset.__name__] = test_loader

# Create DNN with pre-trained weights from the Hendrycks baseline paper
model = (
    WideResNet(num_classes=num_classes, pretrained="{}-pt".format(args.dataset))
    .eval()
    .to(device)
)

# Create OOD detector
detectors = {}
if args.detectors == "msp":
    detectors["MSP"] = MaxSoftmax(model)
elif args.detectors == "mahalanobis":
    detectors["Mahalanobis"] = Mahalanobis(model.features)
elif args.detectors == "maxlogit":
    detectors["MaxLogit"] = MaxLogit(model)
elif args.detectors == "energy":
    detectors["EnergyBased"] = EnergyBased(model)


if args.outlier_name == "imgnet32":
    outlier_data = ImageNetDownSample(
        root="./data/ImageNet32",
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
elif args.outlier_name == "300k":
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
        )
    )
elif args.outlier_name == "tinyimagenet":
    outlier_data = dset.ImageFolder(
        root="DOE/data/tiny-imagenet-200/train",
        transform=tvt.Compose(
            [
                tvt.Resize(32),
                tvt.RandomCrop(32, padding=4),
                tvt.RandomHorizontalFlip(),
                tvt.ToTensor(),
                tvt.Normalize(mean, std),
            ]
        ),
    )
train_loader_out = DataLoader(
    outlier_data,
    batch_size=128,
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

                r = {
                    "Detector": detector_name,
                    "Dataset": dataset_name,
                }
                r.update(metrics.compute())

                results.append(r)
    return results


def test():
    model.eval()
    loss = 0
    correct = 0
    total = 0
    test_accuracies = []
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
            test_accuracies.append(test_acc)
            progress_bar(
                batch_idx,
                len(val_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (loss, test_acc * 100, correct, total),
            )
    return test_accuracies


def save_fig(name, img, dir_path, count):
    img = unnormalize(img, mean, std)
    img = ((img.numpy().transpose((1, 2, 0))) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save("{}/mlm_img_{}_{}_{}.png".format(dir_path, args.dataset, name, count))


def train():
    global losses_val
    print("\nEpoch: %d" % epoch)
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    original_confidences = []
    mixed_confidences = []

    log_step = 30
    inputs_list = []
    out_set_tensor_list = []
    after_mix_list = []
    train_id_features = []
    train_ood_features = []
    train_id_labels = []
    train_ood_labels = []

    accuracies = []

    losses_val = []
    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader, train_loader_out)):
        inputs = in_set[0].to(device)
        inputs_list.append(inputs.detach().cpu().numpy())
        targets = in_set[1].to(device)
        out_set_tensor = out_set[0].to(device)
        out_set_tensor_list.append(out_set_tensor.detach().cpu().numpy())
        ood_targets = torch.tensor([10] * len(out_set_tensor)).to(device)

        if inputs.size()[0] != out_set_tensor.size()[0]:
            length = min(inputs.size()[0], out_set_tensor.size()[0])
            inputs = inputs[:length]
            out_set_tensor = out_set_tensor[:length]

        mixed_input = torch.cat((inputs, out_set_tensor), 0)
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
        if args.save_img:
            if margin == 0.1:  # only save image for one margin, change as necessary
                if batch_idx % log_step == 0:
                    for i in range(5):
                        inputs_viz = inputs[i].cpu()
                        out_set_tensor_viz = out_set_tensor[i].cpu()
                        mixed_input_viz = mixed_input[i].cpu()

                        # Save all id, ood, and mixed figs
                        save_fig(
                            name="inputs_viz",
                            img=inputs_viz,
                            dir_path=dir_path,
                            count=i,
                        )
                        save_fig(
                            name="out_set_tensor_viz",
                            img=out_set_tensor_viz,
                            dir_path=dir_path,
                            count=i,
                        )
                        save_fig(
                            name="mixed_input_viz",
                            img=mixed_input_viz,
                            dir_path=dir_path,
                            count=i,
                        )

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
        margin_loss = 0.5 * torch.clamp(margin - loss_pre, min=0.0)

        total_loss = loss + uniform_loss + margin_loss
        total_loss.backward()

        optimizer.step()

        losses = total_loss / (batch_idx + 1)
        losses_val.append(losses)

        train_acc = predicted.eq(targets).sum().item() / targets.size(0)
        accuracies.append(train_acc)
        train_loss = total_loss.item()
        progress_bar(
            batch_idx,
            len(train_loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (losses, train_acc * 100, correct, total),
        )
    if args.save_array:
        np.save(
            "logs/numpy_data/{}_input_tensor.npy".format(args.dataset),
            np.array(inputs_list, dtype=object),
            allow_pickle=True,
        )
        np.save(
            "logs/numpy_data/{}_out_set_tensor.npy".format(args.dataset),
            np.array(out_set_tensor_list, dtype=object),
            allow_pickle=True,
        )
        np.save(
            "logs/numpy_data/{}_after_mix_tensor.npy".format(args.dataset),
            np.array(after_mix_list, dtype=object),
            allow_pickle=True,
        )

    if args.plot_tsne:
        return train_id_features, train_ood_features, train_id_labels, train_ood_labels
    else:
        return accuracies


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


perm_train = torch.randperm(train_loader.__len__() + train_loader_out.__len__())
select_train = perm_train[: args.num_plot_samples]
dataset = args.dataset
epochs = 10
margins = [0.3]
for margin in margins:
    for epoch in range(epochs):
        if args.plot_tsne:
            (
                train_id_features,
                train_ood_features,
                train_id_labels,
                train_ood_labels,
            ) = train()
            fea_id, label_id = embedding(
                train_id_features, train_id_labels, select_train
            )
            fea_ood, label_ood = embedding(
                train_id_features, train_id_labels, select_train
            )

            total_features = (fea_id, fea_ood)
            total_labels = (label_id, label_ood)

            plot_features(
                "logs/tsne_plot",
                total_features,
                total_labels,
                10,
                epoch,
                "MaPS_{}_{}_{}_samples/".format(
                    args.dataset, args.detectors, args.num_plot_samples
                ),
            )
        else:
            accuracies = train()

        print(losses_val)

        test_accuracies = test()
        import json

        with open("./results/{}_{}.txt".format(args.dataset, args.detectors), "w") as f:
            json.dump(test_accuracies, f)

        do_eval = epoch == epochs - 1
        trial_results = []
        if do_eval and not args.plot_tsne:
            for i in range(args.num_trials):
                result = evaluate()
                trial_results.append(result)
    logging.info(f"Margin: {margin}")
    if not args.plot_tsne:
        all_auroc = []
        all_aupr_in = []
        all_aupr_out = []
        all_fpr95 = []
        # for results in trial_results:
        dfs = [pd.DataFrame(results) for results in trial_results]
        for i, df in enumerate(dfs):
            logging.basicConfig(filename="logs/results.log", level=logging.INFO)
            logging.info(f"The result for {i} th trial.")
            logging.info(df)
            auroc = df.groupby("Detector")["AUROC"].mean() * 100
            aupr_in = df.groupby("Detector")["AUPR-IN"].mean() * 100
            aupr_out = df.groupby("Detector")["AUPR-OUT"].mean() * 100
            fpr95 = df.groupby("Detector")["FPR95TPR"].mean() * 100

            mean_df = [
                "MSP",
                "Mean.",
                round(auroc.values[0], 2),
                round(aupr_in.values[0], 2),
                round(aupr_out.values[0], 2),
                round(fpr95.values[0], 2),
            ]
            print(df)
            print(mean_df)
            logging.info(f"Mean auroc for {i}th trial: {round(auroc.values[0], 2)}")
            logging.info(f"Mean aupr_in for {i}th trial: {round(aupr_in.values[0], 2)}")
            logging.info(
                f"Mean aupr_out for {i}th trial: {round(aupr_out.values[0], 2)}"
            )
            logging.info(f"Mean fpr95 for {i}th trial: {round(fpr95.values[0], 2)}")

            all_auroc.append(round(auroc.values[0], 2))
            all_aupr_in.append(round(aupr_in.values[0], 2))
            all_aupr_out.append(round(aupr_out.values[0], 2))
            all_fpr95.append(round(fpr95.values[0], 2))

            try:
                df.loc[len(df)] = mean_df
            except Exception as e:
                logging.error(e)
            df.to_csv(
                "results/{}/{}/{}_margin_{}_trial_{}.csv".format(
                    args.outlier_name, args.dataset, args.detectors, margin, i
                )
            )

        logging.info(f"Five trial Average AUROC: {all_auroc[0]} ")
        logging.info(f"Five trial Average AUPR_IN: {all_aupr_in[0]} ")
        logging.info(f"Five trial Average AUPR_OUT: {all_aupr_out[0]} ")
        logging.info(f"Five trial Average FPR95: {all_fpr95[0]} ")
