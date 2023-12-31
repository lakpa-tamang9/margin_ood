import numpy as np
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

from models.resnet import ResNet18
from utils import *
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_bs = 128
num_to_avg = 1
out_as_pos = True
dataset = "cifar10"

std = [x / 255 for x in [63.0, 62.1, 66.7]]
mean = [x / 255 for x in [125.3, 123.0, 113.9]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

# Load cifar data
if dataset == "cifar10":
    test_data = dset.CIFAR10("data", train=False, transform=test_transform)
    num_classes = 10
else:
    test_data = dset.CIFAR100("data", train=False, transform=test_transform)
    num_classes = 100


texture_data = dset.ImageFolder(
    root="data/textures/images",
    transform=trn.Compose(
        [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
    ),
)
svhn_data = dset.ImageFolder(
    root="data/svhn",
    transform=trn.Compose(
        [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
    ),
)
places365_data = dset.ImageFolder(
    root="data/places365_standard",
    transform=trn.Compose(
        [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
    ),
)
lsunc_data = dset.ImageFolder(
    root="data/LSUN",
    transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
)
lsunr_data = dset.ImageFolder(
    root="data/LSUN_resize",
    transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
)
isun_data = dset.ImageFolder(
    root="data/iSUN",
    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]),
)

# Data Loaders
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=test_bs,
    shuffle=False,
    pin_memory=False,
)

texture_loader = torch.utils.data.DataLoader(
    texture_data, batch_size=test_bs, shuffle=True, num_workers=4, pin_memory=False
)
svhn_loader = torch.utils.data.DataLoader(
    svhn_data, batch_size=test_bs, shuffle=True, num_workers=4, pin_memory=False
)
places365_loader = torch.utils.data.DataLoader(
    places365_data,
    batch_size=test_bs,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
)
lsunc_loader = torch.utils.data.DataLoader(
    lsunc_data, batch_size=test_bs, shuffle=True, num_workers=4, pin_memory=False
)
lsunr_loader = torch.utils.data.DataLoader(
    lsunr_data, batch_size=test_bs, shuffle=True, num_workers=4, pin_memory=False
)
isun_loader = torch.utils.data.DataLoader(
    isun_data, batch_size=test_bs, shuffle=True, num_workers=4, pin_memory=False
)
cifar_loader = torch.utils.data.DataLoader(
    test_data, batch_size=test_bs, shuffle=True, num_workers=4, pin_memory=False
)

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // test_bs and in_dist is False:
                break
            data, target = data.to(device), target.to(device)
            output = net(data)
            # smax = to_np(F.softmax(output, dim=1))
            smax = to_np(output)
            _score.append(-np.max(smax, axis=1))
    if in_dist:
        return concat(
            _score
        ).copy()  # , concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def calc_accuracy(X, true_labels):
    predicted = np.argmax(X, axis=1)
    total = len(true_labels)
    acc = np.sum(predicted == true_labels) / total
    return acc


def get_results(ood_loader, in_score, num_to_avg=num_to_avg):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        if out_as_pos:  # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0])
        auprs.append(measures[1])
        fprs.append(measures[2])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    return auroc, aupr, fpr


def get_id_acc(loader):
    net.eval()
    X = []
    y = []
    all_accuracy = []
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            X += outputs.cpu().detach().tolist()
            y += targets.cpu().tolist()
        X = np.asarray(X)
        y = np.asarray(y)

        accuracy = calc_accuracy(X, y)
        all_accuracy += [accuracy]
    mean_acc = np.mean(all_accuracy)
    return mean_acc


# Restore model
net = ResNet18()
learning_model = "resnet18"
# OOD loaders for test ood datasets
ood_loaders = {
    "lsunc": lsunc_loader,
    "lsunr": lsunr_loader,
    "svhn": svhn_loader,
    "isun": isun_loader,
    "textures": texture_loader,
    "places_365": places365_loader,
}
metrics = []
for i in range(6):
    margin = i / 10
    model_path = (
        f"checkpoint/{dataset}/{learning_model}/32imgnet_oe_margin_{margin}.pth"
    )

    net.load_state_dict(
        torch.load(model_path, map_location=torch.device(device)), strict=False
    )
    net.to(device)
    net.eval()
    in_score = get_ood_scores(test_loader, in_dist=True)

    id_accuracy = get_id_acc(cifar_loader)
    print(f"The id accuracy is {id_accuracy} %")
    for ood_name, ood_loader in ood_loaders.items():
        auroc, aupr, fpr = get_results(ood_loader, in_score)
        print(f"Margin {margin} with {ood_name} ood, results : {[auroc, aupr, fpr]}")
        metrics.append([margin, ood_name, auroc, aupr, fpr])

with open("logs/results/{}_{}_result.csv".format(dataset, learning_model), "w") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(["margin", "ood_dataset", "auroc", "aupr", "fpr"])
    csvwriter.writerows(metrics)
