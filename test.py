import numpy as np
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
from models.resnet import ResNet18
from utils import *
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Testing",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--exp_name",
    "-en",
    type=str,
    required=True,
)
args = parser.parse_args()

test_bs = 200
num_to_avg = 1
out_as_pos = True
dataset = "cifar10"
model = "wrn"

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
    num_workers=4,
    pin_memory=True,
)

texture_loader = torch.utils.data.DataLoader(
    texture_data, batch_size=test_bs, shuffle=True, pin_memory=False
)
svhn_loader = torch.utils.data.DataLoader(
    svhn_data, batch_size=test_bs, shuffle=True, pin_memory=False
)
places365_loader = torch.utils.data.DataLoader(
    places365_data,
    batch_size=test_bs,
    shuffle=True,
    pin_memory=False,
)
lsunc_loader = torch.utils.data.DataLoader(
    lsunc_data, batch_size=test_bs, shuffle=True, pin_memory=False
)
lsunr_loader = torch.utils.data.DataLoader(
    lsunr_data, batch_size=test_bs, shuffle=True, pin_memory=False
)
isun_loader = torch.utils.data.DataLoader(
    isun_data, batch_size=test_bs, shuffle=True, pin_memory=False
)

ood_num_examples = len(test_data) // 5
print(f"ood num examples is {ood_num_examples}")
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_scores(loader, calc_id_acc=False, in_dist=False):
    _score = []
    acc = []
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            if batch_idx >= ood_num_examples // test_bs and in_dist is False:
                break
            data, targets = data.to(device), targets.to(device)
            _, output = net(data)

            if calc_id_acc:
                # Calculate accuracy
                _, predicted = output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc.append(100.0 * correct / total)

            smax = to_np(F.softmax(output, dim=1))
            # smax = to_np(output)
            max_val = -np.max(smax, axis=1)
            _score.append(max_val)
    if calc_id_acc:
        mean_acc = np.mean(acc)
    ood_score = concat(_score)

    if calc_id_acc:
        return ood_score[:ood_num_examples].copy(), mean_acc
    else:
        return ood_score[:ood_num_examples].copy()


def calc_accuracy(X, true_labels):
    predicted = np.argmax(X, axis=1)
    total = len(true_labels)
    acc = np.sum(predicted == true_labels) / total
    return acc


def get_results(ood_loader, in_score, num_to_avg=num_to_avg):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_scores(ood_loader)
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


# Restore model
if model == "resnet":
    net = ResNet18(num_classes=num_classes)
else:
    net = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.3)

net.to(device)

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
for i in range(5, 6):
    margin = i / 10
    model_path = "checkpoint/{}/{}_{}_{}_ckpt9.pt".format(
        model, dataset, args.exp_name, margin
    )
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()
    in_score, accuracy = get_scores(test_loader, calc_id_acc=True, in_dist=True)
    print(accuracy)
    # id_accuracy = get_id_acc(cifar_loader)
    # print(f"The id accuracy is {id_accuracy} %")
    for ood_name, ood_loader in ood_loaders.items():
        auroc, aupr, fpr = get_results(ood_loader, in_score)
        print(f"Margin {margin} with {ood_name} ood, metrics : {[auroc, aupr, fpr]}")
        metrics.append([margin, ood_name, auroc, aupr, fpr])

output_metrics_dir = os.path.join("./logs", "output_metrics")
if not os.path.exists(output_metrics_dir):
    os.makedirs(output_metrics_dir)

with open(
    "{}/{}_{}_{}.csv".format(output_metrics_dir, model, dataset, args.exp_name), "w"
) as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(["margin", "ood_dataset", "auroc", "aupr", "fpr"])
    csvwriter.writerows(metrics)
