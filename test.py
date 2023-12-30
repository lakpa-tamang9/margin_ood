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


print("Beginning Training\n")
net = ResNet18()

test_bs = 128
num_to_avg = 1
out_as_pos = True
dataset = "cifar10"

# Restore model
model_path = "./ckpt/cifar10_wrn_pretrained_epoch_99.pt"
std = [x / 255 for x in [63.0, 62.1, 66.7]]
mean = [x / 255 for x in [125.3, 123.0, 113.9]]

net.load_state_dict(torch.load(model_path))

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

# Load cifar data
if dataset == "cifar10":
    test_data = dset.CIFAR10("../data/cifarpy", train=False, transform=test_transform)
    num_classes = 10
else:
    test_data = dset.CIFAR100("../data/cifarpy", train=False, transform=test_transform)
    num_classes = 100


texture_data = dset.ImageFolder(
    root="../data/dtd/images",
    transform=trn.Compose(
        [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
    ),
)
places365_data = dset.ImageFolder(
    root="../data/places365_standard/",
    transform=trn.Compose(
        [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
    ),
)
lsunc_data = dset.ImageFolder(
    root="../data/LSUN",
    transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
)
lsunr_data = dset.ImageFolder(
    root="../data/LSUN_resize",
    transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
)
isun_data = dset.ImageFolder(
    root="../data/iSUN",
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
            data, target = data.cuda(), target.cuda()
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


def get_and_print_results(ood_loader, in_score, num_to_avg=num_to_avg):
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
    print_measures(auroc, aupr, fpr, "")
    return fpr, auroc, aupr


net.eval()
in_score = get_ood_scores(test_loader, in_dist=True)
metric_ll = []
print("lsun")
metric_ll.append(get_and_print_results(lsunc_loader, in_score))
print("isun")
metric_ll.append(get_and_print_results(isun_loader, in_score))
print("texture")
metric_ll.append(get_and_print_results(texture_loader, in_score))
print("places")
metric_ll.append(get_and_print_results(places365_loader, in_score))
print("total")
print("& %.2f & %.2f & %.2f" % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))
