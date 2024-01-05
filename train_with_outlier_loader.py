# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.resnet import ResNet18
from models.wrn import WideResNet
from datasets.aggressive_aug import DoTransform
from torchvision import transforms
from dataset_utils.resized_imagenet_loader import ImageNetDownSample

# from datasets.cifar10 import CIFAR10Extended
from torchvision.datasets import *
from loss.loss import MarginLoss
from utils import *
import argparse
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
np.random.seed(1)

batch_size = 128
oe_batch_size = 256

parser = argparse.ArgumentParser(description="Margin OOD")
parser.add_argument(
    "--exp_name",
    "-en",
    type=str,
    default=None,
    help="Folder to save checkpoints.",
)

args = parser.parse_args()

std = [x / 255 for x in [63.0, 62.1, 66.7]]
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
use_class_weighting = False
# Define your train and val transformations
train_transform = trn.Compose(
    [
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(32, padding=4),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ]
)
val_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Create an instance of your extended dataset
train_data_in = dset.CIFAR10(root="./data", train=True, transform=train_transform)
test_data = dset.CIFAR10(root="./data", train=False, transform=val_transform)

# Use DataLoader to iterate over the dataset
ood_data = ImageNetDownSample(
    root="./data/ImageNet32",
    transform=trn.Compose(
        [
            trn.ToTensor(),
            trn.ToPILImage(),
            trn.RandomCrop(32, padding=4),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ]
    ),
)

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=oe_batch_size,
    shuffle=False,
    pin_memory=True,
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
)

# Load model
net = ResNet18()
# net = WideResNet(depth=40, num_classes=10, widen_factor=2)
net = nn.DataParallel(net)
net.to(device)

# Add class_weights
if use_class_weighting:
    weights = [1] * 11
    weights[-1] *= 1.4
    class_weights = torch.tensor(weights)
else:
    class_weights = None

# criterion = MarginLoss(class_weights, margin)
optimizer = torch.optim.SGD(
    net.parameters(),
    0.001,
    momentum=0.9,
    weight_decay=5e-6,
)
epochs = 100
lr = 0.001


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / lr,
    ),
)

start_epoch = 0
best_acc = 0
ood_class_idx = 10


def OE_mixup(x_in, x_out, alpha=10.0):
    if x_in.size()[0] != x_out.size()[0]:
        length = min(x_in.size()[0], x_out.size()[0])
        x_in = x_in[:length]
        x_out = x_out[:length]
    lam = np.random.beta(alpha, alpha)
    x_oe = lam * x_in + (1 - lam) * x_out
    return x_oe


num_classes = 10

state = {}


# Training
def train(epoch):
    global ood_class_idx
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    loss_avg = 0
    correct = 0
    total = 0

    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))

    for batch_idx, (in_set, out_set) in enumerate(
        zip(train_loader_in, train_loader_out)
    ):
        in_oe = OE_mixup(in_set[0], out_set[0])
        data = torch.cat((in_set[0], in_oe), 0)
        targets = in_set[1]
        target_oe = torch.LongTensor(in_oe.shape[0]).random_(
            num_classes, num_classes + 1
        )
        # print(target_oe.shape)

        inputs, targets, target_oe = (
            data.to(device),
            targets.to(device),
            target_oe.to(device),
        )

        optimizer.zero_grad()
        outputs = net(inputs)

        # backprop
        optimizer.zero_grad()

        # loss
        loss = criterion(outputs, targets, target_oe)

        loss.backward()
        optimizer.step()

        scheduler.step()

        train_loss += loss.item()
        # loss_avg = loss_avg * 0.8 + float(train_loss) * 0.2
        # print(train_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted[: len(in_set[0])].eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        losses = train_loss / (batch_idx + 1)
        progress_bar(
            batch_idx,
            len(train_loader_in),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (losses, train_acc, correct, total),
        )
    return losses, train_acc


def test(epoch):
    global best_acc
    global margin
    net.eval()
    test_loss = 0
    loss_avg = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # loss = criterion(outputs, targets, ood_class_idx, test=True)
            loss = F.cross_entropy(outputs, targets)

            predicted = outputs.data.max(1)[1]
            correct += predicted.eq(targets.data).sum().item()

            loss_avg += float(loss.data)

            total += targets.size(0)
            losses = loss_avg / (batch_idx + 1)
            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    losses,
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/{}_margin_{}.pt".format(args.exp_name, margin))
        best_acc = acc
    return losses, acc


# Main loop
# for margin in [0.1, 0.2, 0.3, 0.4, 0.5]:
for margin in [0.3]:
    metrics = []
    for epoch in range(start_epoch, epochs):
        begin_epoch = time.time()
        criterion = MarginLoss(weights=class_weights, margin=margin)
        print(f"Training with margin {margin}")
        train_loss, train_acc = train(epoch=epoch)
        test_loss, test_acc = test(epoch=epoch)
        metrics.append([train_loss, test_loss, train_acc, test_acc])

    with open("logs/{}_margin_{}.csv".format(args.exp_name, margin), "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["train_loss", "test_loss", "train_acc", "test_acc"])
        csvwriter.writerows(metrics)
