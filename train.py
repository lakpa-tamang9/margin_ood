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
from datasets.aggressive_aug import DoTransform
from torchvision import transforms
from datasets.cifar10 import CIFAR10Extended
from loss.loss import MarginLoss
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
np.random.seed(1)

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
train_dataset = CIFAR10Extended(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
    ood_transform=DoTransform.do_transform(dataset="cifar10"),
    m=100,
)
val_dataset = CIFAR10Extended(
    root="./data",
    train=False,
    download=True,
    transform=val_transform,
    ood_transform=DoTransform.do_transform(dataset="cifar10"),
    m=25,
)

# Use DataLoader to iterate over the dataset
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)

# Load model
net = ResNet18()
net = nn.DataParallel(net)
net.to(device)

# Add class_weights
if use_class_weighting:
    weights = [1] * 11
    weights[-1] *= 1.4
    class_weights = torch.tensor(weights)
else:
    class_weights = None

criterion = MarginLoss(class_weights)
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
        epochs * len(trainloader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / lr,
    ),
)

start_epoch = 0
best_acc = 0


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    loss_avg = 0
    correct = 0
    total = 0
    ood_class_idx = 10
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # convert from original dataset label to known class label
        # targets = torch.Tensor([mapping[x] for x in targets]).long().to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets, ood_class_idx)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        loss_avg = loss_avg * 0.8 + float(train_loss) * 0.2

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        total += targets.size(0)
        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(val_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
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
        torch.save(state, "./checkpoint/ckpt.pth")
        best_acc = acc


# Main loop
for epoch in range(start_epoch, 100):
    begin_epoch = time.time()

    train(epoch=epoch)
    test(epoch=epoch)
