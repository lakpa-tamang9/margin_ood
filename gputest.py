import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from loss.loss import MarginLoss
from dataset_utils.validation_dataset import validation_split
from dataset_utils.randimages import RandomImages
from models.resnet import ResNet18
from utils import *
from models.wrn import WideResNet
from dataset_utils.resized_imagenet_loader import ImageNetDownSample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Tunes a CIFAR Classifier with OE",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="Choose between CIFAR-10, CIFAR-100.",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="wrn",
    choices=["resnet", "wrn"],
    help="Choose architecture.",
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
    "--data_num",
    "-dn",
    type=int,
    default=300000,
    help="Total number of auxiliary images from RandImages",
)
parser.add_argument(
    "--calibration",
    "-c",
    default=False,
    help="Train a model to be used for calibration. This holds out some data for validation.",
)
# Optimization options
parser.add_argument(
    "--epochs", "-e", type=int, default=10, help="Number of epochs to train."
)
parser.add_argument(
    "--learning_rate",
    "-lr",
    type=float,
    default=0.001,
    help="The initial learning rate.",
)
parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size.")
parser.add_argument("--oe_batch_size", type=int, default=256, help="Batch size.")
parser.add_argument("--test_bs", type=int, default=200)
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
parser.add_argument(
    "--decay", "-d", type=float, default=0.0005, help="Weight decay (L2 penalty)."
)
# WRN Architecture
parser.add_argument("--layers", default=40, type=int, help="total number of layers")
parser.add_argument("--widen-factor", default=2, type=int, help="widen factor")
parser.add_argument("--droprate", default=0.3, type=float, help="dropout probability")
# Checkpoints
parser.add_argument(
    "--save",
    "-s",
    type=str,
    default="./logs/resnet",
    help="Folder to save checkpoints.",
)
parser.add_argument(
    "--load",
    "-l",
    type=str,
    default="./snapshots/baseline",
    help="Checkpoint path to resume / test.",
)
parser.add_argument("--exp_name", "-en", default="test", type=str)
parser.add_argument("--test", "-t", action="store_true", help="Test only flag.")
# Acceleration
parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
parser.add_argument("--prefetch", type=int, default=4, help="Pre-fetching threads.")
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose(
    [
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(32, padding=4),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ]
)
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == "cifar10":
    train_data_in = dset.CIFAR10("./data", train=True, transform=train_transform)
    test_data = dset.CIFAR10("./data", train=False, transform=test_transform)
    num_classes = 10
elif args.dataset == "cifar100":
    train_data_in = dset.CIFAR100("./data", train=True, transform=train_transform)
    test_data = dset.CIFAR100("./data", train=False, transform=test_transform)
    num_classes = 100
elif args.dataset == "tinyimagenet":
    train_transform = trn.Compose(
        [
            trn.RandomHorizontalFlip(),
            trn.RandomCrop(64, padding=8),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ]
    )
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    train_data_in = dset.ImageFolder(
        root="/home/s223127906/deakin_devs/margin_ood/DOE/data/tiny-imagenet-200/train",
        transform=train_transform,
    )
    test_data = dset.ImageFolder(
        root="/home/s223127906/deakin_devs/margin_ood/DOE/data/tiny-imagenet-200/val",
        transform=test_transform,
    )
    num_classes = 200


calib_indicator = ""
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = "_calib"

if args.outlier_name == "imgnet32":
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
elif args.outlier_name == "300k":
    ood_data = RandomImages(
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
        data_num=args.data_num,
    )
elif args.outlier_name == "tinyimagenet":
    ood_data = dset.ImageFolder(
        root="DOE/data/tiny-imagenet-200/train",
        transform=trn.Compose(
            [
                trn.Resize(32),
                trn.RandomCrop(32, padding=4),
                trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                trn.Normalize(mean, std),
            ]
        ),
    )

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.prefetch,
    pin_memory=True,
)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.prefetch,
    pin_memory=True,
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.prefetch,
    pin_memory=True,
)

# Create model
if args.model == "resnet":
    net = ResNet18(num_classes=num_classes)
else:
    net = WideResNet(
        args.layers, num_classes, args.widen_factor, dropRate=args.droprate
    )

# Restore model
model_found = False
if args.load != "":
    model_name = os.path.join(
        args.load,
        args.dataset
        + calib_indicator
        + "_"
        + args.model
        + "_baseline_epoch_"
        + "99"
        + ".pt",
    )
    print(model_name)
    if os.path.isfile(model_name):
        net.load_state_dict(torch.load(model_name))
        print(f"Model restored! Epoch: 99, Model name: {model_name}")
        model_found = True
    if not model_found:
        assert False, "could not find model to restore"


if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.to(device)

optimizer = torch.optim.SGD(
    net.parameters(),
    state["learning_rate"],
    momentum=state["momentum"],
    weight_decay=state["decay"],
    nesterov=True,
)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate,
    ),
)

print("fine untill now")
