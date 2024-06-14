import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from utils.validation_dataset import validation_split
from utils.randimages import RandomImages
from models.resnet import ResNet18
from utils.utils import *
from models.wrn import WideResNet
from models.allconv import AllConvNet
from utils.resized_imagenet_loader import ImageNetDownSample
from datasets import load_dataset
from models.densenet import DenseNet121
from utils.svhn_loader import SVHN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Tunes a CIFAR Classifier with OE",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="wrn",
    choices=["resnet", "wrn", "allconv", "densenet"],
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

parser.add_argument("--mix_op", type=str, default="mixup", choices=["mixup", "cutmix"])
parser.add_argument(
    "--alpha", type=float, default=1.0, help="Parameter for Beta distribution."
)
parser.add_argument(
    "--beta", type=float, default=1.0, help="Weighting factor for the OE objective."
)
parser.add_argument(
    "--method", type=str, default="mix_oe", choices=["oe", "macs", "div_oe", "mix_oe"]
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
    "--lambda_o", type=float, default=1, help="[0.1, 0.5, 1.0, 1.5, 2] dnl loss weight"
)
parser.add_argument(
    "--load",
    "-l",
    type=str,
    default="./snapshots/baseline",
    help="Checkpoint path to resume / test.",
)
parser.add_argument("--exp_name", "-en", default="1", type=str)
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

elif args.dataset == "svhn":
    train_data_in = SVHN(
        root="data/svhn",
        split="train",
        transform=trn.ToTensor(),
        download=False,
    )
    test_data = SVHN(
        root="data/svhn",
        split="test",
        transform=trn.ToTensor(),
        download=False,
    )
    num_classes = 10
elif args.dataset == "imgnet32":
    train_data_in = ImageNetDownSample(
        root="./data/ImageNet32",
        train=True,
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
    test_data = ImageNetDownSample(
        root="./data/ImageNet32",
        train=False,
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
    num_classes = 1000


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
elif args.model == "allconv":
    net = AllConvNet(num_classes=num_classes)
elif args.model == "densenet":
    net = DenseNet121(num_classes=num_classes)
else:
    net = WideResNet(
        args.layers, num_classes, args.widen_factor, dropRate=args.droprate
    )

# Restore model
model_found = False
if args.load != "":
    model_name = "snapshots/baseline/{}_{}_baseline_epoch_99.pt".format(
        args.dataset, args.model
    )
    print(model_name)
    if os.path.isfile(model_name):
        net.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
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


class SoftCE(nn.Module):
    def __init__(self, reduction="mean"):
        super(SoftCE, self).__init__()
        self.reduction = reduction

    def forward(self, logits, soft_targets):
        preds = logits.log_softmax(dim=-1)
        assert preds.shape == soft_targets.shape

        loss = torch.sum(-soft_targets * preds, dim=-1)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(
                "Reduction type '{:s}' is not supported!".format(self.reduction)
            )


soft_xent = SoftCE()


def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    mean_diffs = []
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for batch_idx, (in_set, out_set) in enumerate(
        zip(train_loader_in, train_loader_out)
    ):
        inset_tensor = in_set[0].to(device)
        out_set_tensor = out_set[0].to(device)
        targets = in_set[1].to(device)

        if inset_tensor.size()[0] != out_set_tensor.size()[0]:
            if (
                len(inset_tensor) < args.batch_size
                or len(out_set_tensor) < args.batch_size
            ):  # done for imagenet32 as it has 96 size for inset
                continue
            length = min(inset_tensor.size()[0], out_set_tensor.size()[0])
            inset_tensor = inset_tensor[:length]
            out_set_tensor = out_set_tensor[:length]

        x, y, oe_x = inset_tensor, targets, out_set_tensor
        bs = x.size(0)

        y = y.long()

        one_hot_y = torch.zeros(bs, num_classes).cuda()
        one_hot_y.scatter_(1, y.view(-1, 1), 1)

        if args.model == "allconv":
            logits = net(x)
        else:
            _, logits = net(x)

        # ID loss
        id_loss = F.cross_entropy(logits, y)

        lam = np.random.beta(args.alpha, args.alpha)
        mixed_x = lam * x + (1 - lam) * oe_x

        # construct soft labels and compute loss
        oe_y = torch.ones(oe_x.size(0), num_classes).cuda() / num_classes
        soft_labels = lam * one_hot_y + (1 - lam) * oe_y

        if args.model == "allconv":
            mixed_out = net(mixed_x)
        else:
            _, mixed_out = net(mixed_x)
        mixed_loss = soft_xent(mixed_out, soft_labels)

        # Total loss
        loss = id_loss + args.beta * mixed_loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        scheduler.step()
        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state["train_loss"] = loss_avg
    print(f"Diff betwen ID and OOD scores: {round(np.mean(mean_diffs), 3)}")


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # forward
            if args.model == "allconv":
                output = net(data)
            else:
                _, output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state["test_loss"] = loss_avg / len(test_loader)
    state["test_accuracy"] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()


logs_n_ckpt_dir = os.path.join(
    "./icdm/{}/train_logs_and_ckpts_{}".format(args.method, args.outlier_name),
    args.model,
)

# Create directories for logging metrics and saving trained model

if not os.path.exists(logs_n_ckpt_dir):
    os.makedirs(logs_n_ckpt_dir)

if not os.path.isdir(logs_n_ckpt_dir):
    raise Exception("%s is not a dir" % logs_n_ckpt_dir)


print("Beginning Training\n")

if args.method == "macs":
    margins_length = 10
else:
    margins_length = 1

# Main loop
for i in range(margins_length):
    margin = i / 10
    print("*****************\n")
    print(f"Training with margin = {margin}")
    with open(
        os.path.join(
            logs_n_ckpt_dir,
            args.dataset + "_" + args.exp_name + f"_{margin}_" + "training_results.csv",
        ),
        "w",
    ) as f:
        f.write("epoch,time(s),train_loss,test_loss,test_error(%)\n")
    metrics = []
    for epoch in range(0, args.epochs):
        state["epoch"] = epoch
        begin_epoch = time.time()

        train()
        test()

        # Save model
        if epoch == args.epochs - 1:
            torch.save(
                net.state_dict(),
                os.path.join(
                    logs_n_ckpt_dir,
                    args.dataset
                    + "_"
                    + args.exp_name
                    + f"_{margin}_"
                    + "ckpt"
                    + str(epoch)
                    + ".pt",
                ),
            )

        # Show results

        with open(
            os.path.join(
                logs_n_ckpt_dir,
                args.dataset
                + "_"
                + args.exp_name
                + f"_{margin}_"
                + "training_results.csv",
            ),
            "a",
        ) as f:
            f.write(
                "%03d,%05d,%0.6f,%0.5f,%0.2f\n"
                % (
                    (epoch + 1),
                    time.time() - begin_epoch,
                    state["train_loss"],
                    state["test_loss"],
                    100 - 100.0 * state["test_accuracy"],
                )
            )

        print(
            "Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}".format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                state["train_loss"],
                state["test_loss"],
                100 - 100.0 * state["test_accuracy"],
            )
        )
