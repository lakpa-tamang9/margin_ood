# -*- coding: utf-8 -*-
from torch.autograd import Variable
import numpy as np
import os
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
from utils.randimages import RandomImages
from utils.resized_imagenet_loader import ImageNetDownSample
from models.resnet import ResNet18
from models.allconv import AllConvNet
from models.densenet import DenseNet121
from utils.svhn_loader import SVHN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(
    description="Tunes a CIFAR Classifier with DivOE",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--exp_name", "-en", default="1", type=str)
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
    choices=["allconv", "wrn", "resnet", "densenet"],
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
parser.add_argument("--method", type=str, default="div_oe")

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
parser.add_argument("--oe_batch_size", type=int, default=128, help="Batch size.")
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
    "--save", "-s", type=str, default="./snapshots/", help="Folder to save checkpoints."
)
parser.add_argument(
    "--load",
    "-l",
    type=str,
    default="./snapshots/pretrained",
    help="Checkpoint path to resume / test.",
)
parser.add_argument("--test", "-t", action="store_true", help="Test only flag.")
# Acceleration
parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
parser.add_argument("--prefetch", type=int, default=4, help="Pre-fetching threads.")
# EG specific
parser.add_argument(
    "--m_in",
    type=float,
    default=-25.0,
    help="margin for in-distribution; above this value will be penalized",
)
parser.add_argument(
    "--m_out",
    type=float,
    default=-7.0,
    help="margin for out-distribution; below this value will be penalized",
)
parser.add_argument("--score", type=str, default="MSP", help="MSP|energy")
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed for np(tinyimages80M sampling); 1|2|8|100|107",
)
parser.add_argument(
    "--extrapolation_ratio",
    type=float,
    default=0.5,
    help="the ratio of extrapolated outliers in the whole batch",
)
parser.add_argument("--epsilon", type=float, default=0.01, help="extrapolation epsilon")
parser.add_argument(
    "--rel_step_size",
    type=float,
    default=1 / 4,
    help="extrapolation relative step size",
)
parser.add_argument("--num_steps", type=int, default=5, help="optimization step number")
parser.add_argument(
    "--extrapolation_score",
    type=str,
    default="MSP",
    help="MSP|energy for extrapolation optimization",
)
parser.add_argument(
    "--data_num",
    "-dn",
    type=int,
    default=300000,
    help="Total number of auxiliary images from RandImages",
)

args = parser.parse_args()

print(
    f"This experiment starts from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}"
)

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)


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
    batch_size=args.oe_batch_size,
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


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 0
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


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
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

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


def extrapolate(model, data, epsilon, rel_step_size=1 / 4, num_steps=5, rand_init=True):
    model.eval()
    data = data.cuda()
    x_adv = (
        data.detach()
        + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape))
        .float()
        .cuda()
        if rand_init
        else data.detach()
    )
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        if args.model == "allconv":
            output = model(x_adv)
        else:
            _, output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if args.extrapolation_score == "MSP":
                loss_adv = -(output.mean(1) - torch.logsumexp(output, dim=1)).mean()
                loss_adv = loss_adv.mean()
            elif args.extrapolation_score == "energy":
                Ec_out = -torch.logsumexp(output, dim=1)
                loss_adv = torch.pow(F.relu(args.m_out - Ec_out), 2).mean()
        loss_adv.backward()
        step_size = epsilon * rel_step_size
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    model.train()
    return x_adv


# /////////////// Training ///////////////


def train():
    net.train()  # enter train mode
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):

        inset_tensor = in_set[0].to(device)
        out_set_tensor = out_set[0].to(device)

        if inset_tensor.size()[0] != out_set_tensor.size()[0]:
            if (
                len(inset_tensor) < args.batch_size
                or len(out_set_tensor) < args.batch_size
            ):  # done for imagenet32 as it has 96 size for inset
                continue
            length = min(inset_tensor.size()[0], out_set_tensor.size()[0])
            inset_tensor = inset_tensor[:length]
            out_set_tensor = out_set_tensor[:length]

        aug_length = int(len(out_set[0]) * args.extrapolation_ratio)
        adv_outlier = extrapolate(
            net,
            out_set[0][:aug_length],
            args.epsilon,
            args.rel_step_size,
            args.num_steps,
        )
        data = torch.cat((in_set[0], adv_outlier.cpu(), out_set[0][aug_length:]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        if args.model == "allconv":
            x = net(data)
        else:
            _, x = net(data)

        # backward
        scheduler.step()
        optimizer.zero_grad()

        loss = F.cross_entropy(x[: len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        if args.score == "energy":
            Ec_out = -torch.logsumexp(x[len(in_set[0]) :], dim=1)
            Ec_in = -torch.logsumexp(x[: len(in_set[0])], dim=1)
            loss += 0.1 * (
                torch.pow(F.relu(Ec_in - args.m_in), 2).mean()
                + torch.pow(F.relu(args.m_out - Ec_out), 2).mean()
            )
        elif args.score == "MSP":
            loss += (
                0.5
                * -(
                    x[len(in_set[0]) :].mean(1)
                    - torch.logsumexp(x[len(in_set[0]) :], dim=1)
                ).mean()
            )

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    state["train_loss"] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

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

if args.extrapolation_score == "MSP":
    save_info = f"MSP_DivOE"
elif args.extrapolation_score == "energy":
    save_info = f"energy_DivOE"

if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception("%s is not a dir" % args.save)

if args.method == "macs":
    margins_length = 10
else:
    margins_length = 1

logs_n_ckpt_dir = os.path.join(
    "./icdm/{}/train_logs_and_ckpts_{}".format(args.method, args.outlier_name),
    args.model,
)

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
