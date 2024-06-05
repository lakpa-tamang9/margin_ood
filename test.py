import numpy as np
import argparse
import torch
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
from models.resnet import ResNet18
from models.allconv import AllConvNet
import statistics as stat
from utils.utils import *
import csv
from utils.svhn_loader import SVHN
from models.densenet import DenseNet121
from utils.resized_imagenet_loader import ImageNetDownSample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Testing",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--exp_name",
    "-en",
    type=str,
    default="1",
)
parser.add_argument(
    "--method",
    type=str,
    default="oe",
    choices=["oe", "macs", "div_oe", "mix_oe", "energy"],
)

parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    default="cifar100",
    choices=["cifar10", "cifar100", "svhn", "imgnet32"],
)
parser.add_argument(
    "--detector", type=str, default="msp", choices=["msp", "xent", "mls"]
)
parser.add_argument(
    "--temp", type=int, default=1, help="Temperature value to scale the output."
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="wrn",
    choices=["resnet", "wrn", "allconv", "densenet"],
    help="Choose architecture.",
)
# WRN Architecture
parser.add_argument("--layers", default=40, type=int, help="total number of layers")
parser.add_argument("--widen_factor", default=2, type=int, help="widen factor")
parser.add_argument("--droprate", default=0.3, type=float, help="dropout probability")
parser.add_argument(
    "--outlier_name",
    "-on",
    type=str,
    default="300k",
    choices=["300k", "imgnet32", "tinyimagenet"],
    help="Choose the outlier data",
)
args = parser.parse_args()

test_bs = 200
num_to_avg = 1
out_as_pos = True

std = [x / 255 for x in [63.0, 62.1, 66.7]]
mean = [x / 255 for x in [125.3, 123.0, 113.9]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

# Load cifar data
if args.dataset == "cifar10":
    test_data = dset.CIFAR10("data", train=False, transform=test_transform)
    num_classes = 10
elif args.dataset == "cifar100":
    test_data = dset.CIFAR100("data", train=False, transform=test_transform)
    num_classes = 100
elif args.dataset == "tinyimagenet":
    test_data = dset.ImageFolder(
        root="data/tiny-imagenet-200/test",
        transform=test_transform,
    )
    num_classes = 200
elif args.dataset == "svhn":
    test_data = SVHN(
        root="data/svhn",
        split="test",
        transform=trn.ToTensor(),
        download=False,
    )
    num_classes = 10
elif args.dataset == "imgnet32":
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

texture_data = dset.ImageFolder(
    root="data/textures/images",
    transform=trn.Compose(
        [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
    ),
)
# Cifar10 and 100 data
cifar10_data = dset.CIFAR10("data", train=False, transform=trn.ToTensor())
cifar100_data = dset.CIFAR100("data", train=False, transform=trn.ToTensor())

svhn_data = SVHN(
    root="data/svhn",
    split="test",
    transform=trn.ToTensor(),
    download=False,
)

places365_data = dset.ImageFolder(
    root="data/places365",
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
cifar10_loader = torch.utils.data.DataLoader(
    cifar10_data, batch_size=test_bs, num_workers=8, shuffle=True, pin_memory=False
)
cifar100_loader = torch.utils.data.DataLoader(
    cifar100_data, batch_size=test_bs, num_workers=8, shuffle=True, pin_memory=False
)
texture_loader = torch.utils.data.DataLoader(
    texture_data, batch_size=test_bs, num_workers=8, shuffle=True, pin_memory=False
)
svhn_loader = torch.utils.data.DataLoader(
    svhn_data, batch_size=test_bs, num_workers=8, shuffle=True, pin_memory=False
)
places365_loader = torch.utils.data.DataLoader(
    places365_data,
    batch_size=test_bs,
    num_workers=8,
    shuffle=True,
    pin_memory=False,
)
lsunc_loader = torch.utils.data.DataLoader(
    lsunc_data, batch_size=test_bs, num_workers=8, shuffle=True, pin_memory=False
)
# lsunr_loader = torch.utils.data.DataLoader(
#     lsunr_data, batch_size=test_bs, shuffle=True, pin_memory=False
# )
isun_loader = torch.utils.data.DataLoader(
    isun_data, batch_size=test_bs, num_workers=8, shuffle=True, pin_memory=False
)

ood_num_examples = len(test_data) // 5
print(f"ood num examples is {ood_num_examples}")
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

from scipy.linalg import inv


def compute_mean_and_cov(feature_vectors):
    mean_vector = np.mean(feature_vectors, axis=0)
    cov_matrix = np.cov(feature_vectors, rowvar=False)
    return mean_vector, cov_matrix


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
            if type(targets) is int:
                continue
            data, targets = data.to(device), targets.to(device)
            if args.model == "allconv":
                output = net(data)
            else:
                feat, output = net(data)
            # print(feat.shape)

            if calc_id_acc:
                # Calculate accuracy
                _, predicted = output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc.append(100.0 * correct / total)

            if args.detector == "msp":
                output = output / args.temp
                smax = to_np(F.softmax(output, dim=1))
                # smax = to_np(output)
                max_val = -np.max(smax, axis=1)
                # max_val = np.max(smax, axis=1)
                _score.append(max_val)

            elif args.detector == "xent":
                output = output / args.temp
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))

            elif args.detector == "mls":
                output = output / args.temp
                smax = to_np(output)
                max_val = -np.max(smax, axis=1)
                _score.append(max_val)
    if calc_id_acc:
        mean_acc = np.mean(acc)
    ood_score = concat(_score)

    if calc_id_acc:
        return ood_score[:ood_num_examples].copy(), mean_acc, acc
    else:
        return ood_score[:ood_num_examples].copy()


def calc_accuracy(X, true_labels):
    predicted = np.argmax(X, axis=1)
    total = len(true_labels)
    acc = np.sum(predicted == true_labels) / total
    return acc


def sample_error(arr):
    std_dev = stat.stdev(arr)
    sample_err = std_dev / (len(arr) ** 0.5)
    return sample_err


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
net.to(device)

# OOD loaders for test ood datasets
ood_loaders = {
    # "cifar10": cifar10_data,
    # "cifar100": cifar100_data,
    "lsunc": lsunc_loader,
    "textures": texture_loader,
    "svhn": svhn_loader,
    "isun": isun_loader,
    "places_365": places365_loader,
}
accuracies = []
output_metrics_dir = os.path.join(
    "icdm/{}/tests".format(args.method), f"{args.dataset}"
)
if not os.path.exists(output_metrics_dir):
    os.makedirs(output_metrics_dir)

if args.detector != "msp":
    detector_output_dirs = "icdm/detectors/{}/{}".format(args.method, args.detector)
    if not os.path.exists(detector_output_dirs):
        os.makedirs(detector_output_dirs)
    output_metrics_dir = detector_output_dirs

if args.method == "macs":
    margins_length = 10
else:
    margins_length = 1

for i in range(margins_length):
    margin = i / 10

    with open(
        "{}/{}_{}_{}_margin_{}.csv".format(
            output_metrics_dir, args.model, args.dataset, args.exp_name, margin
        ),
        "w",
    ) as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["run", "margin", "ood_dataset", "auroc", "aupr", "fpr"])

    trial_results = []
    for run in range(10):
        print(f"Evaluating for trial {run+1}")
        metrics = []
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=test_bs,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        model_path = "icdm/{}/train_logs_and_ckpts_{}/{}/{}_1_{}_ckpt9.pt".format(
            args.method, args.outlier_name, args.model, args.dataset, margin
        )
        net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        net.to(device)
        net.eval()
        in_score, accuracy, test_accuracies = get_scores(
            test_loader, calc_id_acc=True, in_dist=True
        )
        print(f"The accuracy is: {round(accuracy, 2)} %")
        accuracies.append(accuracy)
        aurocs = []
        auprs = []
        fprs = []
        for ood_name, ood_loader in ood_loaders.items():
            if args.dataset == ood_name:
                continue
            auroc, aupr, fpr = get_results(ood_loader, in_score)
            aurocs.append(auroc)
            auprs.append(aupr)
            fprs.append(fpr)
            metrics.append(
                [
                    run + 1,
                    margin,
                    ood_name,
                    round((auroc * 100), 4),
                    round((aupr * 100), 4),
                    round((fpr * 100), 4),
                ]
            )
        mean_auroc = np.mean(aurocs)
        mean_aupr = np.mean(auprs)
        mean_fpr = np.mean(fprs)
        mean_result = [
            f"mean_run_{run+1}",
            margin,
            "all_avg",
            round((mean_auroc * 100), 2),
            round((mean_aupr * 100), 2),
            round((mean_fpr * 100), 2),
        ]
        metrics += [mean_result]

        with open(
            "{}/{}_{}_{}_margin_{}.csv".format(
                output_metrics_dir,
                args.model,
                args.dataset,
                args.exp_name,
                margin,
            ),
            "a",
        ) as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(metrics)

        trial_results.append([mean_auroc, mean_aupr, mean_fpr])

    std_errs = [
        round(sample_error(trial_result), 4) * 100
        for trial_result in zip(*trial_results)
    ]
    print(std_errs)
    std_err_acc = sample_error(accuracies)
    print(std_err_acc)

    avg_trial_result = [
        round((sum(trial_result) / len(trial_result)) * 100, 2)
        for trial_result in zip(*trial_results)
    ]

    # std_errs = [sample_error(avg_trial_result)]
    mean_details = ["Trial", "avg.", "****"]
    err_details = ["std_errs", "acc,auroc,aupr,fpr"]

    print(f"Mean accuracy is {round(np.mean(accuracies), 2)}%")
    final_result = mean_details + avg_trial_result
    errs_result = err_details + [std_err_acc] + std_errs
    with open(
        "{}/{}_{}_{}_margin_{}.csv".format(
            output_metrics_dir,
            args.model,
            args.dataset,
            args.exp_name,
            margin,
        ),
        "a",
    ) as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows([final_result])
        csvwriter.writerows([errs_result])
