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
from dataset_utils.svhn_loader import SVHN
from pytorch_ood.detector import Mahalanobis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Testing",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--exp_name",
    "-en",
    type=str,
    default="test",
)
parser.add_argument(
    "--dataset", "-d", type=str, default="cifar10", choices=["cifar10", "cifar100"]
)
parser.add_argument(
    "--temp", type=int, default=100, help="Temperature value to scale the output."
)
args = parser.parse_args()

test_bs = 200
num_to_avg = 1
out_as_pos = True
dataset = args.dataset
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
# svhn_data = dset.ImageFolder(
#     root="data/svhn",
#     transform=trn.Compose(
#         [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
#     ),
# )
svhn_data = SVHN(
    root="data/svhn",
    split="test",
    transform=trn.Compose(
        [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
    ),
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

texture_loader = torch.utils.data.DataLoader(
    texture_data, batch_size=test_bs, num_workers=12, shuffle=True, pin_memory=False
)
svhn_loader = torch.utils.data.DataLoader(
    svhn_data, batch_size=test_bs, num_workers=12, shuffle=True, pin_memory=False
)
places365_loader = torch.utils.data.DataLoader(
    places365_data,
    batch_size=test_bs,
    num_workers=12,
    shuffle=True,
    pin_memory=False,
)
lsunc_loader = torch.utils.data.DataLoader(
    lsunc_data, batch_size=test_bs, num_workers=12, shuffle=True, pin_memory=False
)
# lsunr_loader = torch.utils.data.DataLoader(
#     lsunr_data, batch_size=test_bs, shuffle=True, pin_memory=False
# )
isun_loader = torch.utils.data.DataLoader(
    isun_data, batch_size=test_bs, num_workers=12, shuffle=True, pin_memory=False
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


def mahalanobis_distance(x, mean_vector, cov_matrix):
    x_minus_mu = x - mean_vector
    inv_covmat = inv(cov_matrix)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return np.array(np.sqrt(mahal))


def get_scores(loader, calc_id_acc=False, detector="msp", in_dist=False):
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
            feat, output = net(data)
            # print(feat.shape)

            if calc_id_acc:
                # Calculate accuracy
                _, predicted = output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc.append(100.0 * correct / total)

            # smax = to_np(F.softmax(output, dim=1))
            if detector == "msp":
                output = output / args.temp
                smax = to_np(output)
                max_val = -np.max(smax, axis=1)
                _score.append(max_val)

            elif detector == "maha":
                feats_np = feat.cpu().numpy()
                # print(feats_np.shape)
                mean_vector, cov_matrix = compute_mean_and_cov(feats_np)
                ood_scores = []
                for feat_np in feats_np:
                    score = mahalanobis_distance(feat_np, mean_vector, cov_matrix)
                    ood_scores.append(score)
                _score.append(ood_scores)

            elif detector == "energy":
                _score.append(
                    to_np((args.temp * torch.logsumexp(output / args.temp, dim=1)))
                )

    if calc_id_acc:
        mean_acc = np.mean(acc)
    ood_score = concat(_score)

    if calc_id_acc:
        return ood_score[:ood_num_examples].copy(), mean_acc, acc
    else:
        return ood_score[:ood_num_examples].copy()


def maha_score(loader):
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data, targets = data.to(device), targets.to(device)
            Mahalanobis(data)


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
    net = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3)

net.to(device)

# OOD loaders for test ood datasets
ood_loaders = {
    "lsunc": lsunc_loader,
    "textures": texture_loader,
    "svhn": svhn_loader,
    "isun": isun_loader,
    "places_365": places365_loader,
}
accuracies = []
output_metrics_dir = os.path.join(
    "./FINAL_RESULTS_imgnet32/wo_margin", "MaPS_temp_{}".format(args.temp)
)
if not os.path.exists(output_metrics_dir):
    os.makedirs(output_metrics_dir)

for i in range(5, 6):
    margin = i / 10

    with open(
        "{}/{}_{}_{}_margin_{}.csv".format(
            output_metrics_dir, model, dataset, args.exp_name, margin
        ),
        "w",
    ) as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["run", "margin", "ood_dataset", "auroc", "aupr", "fpr"])

    trial_results = []
    for run in range(5):
        print(f"Evaluating for trial {run+1}")
        metrics = []
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=test_bs,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
        )
        model_path = (
            "logs_test_and_ckpts_imgnet32/wo_margin/{}/{}_{}_{}_ckpt9.pt".format(
                model, dataset, args.exp_name, margin
            )
        )
        net.load_state_dict(torch.load(model_path))
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
                output_metrics_dir, model, dataset, args.exp_name, margin
            ),
            "a",
        ) as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(metrics)

        trial_results.append([mean_auroc, mean_aupr, mean_fpr])

    avg_trial_result = [round((sum(x) / len(x)) * 100, 2) for x in zip(*trial_results)]
    details = ["Trial", "avg.", "****"]

    print(f"Mean accuracy is {round(np.mean(accuracies), 2)}%")
    final_result = details + avg_trial_result
    with open(
        "{}/{}_{}_{}_margin_{}.csv".format(
            output_metrics_dir, model, dataset, args.exp_name, margin
        ),
        "a",
    ) as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows([final_result])
