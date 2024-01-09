import pandas as pd  # additional dependency, used here for convenience
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import numpy as np
from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
    Places365,
)
from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
    RMD,
    DICE,
    SHE,
)
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed
import os
import csv
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fix_random_seed(123)

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

# Setup preprocessing
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")

# Setup datasets

dataset_in_test = CIFAR10(root="data", train=False, transform=trans, download=True)

# create all OOD datasets
ood_datasets = [
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
    LSUNCrop,
    LSUNResize,
    Places365,
]
datasets = {}
for ood_dataset in ood_datasets:
    dataset_out_test = ood_dataset(
        root="data", transform=trans, target_transform=ToUnknown(), download=True
    )
    test_loader = DataLoader(
        dataset_in_test + dataset_out_test, batch_size=256, num_workers=12
    )
    datasets[ood_dataset.__name__] = test_loader

# **Stage 1**: Create DNN with pre-trained weights from the Hendrycks baseline paper
print("STAGE 1: Creating a Model")
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)

# **Stage 2**: Create OOD detector
print("STAGE 2: Creating OOD Detectors")
detectors = {}
# detectors["Entropy"] = Entropy(model)
# detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
# detectors["Mahalanobis+ODIN"] = Mahalanobis(
#     model.features, norm_std=norm_std, eps=0.002
# )
# detectors["Mahalanobis"] = Mahalanobis(model.features)
# detectors["KLMatching"] = KLMatching(model)
# detectors["SHE"] = SHE(model.features, model.fc)
detectors["MSP"] = MaxSoftmax(model)
# detectors["EnergyBased"] = EnergyBased(model)
# detectors["MaxLogit"] = MaxLogit(model)
# detectors["ODIN"] = ODIN(model, norm_std=norm_std, eps=0.002)
# detectors["DICE"] = DICE(
#     model=model.features, w=model.fc.weight, b=model.fc.bias, p=0.65
# )
# detectors["RMD"] = RMD(model.features)

# fit detectors to training data (some require this, some do not)
print(f"> Fitting {len(detectors)} detectors")
loader_in_train = DataLoader(
    CIFAR10(root="data", train=True, transform=trans), batch_size=256, num_workers=12
)
for name, detector in detectors.items():
    print(f"--> Fitting {name}")
    detector.fit(loader_in_train, device=device)

# **Stage 3**: Evaluate Detectors
print(f"STAGE 3: Evaluating {len(detectors)} detectors on {len(datasets)} datasets.")

with torch.no_grad():
    for detector_name, detector in detectors.items():
        results = []
        auroc = []
        aupr = []
        fpr95 = []
        print(f"> Evaluating {detector_name}")
        for dataset_name, loader in datasets.items():
            print(f"--> {dataset_name}")
            metrics = OODMetrics()
            for x, y in loader:
                metrics.update(detector(x.to(device)), y.to(device))

            r = {"Detector": detector_name, "Dataset": dataset_name}
            r.update(metrics.compute())

            results.append(r)

# calculate mean scores over all datasets, use percent
df = pd.DataFrame(results)
mean_auroc = df.groupby("Detector")["AUROC"].mean() * 100
mean_aupr_in = df.groupby("Detector")["AUPR-IN"].mean() * 100
mean_aupr_out = df.groupby("Detector")["AUPR-OUT"].mean() * 100
mean_fpr95 = df.groupby("Detector")["FPR95TPR"].mean() * 100

df.loc[len(df.index)] = [
    "MSP",
    "Mean Values",
    mean_auroc.values[0],
    mean_aupr_in.values[0],
    mean_aupr_out.values[0],
    mean_fpr95.values[0],
]
df.to_csv("logs/pytorch_ood/fine_tuning_results/{}.csv".format(args.exp_name))
