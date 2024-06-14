import numpy
import statistics as stat
import os
import csv
import numpy as np
import json
from tqdm import tqdm


def sample_error(arr):
    std_dev = stat.stdev(arr)
    sample_err = std_dev / (len(arr) ** 0.5)
    return sample_err


def process_csv(dataset, csv_file):
    output_dicts = []
    for ood_dataset in ood_datasets:
        if dataset == ood_dataset:
            continue
        aurocs = []
        auprs = []
        fprs = []
        output_dict = {}
        with open(os.path.join(root_path, csv_file)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["run"] == "std_errs":
                    continue
                if row["ood_dataset"] == ood_dataset:
                    aurocs.append(float(row["auroc"]))
                    auprs.append(float(row["aupr"]))
                    fprs.append(float(row["fpr"]))
        auroc_err = round(sample_error(aurocs), 2)
        aupr_err = round(sample_error(auprs), 2)
        fpr_err = round(sample_error(fprs), 2)

        output_dict["ood"] = ood_dataset
        output_dict["auroc"] = (
            str(round(np.mean(aurocs), 2)) + "\u00B1" + str(auroc_err)
        )
        output_dict["aupr"] = str(round(np.mean(auprs), 2)) + "\u00B1" + str(aupr_err)
        output_dict["fpr"] = str(round(np.mean(fprs), 2)) + "\u00B1" + str(fpr_err)
        output_dicts.append(output_dict)
    return output_dicts


def main():
    method_results = []
    for method in tqdm(methods):
        method_result = {"method": method}
        dataset_results = []
        for dataset in datasets:
            for model in models:
                dataset_result = {"id": dataset, "model": model}
                if method == "macs":
                    margin_results = []
                    for margin in margins:
                        margin_result = {"margin": margin}
                        csv_file = "icdm/{}/tests/{}/{}_{}_{}_margin_{}.csv".format(
                            method, dataset, model, dataset, exp_name, margin
                        )
                        output_dicts = process_csv(dataset=dataset, csv_file=csv_file)
                        margin_result["test_results"] = output_dicts
                        margin_results.append(margin_result)
                    dataset_result["margin_results"] = margin_results
                    dataset_results.append(dataset_result)

                else:
                    csv_file = "icdm/{}/tests/{}/{}_{}_{}_margin_0.0.csv".format(
                        method, dataset, model, dataset, exp_name
                    )
                    output_dicts = process_csv(dataset=dataset, csv_file=csv_file)
                    dataset_result["test_results"] = output_dicts
                    dataset_results.append(dataset_result)
            method_result["all_results"] = dataset_results
        method_results.append(method_result)
    return method_results


if __name__ == "__main__":

    methods = ["oe", "energy", "mix_oe", "div_oe", "macs"]
    # methods = ["macs"]
    models = ["wrn", "allconv", "resnet", "densenet"]
    datasets = ["cifar10", "cifar100", "imgnet32", "svhn"]
    margins = [i / 10 for i in range(10)]
    exp_name = "1"
    root_path = ""  # root path of the code repository
    ood_datasets = ["lsunc", "textures", "svhn", "isun", "places_365"]

    print("Preparing results...")
    method_results = main()
    with open(os.path.join(root_path, "icdm/final_results.json"), "w") as f:
        json.dump(method_results, f)
