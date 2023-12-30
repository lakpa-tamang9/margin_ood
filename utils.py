"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import sklearn.metrics as sk


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


recall_level_default = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError(
            "cumsum was found to be unstable: "
            "its last element does not correspond to sum"
        )
    return out


def fpr_and_fdr_at_recall(
    y_true, y_score, recall_level=recall_level_default, pos_label=None
):
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = (
        np.r_[recall[sl], 1],
        np.r_[fps[sl], 0],
        np.r_[tps[sl], 0],
        thresholds[sl],
    )

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (
        np.sum(np.logical_not(y_true))
    )  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[: len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def show_performance(pos, neg, method_name="Ours", recall_level=recall_level_default):
    """
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    """

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    # print('\t\t\t' + method_name)
    print("FPR{:d}:\t\t\t{:.2f}".format(int(100 * recall_level), 100 * fpr))
    print("AUROC:\t\t\t{:.2f}".format(100 * auroc))
    print("AUPR:\t\t\t{:.2f}".format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))


def print_measures(
    auroc, aupr, fpr, method_name="Ours", recall_level=recall_level_default
):
    # print('\t\t\t\t' + method_name)
    # print('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
    print("& {:.2f} & {:.2f} & {:.2f}".format(100 * fpr, 100 * auroc, 100 * aupr))
    # print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    # print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    # print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


def print_measures_with_std(
    aurocs, auprs, fprs, method_name="Ours", recall_level=recall_level_default
):
    # print('\t\t\t\t' + method_name)
    # print('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
    print(
        "& {:.2f} & {:.2f} & {:.2f}".format(
            100 * np.mean(fprs), 100 * np.mean(aurocs), 100 * np.mean(auprs)
        )
    )
    print(
        "& {:.2f} & {:.2f} & {:.2f}".format(
            100 * np.std(fprs), 100 * np.std(aurocs), 100 * np.std(auprs)
        )
    )
    # print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    # print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
    # print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))


def show_performance_comparison(
    pos_base,
    neg_base,
    pos_ours,
    neg_ours,
    baseline_name="Baseline",
    method_name="Ours",
    recall_level=recall_level_default,
):
    """
    :param pos_base: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg_base: 0's class scores generated by the baseline
    """
    auroc_base, aupr_base, fpr_base = get_measures(
        pos_base[:], neg_base[:], recall_level
    )
    auroc_ours, aupr_ours, fpr_ours = get_measures(
        pos_ours[:], neg_ours[:], recall_level
    )

    # print('\t\t\t' + baseline_name + '\t' + method_name)
    print(
        "FPR{:d}:\t\t\t{:.2f}\t\t{:.2f}".format(
            int(100 * recall_level), 100 * fpr_base, 100 * fpr_ours
        )
    )
    print("AUROC:\t\t\t{:.2f}\t\t{:.2f}".format(100 * auroc_base, 100 * auroc_ours))
    print("AUPR:\t\t\t{:.2f}\t\t{:.2f}".format(100 * aupr_base, 100 * aupr_ours))
    # print('FDR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
    #     int(100 * recall_level), 100 * fdr_base, 100 * fdr_ours))
