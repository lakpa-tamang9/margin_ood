# Margin bounded Confidence Scores

This repository contains the implementation for the OOD detection method MaCS.

## Usage

1. Fine-tuning:

   For example, to fine-tune the "MaCS" model on Wideresnet architecture using a CIFAR10 dataset, use the following command.

   ```
   python train.py --method macs --model wrn --dataset cifar10
   ```

   Similarly,`train.py` supports three methods **OE**, **Energy**, and **MaCS**. To run other benchmarks as listed in the paper. Run following:

   - For MixOE

   ```
   python mix_oe.py --method mixoe --model wrn --dataset cifar10
   ```

   - For DivOE

   ```
   python div_oe.py --method divoe --model wrn --dataset cifar10
   ```

2. Test:

   For example to test MaCS method on wideresnet architecture using cifar10 as ID data.

   ```
   python test.py --method macs --model wrn --dataset cifar10
   ```

3. Training baseline:

   For WRN, and Allconv models, their pretrained baselines trained on cifar10, and cifar100 were available in OE's original implementation here. Besides that, for those that do not have pretrained baselines, such as resnet-18, and densenet-121, we train the model from scratch with MSP objective. It will be trained upto 100 epochs.

   For example to train a resnet model on cifar10 dataset, run folowing.

   ```
   python baseline.py  --model resnet --dataset cifar10
   ```

## Results

All the results are present in
For MaCS, margin values are between {0.0, 0.9}.

The results of all tests are provided in following file structure:
method/tests/id_datasets/model_dataset_1_margin_0.0.csv

To compile all the outcomes of all methods on different models and datasets and to get the statistically significant results, run following.
