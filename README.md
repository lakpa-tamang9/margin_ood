# Margin bounded Confidence Scores

This repository contains the PyTorch implementation of our OOD detection method, MaCS.

## Required libraries

- PyTorch
- Torchvision
- Numpy

## Dataset used

Download all datasets an put them in the `./data` folder in the root repository of this code. Some of the datasets are accessed from standard PyTorch Torchvision library. Other datasets used in this experiments can be downloaded as:

**Outlier Data**

- [300K Random Images](https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy)

**Test OOD Data**

- [Places 365](http://places2.csail.mit.edu/download.html)
- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- [SVHN](http://ufldl.stanford.edu/housenumbers/)
- [LSUN](https://www.dropbox.com/scl/fi/ohqceel2yrxuhntg0mirg/LSUN.tar.gz?rlkey=l2ovcmekq2gj529m3b2hw2ppp&e=1)
- [iSUN](https://www.dropbox.com/scl/fi/wpkzixs1zbqomg5ufq0dd/iSUN.tar.gz?rlkey=46mty3ly8kk3vdxtlnmdjc6zu&e=1)

## Usage

1. Fine-tuning:

   We provide pretrained models for finetuning at ure using a CIFAR10 dataset, use the following command. `snapshots/baseline` directory. Note that, at this stage we only uploaded a subset of all pretrained models, for WRN and Allconv only. Because pretrained models for resnet and Densenet are quite large, they will be shared publicly through cloud drive later.

   For example, to fine-tune the "MaCS" model on Wideresnet architecture using a CIFAR10 dataset, use the following command. `train.py` supports three methods **OE**, **Energy**, and **MaCS**.

   ```
   python train.py --method macs --model wrn --dataset cifar10
   ```

   To run other benchmarks as listed in the paper. Run following:

   - For MixOE

   ```
   python mix_oe.py --method mixoe --model wrn --dataset cifar10
   ```

   - For DivOE

   ```
   python div_oe.py --method divoe --model wrn --dataset cifar10
   ```

2. Testing:

   We provide the respective fine-tuned models for each network, and dataset. They are located under `./icdm/{method_name}/train_logs_and_ckpts_300k/model_name`. Please note that due to size constrain, we only upload a subset of all fine-tuned models at this stage. Due to this, testing all methods might not be possible. However, all the results are uploaded. Later, we will upload remaining to a cloud drive for access to everyone.

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

The results of all tests are inside `./icdm` folder. The results of all methods, trained on all models and datasets are provided.

Furthermore, we compile all the results of all methods fine-tuned on different models and datasets and present it in a json file. The final output file can be found `./icdm/final_results.json`. To reproduce this run following:

```
python results_stats.py
```
