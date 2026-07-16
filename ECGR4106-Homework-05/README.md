# ECGR 4106 Homework 5 - Vision Transformers for Image Classification

Student: Miguel Diaz-Alvarez  
Student ID: 801402124  
Course: ECGR 4106 - Introduction to Deep Learning  
Repository: https://github.com/MiguelDiaz175/ECGR-4106-Public.git

## Overview

This folder contains the Homework 5 implementation and results for CIFAR-100 image classification with Vision Transformer and Swin Transformer models.

The assignment has two parts:

- Problem 1: scratch Vision Transformer configurations compared with a pretrained ResNet-18 baseline.
- Problem 2: pretrained frozen Swin-Tiny and Swin-Small compared with a compact scratch Swin Transformer.

All final numbers in the report come from the downloaded Google Colab result files in `results/`.

## Folder Structure

```text
homework_5/
├── ECGR4106_Homework5_Report.docx
├── README.md
├── notebooks/
│   ├── problem1_vit_resnet_cifar100.ipynb
│   └── problem2_swin_cifar100.ipynb
├── src/
│   ├── data_utils.py
│   ├── metrics_utils.py
│   ├── reproducibility.py
│   ├── swin_scratch.py
│   ├── training_utils.py
│   └── vit_model.py
├── results/
│   ├── problem1/
│   └── problem2/
├── requirements.txt
└── tools/
```

## How to Run

Open the notebooks in Google Colab and run them from top to bottom.

Problem 1:

```text
notebooks/problem1_vit_resnet_cifar100.ipynb
```

Problem 2:

```text
notebooks/problem2_swin_cifar100.ipynb
```

The notebooks automatically:

- install required packages;
- load CIFAR-100 with `torchvision.datasets.CIFAR100`;
- select CUDA when available and CPU otherwise;
- set reproducibility seeds;
- create result directories;
- save CSV, JSON, checkpoint, and plot files.

## Problem 1 Summary

All Problem 1 models used:

- dataset: CIFAR-100;
- epochs: 10;
- batch size: 64;
- optimizer: Adam;
- learning rate: 0.001.

| Model | Test Accuracy | Total Parameters | Trainable Parameters | Total Time (s) |
|---|---:|---:|---:|---:|
| vit_p4_d256_l4_h4 | 30.74% | 3,214,692 | 3,214,692 | 365.2 |
| vit_p4_d512_l8_h8 | 5.53% | 25,330,276 | 25,330,276 | 1,977.6 |
| vit_p8_d256_l4_h4 | 10.05% | 3,239,268 | 3,239,268 | 307.1 |
| vit_p8_d512_l8_h8 | 6.03% | 25,379,428 | 25,379,428 | 593.9 |
| resnet18_pretrained_imagenet | 70.48% | 11,227,812 | 11,227,812 | 1,706.7 |

Problem 1 result files:

```text
results/problem1/problem1_results.csv
results/problem1/problem1_results.json
results/problem1/training_history_*.csv
results/problem1/checkpoints/*.pt
results/problem1/plots/*_training_curves.png
```

## Problem 2 Summary

Pretrained Swin-Tiny and Swin-Small used:

- epochs: 5;
- batch size: 32;
- optimizer: Adam;
- learning rate: 2e-5;
- trainable parameters: classifier head only.

The scratch Swin model used:

- epochs: 5;
- batch size: 32;
- optimizer: Adam;
- learning rate: 0.001;
- random initialization;
- all parameters trainable.

| Model | Test Accuracy | Total Parameters | Trainable Parameters | Total Time (s) |
|---|---:|---:|---:|---:|
| swin_tiny_pretrained_frozen | 66.42% | 27,596,254 | 76,900 | 1,386.0 |
| swin_small_pretrained_frozen | 70.28% | 48,914,158 | 76,900 | 2,484.2 |
| swin_scratch_compact | 3.20% | 5,165,566 | 5,165,566 | 1,259.0 |

Problem 2 result files:

```text
results/problem2/problem2_results.csv
results/problem2/problem2_results.json
results/problem2/training_history_*.csv
results/problem2/checkpoints/*.pt
results/problem2/plots/*_training_curves.png
```

## Notes

- FLOP values are reported using the convention that FLOPs are approximated as `2 x MACs`.
- The raw complexity tool output is labeled as MACs.
- The pretrained Swin classifier layers were reinitialized for 100 CIFAR-100 classes.
- The pretrained Swin backbones were explicitly frozen with `requires_grad = False`.
- The final report is provided as `ECGR4106_Homework5_Report.docx` and can be converted to PDF for submission.
