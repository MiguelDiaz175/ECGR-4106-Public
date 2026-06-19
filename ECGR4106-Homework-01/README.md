# ECGR 4106 Homework 1

## Overview

This repository contains the solution files for ECGR 4106 Homework 1. The homework studies several convolutional neural network families on CIFAR-10 using PyTorch:

- Modified AlexNet
- Adapted VGGNet
- ResNet-11
- ResNet-18

The experiments compare architecture size, validation behavior, final test accuracy, training time, and the effect of dropout. All models use the same CIFAR-10 train/validation/test split and the same core hyperparameters so that the comparisons are as fair as possible.

## Repository Contents

| File | Description |
|---|---|
| `homework-1-Problem-1.ipynb` | Problem 1 notebook for modified AlexNet on CIFAR-10, including baseline and dropout experiments. |
| `homework-1-Problem-2.ipynb` | Problem 2 notebook for adapted VGGNet on CIFAR-10, including dropout experiments and AlexNet vs. VGG comparison. |
| `homework-1-Problem-3.ipynb` | Problem 3 notebook for ResNet-11 vs. ResNet-18, including dropout experiments and final model-family comparison. |

Each notebook contains markdown explanations directly above the code cells. These explanations describe what each code cell does, why it is needed, and how the output should be interpreted.

## Environment

The notebooks were designed to run in Google Colab with GPU acceleration.

Recommended runtime:

- Python 3
- Google Colab GPU runtime
- PyTorch
- torchvision
- NumPy
- pandas
- matplotlib
- PIL / Pillow

The notebooks automatically select CUDA when available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

If `cuda` is not available, the notebooks can still run on CPU, but training will be much slower.

## Dataset

All experiments use CIFAR-10.

The dataset setup is consistent across all three notebooks:

- Training set: 45,000 images
- Validation set: 5,000 images
- Test set: 10,000 images
- Random seed: 42

Training transforms:

- Random crop to 32x32 with padding 4
- Random horizontal flip
- Convert to tensor
- Normalize with CIFAR-10 channel statistics

Validation and test transforms:

- Convert to tensor
- Normalize with CIFAR-10 channel statistics

Normalization values:

```python
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
```

The notebooks include fallback dataset-loading logic because the official CIFAR-10 download URL sometimes returns `HTTP Error 403: Forbidden` in Colab. If the normal torchvision download fails, the notebook attempts alternate loading methods.

## Shared Hyperparameters

Unless otherwise stated, the experiments use:

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Scheduler | CosineAnnealingLR |
| Random seed | 42 |
| AlexNet epochs | 30 |
| VGG epochs | 30 |
| ResNet epochs | 50 |

The same hyperparameter setup is used so model comparisons reflect architectural differences rather than training setup differences.

## Problem 1: Modified AlexNet

Problem 1 adapts AlexNet for CIFAR-10. Original AlexNet was designed for larger ImageNet images and has approximately 61 million parameters. The modified version uses:

- Smaller 3x3 convolution kernels
- Fewer channels than original AlexNet
- Three max-pooling layers
- A compact fully connected classifier
- A 10-class CIFAR-10 output layer

The notebook trains:

- Modified AlexNet baseline
- Modified AlexNet with dropout `p = 0.3`
- Modified AlexNet with dropout `p = 0.5`

### AlexNet Results

| Variant | Dropout | Parameters | Best Validation Accuracy | Test Accuracy | Mean Epoch Time |
|---|---:|---:|---:|---:|---:|
| AlexNet baseline | None | 4,483,146 | 0.8498 | 0.8477 | 27.52 sec |
| AlexNet dropout | 0.3 | 4,483,146 | 0.8344 | 0.8265 | 25.86 sec |
| AlexNet dropout | 0.5 | 4,483,146 | 0.8466 | 0.8446 | 26.52 sec |

The baseline model performed best. Dropout reduced the train-validation gap but did not improve final test accuracy for the modified AlexNet.

## Problem 2: Adapted VGGNet

Problem 2 adapts a VGG-style CNN for CIFAR-10. The model preserves the VGG idea of repeated 3x3 convolutions but reduces the channel widths and classifier size.

The adapted VGG uses:

- Repeated 3x3 convolution blocks
- Max-pooling between blocks
- Reduced channel widths
- Compact fully connected classifier
- 10-class CIFAR-10 output layer

The notebook trains:

- Adapted VGG baseline
- Adapted VGG with dropout `p = 0.3`
- Adapted VGG with dropout `p = 0.5`

### VGG Results

| Variant | Dropout | Parameters | Best Validation Accuracy | Test Accuracy | Mean Epoch Time |
|---|---:|---:|---:|---:|---:|
| VGG baseline | None | 3,107,338 | 0.8384 | 0.8343 | 32.41 sec |
| VGG dropout | 0.3 | 3,107,338 | 0.8462 | 0.8424 | 32.39 sec |
| VGG dropout | 0.5 | 3,107,338 | 0.8258 | 0.8322 | 32.37 sec |

The best VGG variant used dropout `p = 0.3`. It improved over the VGG baseline, but it did not outperform the best AlexNet result.

## Problem 3: ResNet-11 vs. ResNet-18

Problem 3 implements ResNet models from scratch for CIFAR-10.

The ResNet implementation uses:

- CIFAR-style 3x3 stem convolution
- No initial max-pooling layer
- Residual BasicBlocks
- BatchNorm
- ReLU activations
- Downsampling shortcuts where dimensions change
- Global average pooling
- Fully connected classification head

Dropout is applied after global average pooling for the dropout experiments.

The notebook trains:

- ResNet-11 baseline
- ResNet-18 baseline
- ResNet-11 dropout `p = 0.3`
- ResNet-11 dropout `p = 0.5`
- ResNet-18 dropout `p = 0.3`
- ResNet-18 dropout `p = 0.5`

### ResNet Results

| Model | Dropout | Parameters | Best Validation Accuracy | Test Accuracy | Mean Epoch Time |
|---|---:|---:|---:|---:|---:|
| ResNet-11 | None | 9,623,882 | 0.9260 | 0.9196 | 30.64 sec |
| ResNet-11 | 0.3 | 9,623,882 | 0.9204 | 0.9203 | 30.84 sec |
| ResNet-11 | 0.5 | 9,623,882 | 0.9268 | 0.9203 | 30.67 sec |
| ResNet-18 | None | 11,173,962 | 0.9318 | 0.9290 | 42.10 sec |
| ResNet-18 | 0.3 | 11,173,962 | 0.9336 | 0.9243 | 42.19 sec |
| ResNet-18 | 0.5 | 11,173,962 | 0.9294 | 0.9231 | 42.05 sec |

The ResNet-18 baseline achieved the highest overall test accuracy. Dropout did not improve ResNet-18, likely because BatchNorm and residual connections already provided stable optimization and regularization.

## Final Model Comparison

| Model | Parameters | Test Accuracy | Mean Epoch Time |
|---|---:|---:|---:|
| Best AlexNet | 4,483,146 | 0.8477 | 27.52 sec |
| Best VGG | 3,107,338 | 0.8424 | 32.39 sec |
| Best ResNet-11 | 9,623,882 | 0.9203 | 30.84 sec |
| Best ResNet-18 | 11,173,962 | 0.9290 | 42.10 sec |

The residual networks performed substantially better than the modified AlexNet and adapted VGG models. ResNet-18 produced the highest test accuracy, while ResNet-11 provided a strong tradeoff between accuracy and training time.

## Running The Notebooks

1. Open the notebook in Google Colab.
2. Select a GPU runtime:

   ```text
   Runtime > Change runtime type > Hardware accelerator > GPU
   ```

3. Run the notebook cells from top to bottom.
4. Confirm that the setup cell prints:

   ```text
   Using device: cuda
   ```

5. If Colab disconnects, rerun the setup cells and then continue from the remaining training cells. The notebooks save results to Google Drive.

## Saved Outputs

The notebooks save outputs to Google Drive folders such as:

```text
/content/drive/MyDrive/ECGR4106_HW1/problem1
/content/drive/MyDrive/ECGR4106_HW1/problem2
/content/drive/MyDrive/ECGR4106_HW1/problem3
```

Saved outputs include:

- Training history CSV files
- Summary JSON files
- Best model checkpoints
- Training and validation curves
- Confusion matrices
- Dropout comparison plots
- Final comparison plots

## Notes

- All model code is written in PyTorch.
- All experiments use CIFAR-10.
- The same random seed and dataset split are used throughout.
- The notebooks include explanations above each code cell.
- The report avoids first-person wording and uses black text only.
