"""CIFAR-100 data loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


CIFAR100_NUM_CLASSES = 100
CIFAR100_IMAGE_SIZE = 32
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_cifar100_transforms(
    image_size: int,
    train: bool,
    mean: tuple[float, float, float] = CIFAR100_MEAN,
    std: tuple[float, float, float] = CIFAR100_STD,
    augment: bool = True,
) -> transforms.Compose:
    """Create transforms for CIFAR-100 at either native or resized resolution."""

    steps: list[object] = []
    if image_size != CIFAR100_IMAGE_SIZE:
        steps.append(transforms.Resize((image_size, image_size), antialias=True))
    if train and augment:
        if image_size == CIFAR100_IMAGE_SIZE:
            steps.extend(
                [
                    transforms.RandomCrop(CIFAR100_IMAGE_SIZE, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            steps.append(transforms.RandomHorizontalFlip())
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def _limit_dataset(dataset: Dataset, max_items: int | None) -> Dataset:
    if max_items is None:
        return dataset
    return Subset(dataset, list(range(min(max_items, len(dataset)))))


def get_cifar100_dataloaders(
    data_root: str | Path,
    batch_size: int,
    image_size: int,
    mean: tuple[float, float, float] = CIFAR100_MEAN,
    std: tuple[float, float, float] = CIFAR100_STD,
    augment_train: bool = True,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    validation_size: int = 0,
    seed: int = 42,
    max_train_items: int | None = None,
    max_test_items: int | None = None,
    download: bool = True,
) -> dict[str, DataLoader]:
    """Download CIFAR-100 and return train/test or train/val/test loaders."""

    data_root = Path(data_root)
    train_transform = build_cifar100_transforms(image_size, True, mean, std, augment_train)
    test_transform = build_cifar100_transforms(image_size, False, mean, std, False)

    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=download,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=download,
        transform=test_transform,
    )

    if validation_size > 0:
        if validation_size >= len(train_dataset):
            raise ValueError("bad validation_size")
        train_size = len(train_dataset) - validation_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, validation_size],
            generator=generator,
        )
    else:
        val_dataset = None

    train_dataset = _limit_dataset(train_dataset, max_train_items)
    test_dataset = _limit_dataset(test_dataset, max_test_items)
    if val_dataset is not None:
        val_dataset = _limit_dataset(val_dataset, max_test_items)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    loaders = {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    }
    if val_dataset is not None:
        loaders["val"] = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return loaders


def validate_cifar100_labels(loader: DataLoader) -> tuple[int, int]:
    """Return min/max labels from one loader and assert CIFAR-100 range."""

    min_label = CIFAR100_NUM_CLASSES
    max_label = -1
    for _, labels in loader:
        min_label = min(min_label, int(labels.min().item()))
        max_label = max(max_label, int(labels.max().item()))
    if min_label < 0 or max_label >= CIFAR100_NUM_CLASSES:
        raise ValueError(f"bad labels: {min_label}-{max_label}")
    return min_label, max_label
