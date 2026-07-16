"""Reproducibility and runtime-device helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Small serializable description of the selected torch device."""

    device: torch.device
    name: str
    cuda_available: bool
    cuda_device_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "name": self.name,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
        }


def is_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""

    return "COLAB_RELEASE_TAG" in os.environ or "google.colab" in os.environ


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set practical random seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def select_device(prefer_cuda: bool = True) -> DeviceInfo:
    """Select CUDA when available, otherwise CPU."""

    cuda_available = bool(torch.cuda.is_available())
    if prefer_cuda and cuda_available:
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        name = "CPU"
    return DeviceInfo(
        device=device,
        name=name,
        cuda_available=cuda_available,
        cuda_device_count=torch.cuda.device_count(),
    )


def cuda_synchronize_if_needed(device: torch.device | str) -> None:
    """Synchronize CUDA timings only when the active device is CUDA."""

    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

