"""Metrics, complexity estimates, plotting, and structured exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch of raw logits."""

    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def model_size_megabytes(model: torch.nn.Module) -> float:
    """Estimate parameter storage size in MiB."""

    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_total / (1024**2)


def estimate_macs_thop(
    model: torch.nn.Module,
    input_size: tuple[int, int, int, int],
    device: torch.device | str,
) -> dict[str, Any]:
    """Estimate MACs with thop and label the convention explicitly."""

    try:
        from thop import profile
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "complexity_tool": "thop",
            "complexity_status": "unavailable",
            "macs": None,
            "flops_convention": "not_computed",
            "message": str(exc),
        }

    was_training = model.training
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    try:
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        return {
            "complexity_tool": "thop",
            "complexity_status": "ok",
            "macs": int(macs),
            "params_seen_by_tool": int(params),
            "flops_convention": "thop returns MACs; FLOPs are often approximated as 2x MACs",
            "estimated_flops_if_2x_macs": int(2 * macs),
        }
    except Exception as exc:  # pragma: no cover - model/operator dependent
        return {
            "complexity_tool": "thop",
            "complexity_status": "failed",
            "macs": None,
            "flops_convention": "not_computed",
            "message": str(exc),
        }
    finally:
        if was_training:
            model.train()


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        raise ValueError("empty csv")
    fieldnames: list[str] = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_training_history(history: list[dict[str, Any]], path: str | Path) -> Path:
    return save_csv(history, path)


def plot_training_history(
    history: list[dict[str, Any]],
    title: str,
    path: str | Path,
) -> Path:
    """Save loss and accuracy curves from a per-epoch history table."""

    path = Path(path)
    ensure_dir(path.parent)
    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_loss") for row in history]
    val_acc = [row.get("eval_accuracy") for row in history]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, train_loss, marker="o", label="Train loss", color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, marker="s", label="Eval accuracy", color="#2ca02c")
    ax2.set_ylabel("Eval accuracy")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
