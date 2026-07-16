"""Reusable supervised training and evaluation loops."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from metrics_utils import accuracy_from_logits, ensure_dir
from reproducibility import cuda_synchronize_if_needed


def extract_logits(model_output):
    """Handle plain Tensor outputs and Hugging Face outputs with .logits."""

    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def set_named_modules_eval(model: nn.Module, module_names: Iterable[str]) -> None:
    """Put selected top-level modules into eval mode during head-only training."""

    for name in module_names:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            module.eval()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    max_batches: int | None = None,
    frozen_eval_modules: Iterable[str] = (),
) -> dict[str, float]:
    """Train for one epoch and return loss, accuracy, and elapsed seconds."""

    device = torch.device(device)
    model.train()
    set_named_modules_eval(model, frozen_eval_modules)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    cuda_synchronize_if_needed(device)
    start = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = extract_logits(model(images))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    cuda_synchronize_if_needed(device)
    elapsed = time.perf_counter() - start
    if total_examples == 0:
        raise RuntimeError("no train batches")
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": total_correct / total_examples,
        "epoch_time_seconds": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device | str,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate with gradients disabled."""

    device = torch.device(device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = extract_logits(model(images))
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size
    if was_training:
        model.train()
    if total_examples == 0:
        raise RuntimeError("no eval batches")
    return {
        "eval_loss": total_loss / total_examples,
        "eval_accuracy": total_correct / total_examples,
    }


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Save model weights plus serializable metadata."""

    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )
    return path


def fit_classifier(
    model: nn.Module,
    train_loader,
    eval_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epochs: int,
    checkpoint_path: str | Path | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    frozen_eval_modules: Iterable[str] = (),
) -> tuple[list[dict], dict]:
    """Train/evaluate a classifier and return per-epoch history plus summary."""

    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best_accuracy = -1.0
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches,
            frozen_eval_modules=frozen_eval_modules,
        )
        eval_metrics = evaluate(
            model,
            eval_loader,
            criterion,
            device,
            max_batches=max_eval_batches,
        )
        row = {"epoch": epoch, **train_metrics, **eval_metrics}
        history.append(row)
        if checkpoint_path is not None and eval_metrics["eval_accuracy"] >= best_accuracy:
            best_accuracy = eval_metrics["eval_accuracy"]
            save_checkpoint(
                model,
                checkpoint_path,
                {"best_epoch": epoch, "best_eval_accuracy": best_accuracy},
            )

    total_time = time.perf_counter() - total_start
    summary = {
        "total_training_time_seconds": total_time,
        "mean_epoch_time_seconds": sum(r["epoch_time_seconds"] for r in history) / len(history),
        "final_eval_accuracy": history[-1]["eval_accuracy"],
        "final_eval_loss": history[-1]["eval_loss"],
    }
    return history, summary


def verify_optimizer_parameters(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    """Return trainable parameter count and optimizer parameter count."""

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimized = 0
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) not in seen:
                optimized += param.numel()
                seen.add(id(param))
    if trainable != optimized:
        raise ValueError("bad optimizer params")
    return trainable, optimized
