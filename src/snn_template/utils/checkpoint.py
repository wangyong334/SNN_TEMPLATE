"""模型保存与加载工具。"""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    args_dict: dict,
) -> None:
    """保存训练状态，方便中断后继续训练。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_acc": best_acc,
            "args": args_dict,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, device="cpu"):
    """加载 checkpoint。"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

