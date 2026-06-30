"""训练与评估循环。

这里是 SNN 项目的核心运行逻辑：
    1. images: [N, 3, 32, 32]
    2. encoder(images): [T, N, 3, 32, 32]
    3. model(x_seq): [T, N, 10]
    4. logits_seq.mean(dim=0): [N, 10]
    5. CrossEntropyLoss + BPTT 反向传播
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import torch
from spikingjelly.activation_based import functional
from tqdm.auto import tqdm

from snn_template.utils.metrics import AverageMeter, accuracy_from_logits


PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar:12}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
)


def _effective_batches(loader, max_batches: int | None) -> int:
    """计算本轮实际运行的 batch 数，用于进度条显示。"""
    if max_batches is None:
        return len(loader)
    return min(len(loader), max_batches)


def forward_snn(model, encoder, images):
    """执行一次 SNN 前向传播。

    images:
        [N, 3, 32, 32]，普通图像 batch

    x_seq:
        [T, N, 3, 32, 32]，编码后的时间序列输入

    logits_seq:
        [T, N, 10]，每个时间步的分类证据

    logits:
        [N, 10]，对时间维求平均后的发放率，可送入 CrossEntropyLoss
    """
    x_seq = encoder(images)
    logits_seq = model(x_seq)
    logits = logits_seq.mean(dim=0)
    spike_rate = model.spike_rate() if hasattr(model, "spike_rate") else 0.0
    return logits, logits_seq, spike_rate


def train_one_epoch(
    model,
    encoder,
    loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    log_interval: int,
    max_batches: int | None = None,
):
    """训练一个 epoch。

    替代梯度 + BPTT 在这里发生：
        loss.backward() 会沿着时间维 T 反向传播；
        LIFNode 内部的 surrogate_function 负责给 spike 函数提供近似梯度。
    """
    model.train()
    encoder.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    spike_meter = AverageMeter()
    start_time = time.time()

    total_batches = _effective_batches(loader, max_batches)
    progress = tqdm(
        enumerate(loader),
        total=total_batches,
        desc=f"Train epoch {epoch:03d}",
        bar_format=PROGRESS_BAR_FORMAT,
        leave=True,
        file=sys.stdout,
    )

    for batch_idx, (images, labels) in progress:
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, logits_seq, spike_rate = forward_snn(model, encoder, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # SNN 每个 batch 后必须重置神经元状态，否则上个 batch 的膜电位会污染下个 batch。
        functional.reset_net(model)

        batch_size = labels.numel()
        acc = accuracy_from_logits(logits.detach(), labels)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        spike_meter.update(spike_rate, batch_size)

        if batch_idx % log_interval == 0:
            elapsed = time.time() - start_time
            progress.set_postfix_str(
                f"loss={loss_meter.avg:.4f} "
                f"acc={acc_meter.avg:.4f} "
                f"spk={spike_meter.avg:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"t={elapsed:.1f}s"
            )

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "spike_rate": spike_meter.avg,
    }


@torch.no_grad()
def evaluate(
    model,
    encoder,
    loader,
    criterion,
    device,
    max_batches: int | None = None,
    epoch: int | None = None,
):
    """评估模型。"""
    model.eval()
    encoder.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    spike_meter = AverageMeter()

    total_batches = _effective_batches(loader, max_batches)
    desc = f"Eval  epoch {epoch:03d}" if epoch is not None else "Eval"
    progress = tqdm(
        enumerate(loader),
        total=total_batches,
        desc=desc,
        bar_format=PROGRESS_BAR_FORMAT,
        leave=True,
        file=sys.stdout,
    )

    for batch_idx, (images, labels) in progress:
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, logits_seq, spike_rate = forward_snn(model, encoder, images)
        loss = criterion(logits, labels)
        functional.reset_net(model)

        batch_size = labels.numel()
        acc = accuracy_from_logits(logits, labels)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        spike_meter.update(spike_rate, batch_size)
        progress.set_postfix_str(
            f"loss={loss_meter.avg:.4f} "
            f"acc={acc_meter.avg:.4f} "
            f"spk={spike_meter.avg:.4f}"
        )

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "spike_rate": spike_meter.avg,
    }


def append_csv_log(path: str | Path, row: dict) -> None:
    """把每个 epoch 的指标写入 CSV，方便后续画图分析。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    with path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")
