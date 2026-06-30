"""训练指标工具。"""

from __future__ import annotations

import torch


class AverageMeter:
    """记录一段时间内某个指标的平均值。"""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算 top-1 准确率。

    logits:
        [N, num_classes]
    labels:
        [N]
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

