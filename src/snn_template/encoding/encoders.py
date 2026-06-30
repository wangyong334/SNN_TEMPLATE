"""输入编码模块。

编码的作用：
    把普通图像张量 [N, C, H, W] 转换成时间序列 [T, N, C, H, W]。

SNN 和 ANN 的关键差异之一就在这里：
    ANN 通常一次前向传播一张图；
    SNN 需要在 T 个时间步上逐步更新神经元状态。
"""

from __future__ import annotations

import torch
from torch import nn


class BaseEncoder(nn.Module):
    """编码器基类。

    子类必须输入:
        x: torch.Tensor，形状 [N, C, H, W]

    子类必须输出:
        encoded: torch.Tensor，形状 [T, N, C, H, W]
    """

    def __init__(self, time_steps: int) -> None:
        super().__init__()
        self.time_steps = time_steps


class DirectEncoder(BaseEncoder):
    """Direct Encoding：直接编码。

    原理：
        不生成输入 spike，而是把同一张图像作为每个时间步的输入电流。

    输入:
        x: [N, C, H, W]，连续值图像张量

    输出:
        [T, N, C, H, W]，每个时间步都是同一个 x

    适合:
        深度 SNN 监督训练、入门 baseline、替代梯度训练。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)


class RateEncoder(BaseEncoder):
    """Rate Coding：频率编码。

    原理：
        像素值越大，每个时间步越容易产生输入 spike。

    输入:
        x: [N, C, H, W]，建议范围 [0, 1]

    输出:
        spikes: [T, N, C, H, W]，0/1 脉冲序列

    注意:
        这个编码包含随机采样，同一个输入每次可能产生略不同的 spike。
        如果要复现实验，需要固定随机种子。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prob = x.clamp(0.0, 1.0).unsqueeze(0)
        random_values = torch.rand(
            (self.time_steps, *x.shape),
            device=x.device,
            dtype=x.dtype,
        )
        return (random_values < prob).to(x.dtype)


class LatencyEncoder(BaseEncoder):
    """Latency Coding：延迟编码。

    原理：
        像素值越大，越早在时间窗口内发放一次 spike。

    输入:
        x: [N, C, H, W]，建议范围 [0, 1]

    输出:
        spikes: [T, N, C, H, W]，每个像素最多发放一次 spike

    说明:
        这个实现是教学版，适合观察编码机制。CIFAR-10 baseline 默认不用它。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        spike_time = torch.round((1.0 - x) * (self.time_steps - 1)).long()

        spikes = torch.zeros(
            (self.time_steps, *x.shape),
            device=x.device,
            dtype=x.dtype,
        )

        for t in range(self.time_steps):
            spikes[t] = (spike_time == t).to(x.dtype)

        return spikes


def build_encoder(name: str, time_steps: int) -> BaseEncoder:
    """根据名字创建编码器，方便后续替换编码方式。"""
    name = name.lower()
    if name == "direct":
        return DirectEncoder(time_steps=time_steps)
    if name == "rate":
        return RateEncoder(time_steps=time_steps)
    if name == "latency":
        return LatencyEncoder(time_steps=time_steps)
    raise ValueError(f"未知编码方式: {name}")

