"""CIFAR-10 用小型卷积 SNN。

模型输入:
    x_seq: [T, N, 3, 32, 32]

模型输出:
    logits_seq: [T, N, 10]

读出方式:
    训练脚本中对时间维度求平均：
        logits = logits_seq.mean(dim=0)  # [N, 10]
    然后使用 CrossEntropyLoss。
"""

from __future__ import annotations

import torch
from torch import nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class ConvBNLIF(nn.Module):
    """Conv2d + BatchNorm2d + LIFNode 基础块。

    在多步模式下，输入输出都是:
        [T, N, C, H, W]

    这个块对应 SNN 网络中的一个“空间特征提取 + 脉冲神经元激活”单元。
    """

    def __init__(self, in_channels: int, out_channels: int, tau: float) -> None:
        super().__init__()
        self.conv = layer.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = layer.BatchNorm2d(out_channels)
        self.lif = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate.ATan(),
            detach_reset=False,
        )
        self.last_spike_rate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        spikes = self.lif(x)

        # spikes 是 [T, N, C, H, W] 的 0/1 张量。
        # 这里记录平均发放率，作为能耗/稀疏性的粗略代理指标。
        self.last_spike_rate = spikes.detach().mean()
        return spikes


class CIFAR10SpikingCNN(nn.Module):
    """一个适合 CIFAR-10 入门 baseline 的 Spiking CNN。

    结构:
        ConvBNLIF -> ConvBNLIF -> Pool
        ConvBNLIF -> ConvBNLIF -> Pool
        ConvBNLIF -> Pool
        Flatten -> Linear output

    为什么输出层默认不放 LIF:
        CIFAR-10 监督训练里，用时间平均膜电位/logits 做交叉熵更稳定。
        隐藏层仍然是 LIF 脉冲神经元，训练仍然是替代梯度 + BPTT。
    """

    def __init__(
        self,
        num_classes: int = 10,
        tau: float = 2.0,
        channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        c1 = channels
        c2 = channels * 2
        c3 = channels * 4

        self.net = nn.Sequential(
            ConvBNLIF(3, c1, tau=tau),
            ConvBNLIF(c1, c1, tau=tau),
            layer.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            ConvBNLIF(c1, c2, tau=tau),
            ConvBNLIF(c2, c2, tau=tau),
            layer.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            ConvBNLIF(c2, c3, tau=tau),
            layer.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
            layer.Flatten(start_dim=1),
            layer.Dropout(dropout),
            layer.Linear(c3 * 4 * 4, num_classes),
        )

        # m = multi-step，多步模式。模型会把第 0 维当作时间维 T。
        functional.set_step_mode(self, "m")

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """前向传播。

        输入:
            x_seq: [T, N, 3, 32, 32]

        输出:
            logits_seq: [T, N, 10]
        """
        return self.net(x_seq)

    def spike_rate(self) -> float:
        """统计隐藏层 LIF 的平均发放率。

        这不是严格硬件能耗，只是一个学习阶段很有用的稀疏性指标。
        """
        rates = []
        for module in self.modules():
            if isinstance(module, ConvBNLIF) and module.last_spike_rate is not None:
                rates.append(module.last_spike_rate)
        if not rates:
            return 0.0
        return torch.stack(rates).mean().item()


def build_model(
    name: str,
    num_classes: int,
    tau: float,
    channels: int,
    dropout: float,
) -> nn.Module:
    """根据名字创建模型，后续可以在这里添加 ResNet、Transformer 等结构。"""
    name = name.lower()
    if name == "cifar_snn":
        return CIFAR10SpikingCNN(
            num_classes=num_classes,
            tau=tau,
            channels=channels,
            dropout=dropout,
        )
    raise ValueError(f"未知模型结构: {name}")
