"""分析 SNN_Template 训练结果。

运行：
    python analyze_results.py

功能：
    1. 读取 logs/train_metrics.csv
    2. 自动取最后一次完整训练 run
    3. 打印最佳准确率、最后一轮指标、泛化差距、spike rate
    4. 画出 loss / accuracy / spike_rate 曲线
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib_cache").resolve()))

import matplotlib.pyplot as plt


LOG_PATH = Path("logs/train_metrics.csv")
OUT_PATH = Path("logs/metrics_curve.png")


def read_rows(path: Path) -> list[dict]:
    """读取 CSV 日志，每一行对应一个 epoch。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到训练日志: {path}")

    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["epoch"] = int(row["epoch"])
        for key in [
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "train_spike_rate",
            "test_spike_rate",
            "lr",
        ]:
            row[key] = float(row[key])
    return rows


def select_latest_run(rows: list[dict]) -> list[dict]:
    """选择最后一次训练 run。

    因为快速自检也会写入 epoch=1，所以日志里可能有多个 epoch=1。
    这里从最后一个 epoch=1 开始截取，作为最近一次正式训练。
    """
    start = 0
    for i, row in enumerate(rows):
        if row["epoch"] == 1:
            start = i
    return rows[start:]


def print_summary(rows: list[dict]) -> None:
    """打印训练摘要。"""
    best = max(rows, key=lambda r: r["test_acc"])
    last = rows[-1]
    gap = last["train_acc"] - last["test_acc"]

    print("训练结果摘要")
    print("=" * 60)
    print(f"epochs              : {len(rows)}")
    print(f"best_epoch          : {best['epoch']}")
    print(f"best_test_acc       : {best['test_acc']:.4f}")
    print(f"last_train_acc      : {last['train_acc']:.4f}")
    print(f"last_test_acc       : {last['test_acc']:.4f}")
    print(f"last_train_loss     : {last['train_loss']:.4f}")
    print(f"last_test_loss      : {last['test_loss']:.4f}")
    print(f"generalization_gap  : {gap:.4f}")
    print(f"last_train_spike    : {last['train_spike_rate']:.4f}")
    print(f"last_test_spike     : {last['test_spike_rate']:.4f}")
    print(f"last_lr             : {last['lr']:.8f}")
    print()

    if gap > 0.10:
        print("提示：训练准确率明显高于测试准确率，可能有过拟合趋势。")
    else:
        print("提示：训练和测试准确率差距不大，当前过拟合不明显。")

    if last["test_spike_rate"] > 0.10:
        print("提示：spike_rate 偏高，后续可考虑稀疏正则或调高阈值。")
    elif last["test_spike_rate"] < 0.005:
        print("提示：spike_rate 很低，神经元可能过于安静。")
    else:
        print("提示：spike_rate 处在较稀疏范围，符合 SNN baseline 预期。")


def plot_curves(rows: list[dict], out_path: Path) -> None:
    """画 loss、accuracy、spike_rate 曲线。"""
    epochs = [r["epoch"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="train_loss")
    axes[0].plot(epochs, [r["test_loss"] for r in rows], label="test_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [r["train_acc"] for r in rows], label="train_acc")
    axes[1].plot(epochs, [r["test_acc"] for r in rows], label="test_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, [r["train_spike_rate"] for r in rows], label="train_spike")
    axes[2].plot(epochs, [r["test_spike_rate"] for r in rows], label="test_spike")
    axes[2].set_title("Hidden Spike Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"曲线图已保存: {out_path}")


def main() -> None:
    rows = read_rows(LOG_PATH)
    latest = select_latest_run(rows)
    print_summary(latest)
    plot_curves(latest, OUT_PATH)


if __name__ == "__main__":
    main()
