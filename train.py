"""SNN_Template 训练入口。

推荐运行：
    D:\Environment\envs\snn\python.exe train.py --epochs 10 --batch-size 64 --time-steps 4

快速自检：
    D:\Environment\envs\snn\python.exe train.py --epochs 1 --max-train-batches 2 --max-test-batches 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_template.data.build import build_dataloaders
from snn_template.encoding.encoders import build_encoder
from snn_template.models.cifar_snn import build_model
from snn_template.training.engine import append_csv_log, evaluate, train_one_epoch
from snn_template.utils.checkpoint import load_checkpoint, save_checkpoint
from snn_template.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SpikingJelly SNN on CIFAR-10.")
    parser.add_argument("--data-path", type=str, default="CIFAR-10.zip")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--encoder", type=str, default="direct", choices=["direct", "rate", "latency"])
    parser.add_argument("--model", type=str, default="cifar_snn")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--log-file", type=str, default="logs/train_metrics.csv")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """选择训练设备。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    device = resolve_device(args.device)
    print("环境信息")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"device={device}")
    print()

    print("构建数据集")
    train_loader, test_loader = build_dataloaders(
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"train_batches={len(train_loader)}, test_batches={len(test_loader)}")
    print()

    encoder = build_encoder(args.encoder, time_steps=args.time_steps).to(device)
    model = build_model(
        name=args.model,
        num_classes=10,
        tau=args.tau,
        channels=args.channels,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))
        print(f"已从 {args.resume} 恢复，start_epoch={start_epoch}, best_acc={best_acc:.4f}")

    max_train_batches = args.max_train_batches or None
    max_test_batches = args.max_test_batches or None

    print("训练配置")
    print(f"encoder={args.encoder}, model={args.model}, time_steps={args.time_steps}")
    print(f"tau={args.tau}, channels={args.channels}, batch_size={args.batch_size}")
    print(f"lr={args.lr}, weight_decay={args.weight_decay}, epochs={args.epochs}")
    print()

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            encoder=encoder,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            max_batches=max_train_batches,
        )
        test_metrics = evaluate(
            model=model,
            encoder=encoder,
            loader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=max_test_batches,
            epoch=epoch,
        )
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['acc']:.4f} "
            f"train_spike={train_metrics['spike_rate']:.4f} test_spike={test_metrics['spike_rate']:.4f} "
            f"lr={lr:.6f}"
        )

        append_csv_log(
            args.log_file,
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_acc": f"{train_metrics['acc']:.6f}",
                "test_loss": f"{test_metrics['loss']:.6f}",
                "test_acc": f"{test_metrics['acc']:.6f}",
                "train_spike_rate": f"{train_metrics['spike_rate']:.6f}",
                "test_spike_rate": f"{test_metrics['spike_rate']:.6f}",
                "lr": f"{lr:.8f}",
            },
        )

        is_best = test_metrics["acc"] > best_acc
        if is_best:
            best_acc = test_metrics["acc"]
            save_checkpoint(
                Path(args.output_dir) / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                vars(args),
            )

        save_checkpoint(
            Path(args.output_dir) / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_acc,
            vars(args),
        )

        print(f"best_acc={best_acc:.4f}")
        print("-" * 80)
        scheduler.step()


if __name__ == "__main__":
    main()
