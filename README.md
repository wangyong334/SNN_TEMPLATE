# SNN_Template：SpikingJelly + CIFAR-10 图像识别项目

这是一个标准、模块化、适合学习和扩展的 SNN 图像识别项目。

## 项目设定

```text
任务：CIFAR-10 图像分类
框架：SpikingJelly
神经元：LIFNode
训练：Surrogate Gradient + BPTT
默认编码：Direct Encoding
默认网络：小型 Spiking CNN
读出：时间维 logits / 膜电位证据平均
损失：CrossEntropyLoss
```

## 本机环境

已检测到你的 `snn` 环境：

```text
Python: 3.11.15
torch: 2.11.0+cu128
torchvision: 0.26.0+cu128
CUDA Runtime: 12.8
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
```

你的 `nvidia-smi` 显示：

```text
Driver Version: 573.22
CUDA Version: 12.8
```

建议使用：

```powershell
D:\Environment\envs\snn\python.exe
```

## 快速自检

在 `SNN_Template` 目录下运行：

```powershell
D:\Environment\envs\snn\python.exe train.py --epochs 1 --max-train-batches 2 --max-test-batches 2
```

这个命令只跑少量 batch，用来检查：

```text
数据能否读取
模型能否前向传播
loss 是否能反向传播
checkpoint 和日志能否写入
```

## 正式训练

```powershell
D:\Environment\envs\snn\python.exe train.py --epochs 10 --batch-size 64 --time-steps 4 --encoder direct
```

也可以运行：

```powershell
.\scripts\run_train.ps1
```

## 训练输出

训练过程中会打印：

```text
train/eval progress bar
loss
accuracy
spike_rate
time
learning_rate
best_acc
```

并保存：

```text
checkpoints/best.pt
checkpoints/last.pt
logs/train_metrics.csv
```

## 分析训练结果

```powershell
python analyze_results.py
```

会输出：

```text
best_test_acc
last_train_acc / last_test_acc
generalization_gap
train/test spike_rate
```

并生成：

```text
logs/metrics_curve.png
```

## 张量形状主线

```text
DataLoader:
    images [N, 3, 32, 32]
    labels [N]

Encoder:
    x_seq [T, N, 3, 32, 32]

Model:
    logits_seq [T, N, 10]

Readout:
    logits = logits_seq.mean(dim=0)
    logits [N, 10]

Loss:
    CrossEntropyLoss(logits, labels)
```

## 如何替换模块

### 替换编码方式

```powershell
D:\Environment\envs\snn\python.exe train.py --encoder rate
```

编码代码在：

```text
src/snn_template/encoding/encoders.py
```

### 替换网络结构

新增模型类后，在这里注册：

```text
src/snn_template/models/cifar_snn.py
build_model()
```

### 调整时间步

```powershell
D:\Environment\envs\snn\python.exe train.py --time-steps 8
```

一般规律：

```text
T 越大，时间信息更充分，但训练更慢、延迟更高。
T 越小，训练更快、延迟更低，但精度可能下降。
```
