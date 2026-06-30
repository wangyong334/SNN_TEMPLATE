# SNN_Template 项目模块地图

这个项目按 8 个 SNN 开发模块组织，方便后续替换和学习。

## 1. 输入数据模块

文件：

```text
src/snn_template/data/cifar10_zip.py
src/snn_template/data/build.py
```

职责：

```text
CIFAR-10.zip -> image tensor [N, 3, 32, 32] + label [N]
```

## 2. 脉冲编码模块

文件：

```text
src/snn_template/encoding/encoders.py
```

职责：

```text
[N, C, H, W] -> [T, N, C, H, W]
```

当前支持：

```text
direct
rate
latency
```

默认使用：

```text
direct
```

## 3. 神经元模型模块

文件：

```text
src/snn_template/models/cifar_snn.py
```

使用：

```text
spikingjelly.activation_based.neuron.LIFNode
```

训练梯度：

```text
surrogate.ATan()
```

## 4. 时间展开模块

位置：

```text
encoder 输出 [T, N, C, H, W]
functional.set_step_mode(model, "m")
```

含义：

```text
模型按 T 个时间步运行，神经元状态在时间中更新。
```

## 5. 网络结构模块

默认网络：

```text
CIFAR10SpikingCNN
```

结构：

```text
ConvBNLIF -> ConvBNLIF -> Pool
ConvBNLIF -> ConvBNLIF -> Pool
ConvBNLIF -> Pool
Flatten -> Linear -> LIF output
```

后续替换位置：

```text
src/snn_template/models/cifar_snn.py
build_model()
```

## 6. 训练算法模块

文件：

```text
src/snn_template/training/engine.py
```

使用：

```text
替代梯度 + BPTT
```

关键代码：

```text
loss.backward()
```

SpikingJelly 的 LIFNode 会通过 surrogate_function 给 spike 函数提供近似梯度。

## 7. 输出读出与损失函数模块

读出：

```text
logits_seq: [T, N, 10]
logits = logits_seq.mean(dim=0): [N, 10]
```

损失：

```text
CrossEntropyLoss(logits, labels)
```

## 8. 评估指标模块

文件：

```text
src/snn_template/utils/metrics.py
```

当前指标：

```text
loss
accuracy
hidden_spike_rate
learning_rate
```

日志：

```text
logs/train_metrics.csv
```
