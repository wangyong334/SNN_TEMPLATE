# SNN_Template 环境安装命令

以下命令请在 **Anaconda Prompt** 中执行。

## 1. 创建新环境

```bat
conda create -n snn_template python=3.11 -y
```

激活环境：

```bat
conda activate snn_template
```

确认 Python：

```bat
python --version
```

## 2. 进入项目目录

```bat
cd /d C:\Users\admin\Desktop\SNN学习\SNN_Template
```

## 3. 安装 PyTorch CUDA 12.8

先安装 CUDA 版 PyTorch：

```bat
python -m pip install --upgrade pip
pip uninstall torch torchvision torchaudio -y
pip install -r requirements-cu128.txt
```

确认不是 CPU 版：

```bat
python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

期望看到类似：

```text
torch_cuda= 12.8
cuda_available= True
gpu= NVIDIA GeForce RTX 5060 Laptop GPU
```

## 4. 安装项目普通依赖

再安装 SpikingJelly 等普通依赖：

```bat
pip install -r requirements.txt
```

`requirements.txt` 会安装：

```text
spikingjelly
numpy
pillow
tqdm
```

## 5. CUDA 说明

你的 `nvidia-smi` 显示：

```text
Driver Version: 573.22
CUDA Version: 12.8
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
```

对本项目来说，通常不需要单独安装系统级 CUDA Toolkit。

原因：

```text
PyTorch 的 cu128 wheel 已经自带运行时 CUDA 组件；
只要显卡驱动足够新，torch.cuda 就可以使用 GPU。
```

只有在你后面要编译自定义 CUDA/C++ 扩展时，才需要额外安装 CUDA Toolkit。

## 6. 验证 PyTorch + CUDA

```bat
python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

期望看到：

```text
cuda_available= True
gpu= NVIDIA GeForce RTX 5060 Laptop GPU
```

## 7. 验证 SpikingJelly

```bat
python -c "import spikingjelly; from spikingjelly.activation_based import neuron, surrogate, functional; print('spikingjelly ok'); print(neuron.LIFNode); print(surrogate.ATan); print(hasattr(functional, 'reset_net'))"
```

## 8. 快速跑通项目

只跑 1 个训练 batch 和 1 个测试 batch，用于检查环境：

```bat
python train.py --epochs 1 --batch-size 8 --time-steps 2 --num-workers 0 --max-train-batches 1 --max-test-batches 1 --log-interval 1
```

## 9. 正式训练命令

```bat
python train.py --epochs 10 --batch-size 64 --time-steps 4 --encoder direct
```

如果显存不够，把 batch size 调小：

```bat
python train.py --epochs 10 --batch-size 32 --time-steps 4 --encoder direct
```

如果想试 Rate Coding：

```bat
python train.py --epochs 10 --batch-size 64 --time-steps 4 --encoder rate
```
