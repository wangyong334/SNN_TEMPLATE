$Python = "D:\Environment\envs\snn\python.exe"

& $Python train.py `
  --data-path CIFAR-10.zip `
  --epochs 10 `
  --batch-size 64 `
  --time-steps 4 `
  --encoder direct `
  --model cifar_snn `
  --device auto

