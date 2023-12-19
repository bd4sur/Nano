
# nano-gpt

![ ](./nano.jpg)

经典NLP学习项目 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的简化复刻版本，供个人学习和测试炼丹炉之用。

## Quick Start

首先安装依赖，建议在虚拟环境中安装。然后准备数据、训练模型、测试模型。

```
# Install dependencies
pip install -r requirements.txt

# Tokenize the raw corpus text
python tokenizer.py

# (optional) Start TensorBoard
tensorboard --logdir .

# Start pre-training (DDP)
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 train.py
# or train on CPU/single-GPU
python train.py

# Test the DDP pre-trained model
torchrun test.py
# or test the CPU/single-GPU trained model
python test.py
```

## Acknowledgements

[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)