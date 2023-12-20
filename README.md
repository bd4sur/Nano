
# nano-gpt

![ ](./nano.jpg)

经典NLP学习项目 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的简化复刻版本，供个人学习和测试炼丹炉之用。

## 使用方法

首先安装依赖，建议在虚拟环境中安装。然后准备数据、训练模型、测试模型。

```
# Install dependencies
pip install -r requirements.txt

# Tokenize the raw corpus text
python tokenizer.py

# (optional) Start TensorBoard
tensorboard --logdir .

# Start pre-training (DDP)
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node 2 train.py
# or train on CPU/single-GPU
python train.py

# Test the DDP pre-trained model
python -m torch.distributed.run test.py
# or test the CPU/single-GPU trained model
python test.py
```

## 研究笔记

炼丹炉配置：

- CPU：Intel Xeon E5-2686 v4 @ 2.30GHz
- Mem：128GB DDR4 2400Mtps
- GPU0 (cuda:0)：Nvidia Tesla P100 PCIE 16GB
- GPU1 (cuda:1)：Nvidia Tesla P40 (24GB)

**实验：多头注意力算子`scaled_dot_product_attention`的性能**

PyTorch 2.0 以上支持基于 [FlashAttention](https://arxiv.org/abs/2205.14135) 的多头注意力计算加速。目前有3种kernel，但是不支持较旧的GPU。分别启用3种kernel，实测相对性能如下：

|Kernel|flash_sdp|mem_efficient_sdp|math_sdp|
|------|------|----|--|
|相对时间|(不支持)|2.75|1(基准)|
|相对显存|(不支持)|0.78|1(基准)|

其他参考资料：

- [论文分享：新型注意力算法FlashAttention](https://www.bilibili.com/video/BV1zs4y1J7tb/)
- https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention
- https://github.com/facebookresearch/xformers
- https://github.com/vllm-project/vllm
- https://github.com/Dao-AILab/flash-attention

**改造：流式输出**

TODO

**魔改：服务于文本嵌入和检索任务**

为什么我认为检索问题极端重要？因为许多问题不在于“不知道”，而在于“不知道知道”。大模型知道一切，但是我们不知道祂知道什么，祂自己也不知道。想要“知道知道”，就要靠检索。

文本生成任务，也可以理解成在prompt的触发下检索出相关信息的检索过程。

- https://github.com/Muennighoff/sgpt
- https://www.zhihu.com/question/510987022/answer/2697787852
