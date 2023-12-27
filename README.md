
# nano-gpt

![ ](./nano.jpg)

经典NLP学习项目 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的简化复刻版本，是生成式大规模语言模型在原理上的完备最小集，供个人赏玩/魔改和炼丹炉煲机之用。

- [Attn] A Vaswani, N Shazeer, N Parmar, et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) [J]. Advances in Neural Information Processing Systems, 2017, 30.
- [GPT-1] A Radford, K Narasimhan, T Salimans, et al. [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [J]. 2018.
- [bbycroft] [GPT可视化](https://bbycroft.net/llm)
- [2001.08361] [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Chinchilla] [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

## 使用方法

首先安装依赖，建议在虚拟环境中安装。

```
# 建议在虚拟环境中玩耍，例如：
conda create -n nanogpt python=3.11 pysocks -y
conda activate nanogpt
# 然后在虚拟环境中安装依赖
pip install -r requirements.txt
```

准备训练数据。

```
python tokenizer.py
```

启动TensorBoard以观察损失函数变化情况。（可选）

```
tensorboard --logdir .
```

### 单机单卡或者分布式数据并行训练

以分布式数据并行（DDP）方式启动训练：

```
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node 2 train_ddp.py
```

或者单机单卡或CPU训练（注意将`train.py`中的`device`选项设为`"cpu"`）：

```
python train.py
```

### 基于DeepSpeed的3D并行训练

可以修改`ds_config.json`以优化训练效果。这里默认是采用ZeRO3-Offload方式训练。以本人炼丹炉的资源，实测发现，最多可以预训练约85M参数的模型。

```
deepspeed --num_nodes=1 --num_gpus=2 train_ds.py --deepspeed --deepspeed_config ds_config.json
```

### 电子鹦鹉：交互式文本生成

如果是以DDP方式或者单机单卡或者CPU训练的模型，则执行以下命令。

```
python test.py
```

如果是DeepSpeed训练的模型，则需要先执行`ckpt/ds`目录中的转换脚本，将其转化为PyTorch能够接受的state_dict格式，再执行推理脚本：

```
cd nano-gpt/ckpt/ds
python zero_to_fp32.py . ckpt_ds.pt
cd nano-gpt
python test_ds.py
```

## 研究笔记

炼丹炉配置：

```
OS: Ubuntu 20.04.6 LTS x86_64
Host: PowerEdge R730
Kernel: 5.4.0-169-generic
CPU: Intel Xeon E5-2686 v4 (72) @ 3.000GHz
GPU: NVIDIA Tesla P100 PCIe 16GB
GPU: NVIDIA Tesla P40
Memory: 128806MiB
```

**多头注意力算子`scaled_dot_product_attention`的性能**

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

**微调和下游任务适配（详见[GPT-1]）**

- 分类
- 后承（蕴涵）
- 相似度
- 多选

**魔改：服务于文本嵌入和检索任务**

为什么我认为检索问题极端重要？因为许多问题不在于“不知道”，而在于“不知道知道”。大模型知道一切，但是我们不知道祂知道什么，祂自己也不知道。想要“知道知道”，就要靠检索。

文本生成任务，也可以理解成在prompt的触发下检索出相关信息的检索过程。

- https://github.com/Muennighoff/sgpt
- https://www.zhihu.com/question/510987022/answer/2697787852
