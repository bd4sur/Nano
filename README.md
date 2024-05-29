
# Nano

大模型，小玩具。

![ ](./nano.jpg)

## NLP

经典NLP学习项目 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的简化复刻版本，供个人赏玩、魔改和炼丹炉煲机之用。

- [Attn] A Vaswani, N Shazeer, N Parmar, et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) [J]. Advances in Neural Information Processing Systems, 2017, 30.
- [GPT-1] A Radford, K Narasimhan, T Salimans, et al. [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [J]. 2018.
- [bbycroft] [GPT可视化](https://bbycroft.net/llm)
- [2001.08361] [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Chinchilla] [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

```json
{
    # GPT Model Args
    "block_size": 128, # 如果是Q问题，则为 q_digits() + 1,
    "vocab_size": 10000,
    "n_layer": 2,
    "n_head": 4,
    "n_embd": 64,
    "dropout": 0.0, # for pretraining 0 is good, for finetuning try 0.1+
    "bias": False, # do we use bias inside LayerNorm and Linear layers?
    "is_causal": True, # 如果是排序问题，则为False

    # AdamW Optimizer Args
    "learning_rate": 6e-4, # max learning rate
    "max_iters": 100000, # total number of training iterations
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.99,

    # Learning Rate Scheduler
    "decay_lr": True, # whether to decay the learning rate
    "warmup_iters": 300, # how many steps to warm up for
    "lr_decay_iters": 100000, # should be ~= max_iters per Chinchilla
    "min_lr": 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # Training Task
    "init_from": 'pretrain', # 'pretrain' or 'finetune'
    "batch_size": 300,
    "random_seed": 114514,
    "eval_only_last_token_loss": False, # 如果是Q问题，则为True；如果是NLG问题，则为False
    "data_dir": 'data_q',
    "ckpt_dir": 'ckpt_q',
    "eval_interval": 100,
    "log_interval": 10,
    "eval_iters": 5,

    # Misc
    "backend": 'nccl', # 'nccl', 'gloo', etc.
    "device": 'cuda:0', # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
}
```

### 玩法1：人类的本质是复读机！

- 数据预处理：`data.py`主函数中调用`generate_text`，执行。
- 训练模型：`train.py`中，修改`block_size`为适当值，修改`is_causal`为`True`，修改`eval_only_last_token_loss`为`False`。
- 测试模型：`test.py`中，主函数`test()`。

注：仓库中增加了来自[hhiim/Lacan](https://github.com/hhiim/Lacan)的精神分析黑话数据集，特此致谢。

### 玩法2：丘成桐先生也答不出的Q问题

所谓“Q问题”，是《鲁豫有约》20150902期节目中，主持人给丘成桐出的一道脑筋急转弯题。

![ ](./q.jpg)

- 数据预处理：`data.py`主函数中调用`generate_problem_q`，执行。
- 训练模型：`train.py`中，修改`block_size`为`q_digits() + 1`，修改`is_causal`为`True`，修改`eval_only_last_token_loss`为`True`（不改也行，但是有点奇怪）。
- 测试模型：`test.py`中，主函数`test("q")`。

### 玩法3：排序

- 数据预处理：`data.py`主函数中调用`generate_sorting`，执行。
- 训练模型：`train.py`中，修改`block_size`为`q_digits()`，修改`is_causal`为`False`，修改`eval_only_last_token_loss`为`False`。
- 测试模型：`test.py`中，主函数`test("sorting")`。

### 使用方法

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

可以修改`ds_config.json`以优化训练效果。注意：根据[文档](https://www.deepspeed.ai/docs/config-json/)，`train_batch_size` must be equal to `train_micro_batch_size_per_gpu` * `gradient_accumulation` * number of GPUs。这里采用2节点4卡ZeRO3-Offload方式训练。以本人炼丹炉的资源，实测发现，最多可以预训练约85M参数的模型。

```
deepspeed --hostfile=hostfile.txt train_ds.py --deepspeed --deepspeed_config ds_config.json
```

其中`hostfile.txt`的内容如下：

```
192.168.10.52 slots=2
192.168.10.61 slots=2
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

### 研究笔记

炼丹炉（集群，嘿嘿）配置：

||0号机|1号机|
|--|--|--|
|集群内IP|192.168.10.52|192.168.10.61|
|机器型号|PowerEdge R730|PowerEdge R730|
|OS|Ubuntu 20.04.6 LTS|Ubuntu 20.04.6 LTS|
|内核|5.4.0-169|5.15.0-91|
|CPU|双路 Xeon E5-2686 v4|双路 Xeon E5-2680 v4|
|内存|128GB|32GB|
|GPU驱动版本|545.23.08|545.23.08|
|CUDA版本|12.3|12.3|
|GPU 0|Tesla P100 PCIe 16GB|Tesla P100 PCIe 16GB|
|GPU 1|Tesla P100 PCIe 16GB|Tesla P40|
|PyTorch|2.1.2+cu121|2.1.2+cu121|
|DeepSpeed|0.12.6|0.13.1|

两机之间通过 Cisco Catalyst 4948E 交换机的两个10GbE的SFP+光口进行通信，通过iperf3测速，可以跑满10Gbps。

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

## CV

### CLIP：开集图像分类

### BLIP：视觉理解和语言生成

## Audio
