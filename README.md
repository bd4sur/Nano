
# Nano：大模型，小玩具

本仓库：

- 是Transformer语言模型的极简实现，供个人赏玩、研究、魔改和炼丹炉煲机之用。
- 主要复刻自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，并借鉴了多个开源模型实现和学习项目。

期望：

- 用尽可能少的依赖，实现一个具体而微的Transformer语言模型。
- 完整实现数据预处理、词元化、预训练、监督微调、推理过程。
- 在廉价硬件上从头训练一个[千万量级参数的模型](https://huggingface.co/bd4sur/ham_radio_nanolm_0930)，用于演示（[B站视频](https://www.bilibili.com/video/BV1MtxteUEge)）。
- 研究模型训练的动力学、训/推加速、算法改进等问题，形成研究笔记。
- 探索Transformer模型在自然语言处理以外的问题上的潜能。

为了玩得开心，在下建议读者：

- 掌握基本的机器学习、深度学习知识，及其工程实现技能，如Python语言等。
- 自备计算设备和环境。当然多多益善，但是丰俭由人。以计算能力7.0+的英伟达GPU为宜，如果有多卡甚至多机多卡环境最好；不过CPU也可以运行。
- 将绝大多数工作投入到数据处理上。数据质量对于模型质量有决定性的影响。

为什么叫“Nano”：

- 東雲なの（Shinonome **Nano**）和坂本是动画《日常》的角色。なの是博士创造的女高中生机器人，而坂本是一只会说话的黑猫。
- 致(chao)敬(xi) Karpathy大佬的nanoGPT项目。

![ ](./doc/nano.jpg)

## 立刻开始

[B站视频](https://www.bilibili.com/video/BV1uv42127qP)

**1️⃣ 安装依赖**：建议在conda虚拟环境中安装。

```
conda create -n nano python=3.11 pysocks -y
conda activate nano
pip install -r requirements.txt
```

**2️⃣ 数据预处理**

执行`python data.py`，对预训练和监督微调数据作预处理，包含文本分块、数据集划分、随机打乱、SFT模板组装、词元化等处理步骤。TODO 注意修改代码中语料文本文件的路径。

**词元编码**：本仓库使用最简单的字符映射作为词元编码算法，也就是给语料中所有包含的Unicode字符赋予唯一整数编号，作为词元编号。因此词元实际上就等于是Unicode字符。仓库中同时包含了tiktoken提供的一个BPE词元编码算法，由于速度很慢，并不实用，因此仅作为文档用途。

**预训练数据格式**：原则上讲，随便什么文本都可以，没有任何的格式要求。但是要注意“垃圾进、垃圾出”喔！因此，如果想获得比较好的模型，就务必重视预训练数据的处理工作。

**监督微调（指令微调）数据格式**：本仓库所使用的指令模板格式是`<|InstructMark|>提示语<|ResponseMark|>期望的答复<|Padding|>*`，填充至上下文长度。仓库现有的数据预处理代码，从业余无线电操作技术能力验证题库中抽取问题和正确答案，拼接成指令模板，形成SFT数据集。

**3️⃣ 预训练和监督微调**

单机单卡或CPU训练：执行`python train.py -t "pretrain" (or "sft")`即可。注意：

- 若使用CPU训练，最好将`train_config.json`中的`device`选项设为`"cpu"`。
- 对于 Jetson Orin NX 16GB、Jetson AGX Orin 64GB（均为Ampere架构，计算能力8.7）这样的比较新的设备，可以使用自动混合精度（AMP）技术和 Flash Attention 技术加速训练。在`train_config.json`中，可以将`sdp_kernel`设置为`flash`，将`dtype`设置为`bfloat16`。
- 对于P40、P100这样的比较老旧的设备，由于 compute capability 较低，不支持BF16等数据类型和flash_sdp等高效算子，训练效率可能会比较低。

单机多卡或多机多卡DDP训练：在主节点上执行以下命令（以单机4卡为例）。注意：`train_config.json`中的`gradient_accumulation_steps`应为显卡数的倍数。

```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node 4 train_ddp.py
```

基于DeepSpeed框架，使用ZeRO（零冗余优化）技术训练更大的模型。可以修改`ds_config.json`以优化训练效果。注意：根据[文档](https://www.deepspeed.ai/docs/config-json/)，`train_batch_size`必须等于`train_micro_batch_size_per_gpu` * `gradient_accumulation` * GPU数量。这里采用2节点4卡ZeRO3-Offload方式训练。

```
deepspeed train_deepspeed.py --deepspeed --deepspeed_config deepspeed_config.json --hostfile=hostfile.txt
```

其中`hostfile.txt`的内容如下：

```
192.168.10.52 slots=2
192.168.10.61 slots=2
```

**模型结构参数**

|参数|类型|默认值|说明|
|-|-|-|-|
|block_size|int|256|上下文（窗口）长度|
|vocab_size|int|10000|词典长度（实际取决于词元编码器）|
|n_layer|int|4|模型深度，即Transformer模型层数|
|n_head|int|4|注意力头数|
|n_embd|int|256|模型宽度：内部表示向量的维度|
|dropout|float|0.0|随机丢弃层的丢弃概率|
|bias|bool|False|线性变换层加偏置？|
|use_rope|bool|True|使用RoPE位置编码？反之使用训练位置编码|
|norm_eps|float|1e-5|均方根标准化层参数|
|is_causal|bool|True|因果注意力？|

**训练参数**

|参数|类型|默认值|说明|
|-|-|-|-|
|dropout|float|0.0|随机丢弃层的丢弃概率，覆盖模型参数|
|learning_rate|float|6e-4|初始学习率|
|weight_decay|float|1e-1|权重衰减|
|beta1|float|0.9|AdamW优化器的参数|
|beta2|float|0.99|AdamW优化器的参数|
|decay_lr|bool|True|学习率调节器（衰减）？|
|warmup_iters|int|300|学习率预热步数|
|lr_decay_iters|int|100000|学习率衰减步数|
|min_lr|float|6e-5|最小学习率|
|batch_size|int|100|训练批大小|
|random_seed|int|1337|随机数初始化种子|
|dataset_path|[[str, str]]|None|数据集(相对于`train.py`的路径)|
|eval_interval|int|100|验证间隔步数|
|log_interval|int|1|日志间隔步数|
|eval_iters|int|5|每次验证需要用几批数据|
|backend|str|"nccl"|分布式通信后端|
|device|str|"cuda"|计算设备|
|sdp_kernel|str|"flash"|缩放点积注意力实现|
|dtype|str|"bfloat16"|训练数据类型|
|grad_clip|float|1.0|梯度压限|
|gradient_accumulation_steps|int|4|梯度累加步数|

**4️⃣ 推理（问答式文本生成）**

如果是以DDP方式或者单机单卡或者CPU训练的模型，直接执行`python inference.py`。

如果是DeepSpeed训练的模型，则需要先执行`checkpoint/ds`目录中的转换脚本，将其转化为PyTorch能够接受的state_dict格式，再执行推理脚本：

```
cd Nano/checkpoint/ds
python zero_to_fp32.py . ckpt_ds.pt
cd Nano
python inference_ds.py
```

## 其他玩法

```
python problem.py [q|sort|palindrome]
```

**玩法1：丘成桐先生也答不出的Q问题**

所谓“Q问题”，是《鲁豫有约》20150902期节目中，主持人给丘成桐出的一道脑筋急转弯题。

![ ](./doc/q.jpg)

**玩法2：排序，但是GPT**

[B站视频](https://www.bilibili.com/video/BV1XZ421s7bM)

eg. 114515 -> 111455

**玩法3：回文序列**

eg. 123456 -> 654321

**玩法4：布尔逻辑表达式求值**

随机生成前缀式布尔逻辑表达式，表达式只含有逻辑与“*”和逻辑或“+”两个谓词。训练Transformer模型，让模型掌握布尔逻辑表达式的求值能力。例如：输入`(+ (* 0 1) (* 1 1)) =`（其中等号是求值的提示词），模型应当输出`1`。

实验记录（RoPE = 0, Causal = 1, MaxLen = Blocksize, VocabSize = 7 + 8 + (Max - Min + 1)）

|Min|Max|Depth|Block|Layer|Head|Embd|Batch|Steps| LR |GFLOPS|Loss|Acc|
|---|---|-----|-----|-----|----|----|-----|-----|----|------|----|---|
| 0 | 1 |  4  |  64 |  8  | 64 |512 | 100 |1000 |1e-3| ---- |0.23|85%|
| 0 | 1 |  4  |  64 | 10  | 64 |512 | 100 |1000 |1e-3| 1100 |0.23|85%|
| 0 | 1 |  4  |  64 | 10  |128 |512 | 100 |1000 |1e-3|  800 |0.23|88%|
| 0 | 1 |  4  |  64 | 16  | 64 |512 | 100 |1000 |1e-3| 1100 |0.23|87%|
| 0 | 1 |  4  |  64 | 10  | 64 |1024| 100 | 500 |1e-3| 1900 |0.24|83%|


## 研究笔记

### 训练性能

训练参数：BlockSize=512, VocabSize=2114, Layers=2, Heads=4, Embd=512, BatchSize=100（参数量13.67M，显存占用9045MiB）

|设备|设置|速度|
|----|----|----|
|Jetson AGX Orin (64GB)|BF16, AMP, FlashAttn|30～32TFLOPS|
|Jetson AGX Orin (64GB)|FP32, w/o AMP|8.7~8.9TFLOPS|
|Jetson Orin NX (16GB)|BF16, AMP, FlashAttn|12～13TFLOPS|
|Jetson Orin NX (16GB)|FP32, w/o AMP|3.0～3.3TFLOPS|
|单卡P40 (24GB)|FP32, w/o AMP|6.4～6.5TFLOPS|
|单卡P100 (16GB)|FP32, w/o AMP|--TFLOPS|
|双路E5-2680v4 (64GB)|FP32, w/o AMP|--GFLOPS|
|双路E5-2686v4 (128GB)|FP32, w/o AMP|550～650GFLOPS|
|Ryzen 7 5800H (16GB)|FP32, w/o AMP|200～210GFLOPS|
|Core i5-8259U (16GB)|FP32, w/o AMP|150～180GFLOPS|

### 算子`scaled_dot_product_attention`的性能

PyTorch 2.0 以上支持基于 [FlashAttention](https://arxiv.org/abs/2205.14135) 的注意力算子计算加速。目前有3种kernel，但是不支持较旧的GPU。分别启用3种kernel，实测相对性能如下：

|Kernel|flash_sdp|mem_efficient_sdp|math_sdp|
|------|------|----|--|
|相对时间|(不支持)|2.75|1(基准)|
|相对显存|(不支持)|0.78|1(基准)|

## 参考文献

- A Vaswani, N Shazeer, N Parmar, et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) [J]. Advances in Neural Information Processing Systems, 2017, 30.
- A Radford, K Narasimhan, T Salimans, et al. [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [J]. 2018.
- [GPT可视化](https://bbycroft.net/llm)
- [minimind](https://github.com/jingyaogong/minimind)
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

## 权利声明

版权所有 © 2024 BD4SUR，保留所有权利。

本系统“按原样”提供，采用MIT协议授权。本系统为作者个人以学习和自用目的所创作的作品。作者不对本系统的质量作任何承诺。作者不保证提供有关本系统的任何形式的解释、维护或支持。作者不为任何人使用此系统所造成的任何正面的或负面的后果负责。

**以部分或全部代码形式集成的开源软件**

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- [openai/tiktoken](https://github.com/openai/tiktoken)

**数据集来源**

- 精神分析黑话数据集：来自[hhiim/Lacan](https://github.com/hhiim/Lacan)。
- 业余无线电操作技术能力验证试题。
- 国际电联《无线电规则》《频谱监测手册》等。
- 中国无线电相关法规。
- 商用大模型生成的问答类内容。
- 其他公开数据集。
