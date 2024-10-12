
# Nano：大模型，小玩具

本仓库：

- 是Transformer语言模型的极简实现，供个人赏玩、研究、魔改和炼丹炉煲机之用。
- 主要复刻自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，并借鉴了多个开源模型实现和学习项目。

期望：

- 用尽可能少的依赖，实现一个具体而微的Transformer语言模型。
- 完整实现数据预处理、词元化、预训练、监督微调、推理过程。暂不打算实现高效微调（如LoRA）、人类对齐等过程。
- 在廉价硬件上从头训练一个[千万量级参数的模型](https://huggingface.co/bd4sur/ham_radio_nanolm_0930)（[B站视频](https://www.bilibili.com/video/BV1MtxteUEge)）。
- 研究模型训练的动力学、训/推加速、算法改进等问题。
- 探索Transformer模型在自然语言处理以外的问题和模态上的潜能。

建议读者：

- 对语言模型的能力有合乎理性的预期，认识到大语言模型（的生产）是昂贵而复杂的系统工程。
- 准备好计算资源，做好运维保障。大模型是个非常烧钱的东西，资源虽说多多益善，但终究还是丰俭由人。

为什么叫“Nano”：

- 東雲なの（Shinonome **Nano**）和坂本是动画《日常》的角色。なの是博士创造的女高中生机器人，而坂本是一只会说话的黑猫。
- 致(chao)敬(xi) Karpathy大佬的nanoGPT项目。

![ ](./doc/nano.jpg)

## 全流程速通

[B站视频](https://www.bilibili.com/video/BV1uv42127qP)

- 全流程速通，旨在无需设置任何参数即可“开箱即用”地跑通全部流程，直至训练出一个能用的语言模型。
- 如果只是想体验推理效果，可直接执行第1、4步骤。
- 尽管全流程速通不需要对Nano有任何了解，但我还是建议你对于深度学习的环境搭建和配置有一定的经验。

**1️⃣ 安装依赖**：建议在conda虚拟环境中安装。

```
conda create -n nano python=3.11 pysocks -y
conda activate nano
python -m pip install -r requirements.txt
```

**2️⃣ 数据下载·预处理**

- 下载本人整理的[数据集](https://huggingface.co/bd4sur/nano_dataset)，解压口令“`nano`”。
- 解压得到`pretrain.txt`和`sft.jsonl`两个文件，将这两个文件移动到`dataset`目录下。
- 执行`python data.py`，进行数据预处理。

**3️⃣ 预训练和监督微调**

开始训练之前，请先确认几件事：

- 训练可能耗费几小时乃至几天的时间！具体时间取决于训练设置和硬件。
- 建议使用nvidia计算卡，以计算能力7.0+为宜，如V100、2080ti、3090、4090、A100等，如果有多卡甚至多机多卡环境最好。
- 若使用CPU训练，将`config_pretrain/sft.json`中的`device`字段设为`"cpu"`。
- 若使用P40、P100等老旧设备，将`config_pretrain/sft.json`中的`sdp_kernel`字段设为`"math"`。
- 首次运行，建议使用单机单卡或CPU进行验证性训练。若使用多机分布式训练，请先提前配置好分布式环境，例如无密码ssh认证等。

**预训练**：执行`python train.py -t pretrain`。请注意：

- 训练没有最大步数限制。因此，需要自行决定何时中止训练。
- 建议训练至少10轮（epoch），最好不要少于1轮，保证模型“见过”全部语料。
- 支持保存模型检查点。训练过程中，程序将按照模型保存策略，保存模型训练检查点到`checkpoint`目录。保存策略主要有三点：一是根据训练配置文件中规定的间隔，每隔一定的步数保存一个检查点；二是只有当验证集损失下降才会保存检查点；三是每隔1000步定期保存一次检查点。优先级：策略3 > 策略2 > 策略1。
- 支持断点续训。如果预训练意外中止，可以将`config_pretrain.json`中的`from_checkpoint`字段设为上一个检查点的相对路径`"checkpoint/xxx.pt"`，然后重新启动训练。
- 支持训练过程监控。每次训练，程序都会记录一个新的训练日志文件`train_xxx.log`，位于仓库根目录。执行`python plot_loss.py -n train_xxx.log`，绘制训练集损失曲线。

**监督微调**：首先将`config_sft.json`中的`from_checkpoint`字段设为预训练模型的相对路径`"checkpoint/xxx.pt"`。然后执行`python train.py -t sft`。其余与预训练类似。需要指出的是，监督微调的训练轮数，应当根据实际情况灵活选择。一般来说，如果训练轮数过少，模型可能难以学习到指令跟随能力。而训练轮数过多，则可能遗忘预训练过程中获得的语言能力，以及在监督微调数据集上过拟合。

**4️⃣ 推理**

如果只是想体验推理效果而不训练，首先下载[预训练或指令微调模型](https://huggingface.co/bd4sur/nano-1010)到`checkpoint`目录。

执行`python inference.py -i -m checkpoint/xxx.pt`，其中`xxx.pt`是模型检查点文件（下载的或者自己训练的）。可选的命令行参数如下：

- `-m` or `--model`：字符串，模型相对路径。
- `-i` or `--instruct`：开关标识。若启用，则对输入套用指令模板，以支持指令微调模型上的指令问答；若不启用，则为自回归式文本生成。
- `-l` or `--max_length`：整数，序列最大长度，默认为模型的上下文窗口长度。
- `-t` or `--temperature`：浮点数，生成温度参数，默认值为1.0，越高则生成越随机。
- `-k` or `--top_k`：整数，前k采样，默认值为5，越高则生成越多样。
- `-r` or `--repetition_penalty`：浮点数，复读惩罚，默认值为1.2，越大则越抑制生成重复的词元。

## 技术要点简述

**数据预处理**

- 包含文本分块、数据集划分、随机打乱、SFT模板组装、词元化等处理步骤。
- 在默认实现中，数据集划分实际上并未严格隔离训练集和验证集，验证集是从训练集中简单抽取5%得到。

**词元编码**

- Nano使用最简单的词元编码算法，也就是给语料中所有包含的独立Unicode字符，赋予唯一整数编号，作为词元编号。因此词元实际上就等于是Unicode字符。
- 仓库中同时包含了tiktoken提供的一个BPE词元编码算法，由于速度很慢，并不实用，因此仅作为文档，并不实际使用。
- 之所以不使用额外的词元编码工具，例如tiktoken、Tokenizers等，一方面是为了最小化外部依赖，另一方面也是想探索不含（高效）词元编码的语言模型效果如何。

**预训练数据格式**

- 原则上讲，随便什么文本都可以，没有任何的格式要求。
- 建议在独立文章的前后加上定界用的特殊词元`<|bos|>`和`<|eos|>`。这有助于避免训练时将不相关的文本混淆到同一个上下文窗口中（目前暂未实现）。
- 但是要注意“垃圾进、垃圾出”喔！因此，如果想获得比较好的模型，就务必重视预训练数据的处理工作。

**监督微调（指令微调）数据格式**

- Nano指令模板格式：`<|InstructMark|>提示语<|ResponseMark|>答复<|eos|><|Padding|>*`，填充至上下文长度。
- SFT数据集是JSONL格式，每一行是一轮QA，格式为`{"question": "提示语", "answer": "答复"}`，在数据预处理阶段转换为指令模板的格式。
- Nano现在不支持多轮对话，因多轮对话在原理上与单轮对话的SFT没有本质区别。后续可能会支持。

**Transformer模型结构**

- 模型结构以GPT（[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)）为主要参考，加入了RoPE位置编码（同时兼容原有的训练位置编码），将LayerNorm替换为RMSNorm。
- 可选择因果自注意力或完全的自注意力，前者用于语言模型，后者用于在其他任务上的探索。
- 词元嵌入层（`wte`）与解码层（`lm_head`）共享权重。关于这个问题，可参考文献[]。
- 后续计划加入分组查询注意力（GQA）和KV-Cache。

模型结构参数`model_config.json`：

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

**模型训练**

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

**分布式训练**

Nano现在支持分布式数据并行（DDP）训练和基于DeepSpeed的零冗余优化（ZeRO）训练。

DDP训练：以单机4卡为例，在主节点上执行以下命令。注意：`train_config.json`中的`gradient_accumulation_steps`应为显卡数的倍数。

```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node 4 train_ddp.py
```

基于DeepSpeed框架，使用ZeRO技术训练：以2节点4卡ZeRO3-Offload方式为例，在主节点上执行以下命令。可以修改`ds_config.json`以调整ZeRO设置。注意：根据[文档](https://www.deepspeed.ai/docs/config-json/)，`train_batch_size`必须等于`train_micro_batch_size_per_gpu` * `gradient_accumulation` * GPU数量。

```
deepspeed train_deepspeed.py --deepspeed --deepspeed_config deepspeed_config.json --hostfile=hostfile.txt
```

其中`hostfile.txt`的内容如下：

```
192.168.10.52 slots=2
192.168.10.61 slots=2
```

推理阶段注意：如果是DeepSpeed训练的模型，则需要先执行`checkpoint/ds`目录中的转换脚本，将其转化为PyTorch能够接受的state_dict格式，再执行推理脚本：

```
cd Nano/checkpoint/ds
python zero_to_fp32.py . ckpt_ds.pt
cd Nano
python inference_ds.py
```

**解码策略**

- Nano采用基于温度的采样策略，结合top-k采样和重复惩罚机制，从语言模型中自回归地采样出词元序列。
- Nano同时提供序列到序列的（非自回归）推理，用于NLP以外的其他问题的研究。

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

实验记录（RoPE = 0, Causal = 1, VocabSize = 7 + 8 + (Max - Min + 1), LR_Decay = 1）

Min = 0, Max = 1, Depth = 4, BlockSize = MaxLen = 64

|Layer|Head|Embd|Batch|Steps| LR |GFLOPS|Loss|Acc|
|-----|----|----|-----|-----|----|------|----|---|
|  8  | 64 |512 | 100 |1000 |1e-3| ---- |0.23|85%|
| 10  | 64 |512 | 100 |1000 |1e-3| 1100 |0.23|85%|
| 10  |128 |512 | 100 |1000 |1e-3|  800 |0.23|88%|
| 16  | 64 |512 | 100 |1000 |1e-3| 1100 |0.23|87%|
| 10  | 64 |1024| 100 | 500 |1e-3| 1900 |0.24|83%|


## 动力学研究

### 模型质量度量

- 预训练损失：下一词元预测序列的交叉熵损失。
- 指令微调损失：带掩模的交叉熵损失。

通用评测：C-Eval等。

### 尺度缩放定律（Scaling law）

2024年10月8日

|Block|Vocab|Layer|Head|Embd|RoPE|
|-----|-----|-----|----|----|----|
| 256 |32768|  8  | 64 |512 |True|

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
