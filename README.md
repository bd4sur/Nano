
# Nano：大模型，小玩具

**Pre-alpha · 正在积极开发**

**Nano**是Transformer语言模型的极简实现，供个人赏玩、研究、魔改和炼丹炉煲机之用。主要复刻自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，并借鉴了多个开源模型实现和学习项目。

https://github.com/user-attachments/assets/94181a6e-6016-42e2-b617-f8d6cbeb35ab

期望：

- 用尽可能少的依赖，尤其不依赖🤗，实现一个具体而微的Transformer语言模型。
- 完整实现数据处理、预训练、监督微调（含LoRA）、推理过程。暂不实现人类对齐。
- 从头训练一个会说人话的50M级参数规模的语言模型。
- 实现基于Web浏览器的、完全离线的端侧推理（[在线体验](https://bd4sur.com/Nano/infer)）。
- 研究模型训练的动力学、训/推加速、算法改进等问题。
- 探索Transformer模型在自然语言处理以外的问题和模态上的潜能。
- 建立起关于大语言模型的合理预期和感性经验，对大语言模型技术祛魅。

为什么叫“Nano”：

- 東雲なの（Shinonome **Nano**）和坂本是动画《日常》的角色。なの是博士创造的女高中生机器人，而坂本是一只会说话的黑猫。
- 致(chao)敬(xi) Karpathy大佬的nanoGPT项目。

![ ](./doc/nano.jpg)

## 模型和数据

|预训练模型|预训练数据|指令微调模型|指令微调数据|
|---------|---------|-----------|-----------|
|[Nano-56M](https://huggingface.co/bd4sur/Nano-56M)|[Nano-PT-10B](https://huggingface.co/datasets/bd4sur/Nano-PT-10B)|[Nano-56M-Instruct](https://huggingface.co/bd4sur/Nano-56M-Instruct)|Nano-SFT|

数据集为7z格式，解压口令“nano”。

## 全流程速通

[B站视频](https://www.bilibili.com/video/BV1uv42127qP)

- 全流程速通，旨在“开箱即用”地跑通推理乃至训练流程。
- 如果只是想体验推理效果，可直接执行第1、4步骤。

### 1️⃣ 安装依赖

- 硬件：建议使用英伟达GPU，以计算能力7.0以上的为宜，详见[英伟达官网](https://developer.nvidia.com/cuda-gpus)。若只有CPU也无妨。
- 软件：建议使用Ubuntu等Linux操作系统，并且在conda虚拟环境中安装依赖：

```
conda create -n nano python=3.11 pysocks -y
conda activate nano
python -m pip install -r requirements.txt
```

### 2️⃣ 数据下载·预处理

如果只是想跑通流程：

- 直接执行`python data.py`，对预置的精神分析黑话语料作预处理。

如果真的想让模型学会说人话：

- 下载预训练数据集和指令微调数据集。
- 解压得到`pretrain.txt`和`sft.jsonl`两个文件，移动到`dataset`目录下。
- 将`data.py`中`PRETRAIN_DATASETS`和`SFT_DATASET`替换为刚刚下载的两个文件。
- 执行`python data.py`，进行数据预处理。可能占用大量记忆和存储空间，请提前预留。

### 3️⃣ 预训练和监督微调

如果只是想跑通流程：

- 将`config_pretrain.json`中`dataset_path`和`tokenizer_path`改成实际的绝对路径。
- 直接执行`bash start_pretrain.sh`。如果使用Windows，则执行`python train.py -m model_config.json -t config_pretrain.json`。

如果真的想让模型学会说人话，开始训练之前，请先确认几件事：

- 训练可能耗费几小时乃至几天的时间！具体时间取决于训练设置和硬件，参考下文。
- 若长时间训练，**强烈建议使用 [GNU Screen](https://www.gnu.org/software/screen/) 等终端切换工具，保证训练进程不被意外杀掉**。
- 若使用多机分布式训练，请先提前配置好分布式环境，例如无密码ssh认证等。

> 简单估算训练时间：对58M参数的语言模型(L=16, H=16, E=512, VocabSize=512)作预训练，按照[文献](https://arxiv.org/abs/2204.02311)中提供的算法进行计算，每个词元所需计算量约为403MFlop。如果使用10亿(即1B=1e9)词元的语料进行一轮(epoch)预训练，则总计算量约为403PFlop。实际使用单卡A100进行训练，**实测耗时约5200秒（1.44小时）**，对应运算速度为78TFlop/s，是A100标称BF16算力312TFlop/s的25%，也即MFU为25%左右。

**预训练**：

将`model_config.json`中的模型参数设置为：

```json
"block_size": 512,
"vocab_size": 16384,
"n_layer": 16,
"n_head": 16,
"n_embd": 512,
"dropout": 0.0,
"bias": false,
"use_rope": true,
"norm_eps": 1e-5,
"is_causal": true
```

将`config_pretrain.json`中的`batch_size`设置为一个能够充分利用显存的值。对于 AGX Orin (64GB)，可设置为160。

单机单卡或者CPU训练，执行`bash start_pretrain.sh`。

单机多卡或者多机多卡分布式数据并行（DDP）训练，在主节点上执行`bash start_pretrain_ddp.sh`，注意修改脚本中的卡数。

请注意：

- 训练没有最大步数限制。因此，需要自行决定何时中止训练。
- 建议训练至少10轮（epoch），最好不要少于1轮，保证模型“见过”全部语料。
- 如果使用DDP训练，`gradient_accumulation_steps`应设置为显卡数的整数倍。
- 支持保存模型检查点。训练过程中，程序将按照模型保存策略，保存模型训练检查点到`checkpoint`目录。保存策略主要有三点：一是根据训练配置文件中规定的间隔，每隔一定的步数保存一个检查点；二是只有当验证集损失下降才会保存检查点；三是每隔1000步定期保存一次检查点。优先级：策略3 > 策略2 > 策略1。
- 支持断点续训。如果预训练意外中止，可以将`config_pretrain.json`中的`from_checkpoint`字段设为上一个检查点的相对路径`"checkpoint/xxx.pt"`，然后重新启动训练。
- 支持训练过程监控。每次训练，程序都会记录一个新的训练日志文件`train_xxx.log`，位于仓库根目录。执行`python plot_loss.py -n train_xxx.log`，绘制训练集损失曲线。

**监督微调（全参数）**：首先将`config_sft.json`中的`from_checkpoint`字段设为预训练模型的相对路径`"checkpoint/xxx.pt"`。然后执行`bash start_sft.sh`（或者`bash start_sft_ddp.sh`）。其余与预训练类似。需要指出的是，监督微调的训练轮数，应当根据实际情况灵活选择。一般来说，如果训练轮数过少，模型可能难以学习到指令跟随能力。而训练轮数过多，则可能遗忘预训练过程中获得的语言能力，以及在监督微调数据集上过拟合。

**监督微调（LoRA）**：TODO

### 4️⃣ 推理

**方式一：浏览器推理**

- 访问[在线体验页面](https://bd4sur.com/Nano/infer)，或者用浏览器直接打开`Nano/infer/index.html`，按页面提示打开本地预先下载好的基座模型文件（扩展名为bin），模型下载地址见上文。
- 可切换文本续写模式和指令问答模式，默认后者。推荐使用指令微调后的模型，在指令问答模式下体验。
- 可随时加载或卸载LoRA插件。注意LoRA插件需要与某个预训练基座模型匹配。
- 使用`export.py`将检查点文件转换为基座模型或者LoRA插件。
- 所有推理过程均在本地浏览器内部进行。

**方式二：PyTorch推理**

如果只是想体验推理效果而不训练，首先下载[预训练或指令微调模型](https://huggingface.co/bd4sur/nano-1010)到`checkpoint`目录。

执行`python inference.py -i -m checkpoint/xxx.pt`，其中`xxx.pt`是模型检查点文件（下载的或者自己训练的）。可选的命令行参数如下：

- `-m` or `--model`：字符串，模型相对路径。
- `-i` or `--instruct`：开关标识。若启用，则对输入套用指令模板，以支持指令微调模型上的指令问答；若不启用，则为自回归式文本生成。
- `-l` or `--max_length`：整数，序列最大长度，默认为模型的上下文窗口长度。
- `-t` or `--temperature`：浮点数，生成温度参数，默认值为1.0，越高则生成越随机。
- `-k` or `--top_k`：整数，前k采样，默认值为5，越高则生成越多样。
- `-r` or `--repetition_penalty`：浮点数，复读惩罚，默认值为1.2，越大则越抑制生成重复的词元。
- `-p` or `--profile`：开关标识。若启用，则统计性能数据，包括首词元延迟、词元生成速率等。

## 技术要点简述

![ ](doc/nano-llm.png)

**数据预处理**

- 包含文本分块、词元编码、数据集划分、随机打乱、SFT模板组装等处理步骤。
- 在默认实现中，数据集划分实际上并未严格隔离训练集和验证集，验证集是从训练集中简单抽取5%得到。

**词元编码**

- Nano使用最简单的词元编码算法，也就是给语料中所有包含的独立Unicode字符，赋予唯一整数编号，作为词元编号。因此词元实际上就等于是Unicode字符。
- 为了提升英文编码效率，在词表中手工添加了部分英文单词。
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

- 模型结构以Llama2和GPT（[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)）为主要参考。
- 使用RoPE位置编码（可选用训练位置编码）和前置RMSNorm。
- 使用分组查询注意力（GQA）。
- 使用SwiGLU，参考[文献](https://arxiv.org/pdf/2002.05202)。
- 可选择因果自注意力或完全的自注意力，前者用于语言模型，后者用于在其他任务上的探索。
- 词元嵌入层与分类层共享权重。关于这个问题，可参考[文献](https://spaces.ac.cn/archives/9698)。
- 支持KV-Cache。
- 支持插件化的低秩适配（LoRA）训练和推理。

模型结构参数`model_config.json`：

|参数|类型|默认值|说明|
|-|-|-|-|
|block_size|int|256|上下文（窗口）长度|
|vocab_size|int|10000|词典长度（实际取决于词元编码器）|
|n_layer|int|4|模型深度，即Transformer模型层数|
|n_head|int|4|Q注意力头数|
|n_kv_head|int|4|KV注意力头数|
|n_embd|int|256|模型宽度：内部表示向量的维度|
|dropout|float|0.0|随机丢弃层的丢弃概率|
|bias|bool|False|线性变换层加偏置？|
|use_rope|bool|True|使用RoPE位置编码？反之使用训练位置编码|
|norm_eps|float|1e-5|均方根标准化层参数|
|is_causal|bool|True|因果注意力？|

**模型训练**

- `from_checkpoint: str`：从哪个检查点继续训练。其值是绝对路径。说明：**训练选项中涉及的所有路径，都是绝对路径**。
- `save_checkpoint_to: str`：检查点保存位置的绝对路径。其值必须是目录。默认值为仓库根目录下`checkpoint/`目录。
- `dataset_path: [[str, str], ...]`：预处理后的数据集的绝对路径。该字段的值为列表，列表的每一项都是含有两个元素的子列表，子列表的第一个元素是训练集的绝对路径，第二个元素是验证集的绝对路径。
- `tokenizer_path: str`：词表绝对路径。默认值为仓库根目录下`tokenizer/tokenizer_16384.json`。
- `random_seed: int`：Torch的随机数种子。默认值为39。固定这个值，便于复现特定结果，利于调试。
- `batch_size: int`：训练批大小。默认值为32。一般来说，批大小越大，越有利于模型收敛，也更能充分利用算力资源。但代价是成倍消耗显存。如果启用梯度累加，则实际等效批大小为`batch_size`乘以`gradient_accumulation_steps`。
- `gradient_accumulation_steps: int`：梯度累加步数。默认值：1。在DDP场景下，梯度累积步数必须是GPU卡数的整数倍。梯度累加技术可以在有限的批次大小上模拟以较大批大小训练的效果，其原理是以时间换空间，根据偏导数的加法分配律，将几个小批次上多步迭代得到的梯度进行累加，使用累加后的梯度一次性更新参数，达到模拟较大批次的效果。
- `grad_clip: float`：梯度压限系数，用于防止梯度爆炸。默认值：1.0。
- `dropout: float`：随机丢弃层的丢弃概率，仅在训练阶段有效。默认值：0。预训练阶段一般设置为0，微调阶段一般为非0。
- `learning_rate: float`：初始学习率。默认值：6e-4。
- `weight_decay: float`：权重衰减系数。默认值：1e-1。
- `beta1: float`：AdamW优化器参数，详见[文档](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。默认值：0.9。
- `beta2: float`：AdamW优化器参数，详见[文档](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。默认值：0.99。
- `decay_lr: bool`：是否启用学习率调度？若不启用，则为恒定学习率。默认值：true。
- `warmup_iters: int`：学习率预热阶段的步数，仅当启用学习率调度时有效。默认值：10000。
- `lr_decay_iters: int`：学习率调度的总步数，仅当启用学习率调度时有效。默认值：1e9。
- `min_lr: float`：最小学习率，仅当启用学习率调度时有效。默认值：6e-5。
- `eval_interval: int`：每隔几步在验证集上计算一次损失。默认值：100。说明：如果满足检查点保存条件，将保存检查点。
- `log_interval: int`：每隔几步打印一次日志。默认值：10。注意：打印日志会计算损失值，比较耗时，因此不建议过于频繁地打印日志。
- `eval_iters: int`：每次验证需要用几批数据。默认值：5。
- `backend: str`：分布式通信后端。可选值：`nccl`等。用于DDP。
- `device: str`：计算设备。可选值：`cuda`、`cuda:x`用于指定某个GPU、`cpu`、`mps`等。一般无需特别设置，除非：①设备无显卡，将自动回落到CPU；②DDP模式下将自动设置为某一块GPU。
- `sdp_kernel: str`：缩放点积注意力的实现。可选值：`math`基础、`flash`高效（默认）、`mem_efficient`节省显存。其中`flash`仅支持FP16和BF16两种输入精度，且可能存在其他限制条件。
- `dtype: str`：训练数据类型。可选值：`float32`单精度（E8M23）、`float16`半精度（E5M10）、`bfloat16`半精度（E8M7，默认）。一般而言，若使用Ampere及以上的GPU架构，建议使用BF16。
- `use_amp: bool`：是否使用自动混合精度技术？仅当`dtype`设置为FP16和BF16时，才支持AMP。一般而言，启用AMP可节约显存占用，同时有助于训练稳定和收敛，也能够充分利用半精度运算所带来的速度增益。但是笔者实测发现，在 AGX Orin 和 Orin NX 等Ampere架构的GPU上，关闭AMP并使用BF16数据类型，性能更高，但代价是损失数值计算精度，可能带来模型难以收敛的风险。若AMP开启，默认同时启用TF32支持，以提升32位浮点数的运算性能。

**分布式训练**

Nano支持基于DeepSpeed的零冗余优化（ZeRO）训练。以2节点4卡ZeRO3-Offload方式为例，在主节点上执行以下命令。可以修改`ds_config.json`以调整ZeRO设置。注意：根据[文档](https://www.deepspeed.ai/docs/config-json/)，`train_batch_size`必须等于`train_micro_batch_size_per_gpu` * `gradient_accumulation` * GPU数量。

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

**2024-10-14 预训练**

|Block|Vocab|Layer|Head|Embd|RoPE|Batch| GA |  dtype  |
|-----|-----|-----|----|----|----|-----|----|---------|
| 512 |16384|  16 | 16 |512 |True| 220 | 1  | BF16 AMP|

- 设备：租用单卡A800-80GB-PCIe
- 软件：CUDA 12.4 / PyTorch 2.3.0
- 显存占用：71.6GB
- 平均FLOPS：79TFLOPS
- 平均吞吐率：193k tokens/s

**2024-10-14 监督微调**

|Block|Vocab|Layer|Head|Embd|RoPE|Batch| GA |  dtype  |
|-----|-----|-----|----|----|----|-----|----|---------|
| 512 |16384|  16 | 16 |512 |True| 16  | 1  | BF16 AMP|

- 设备：Jetson Orin NX 16GB (MAXN)
- 软件：CUDA 12.2 / PyTorch 2.3.0
- 显存占用：6.0GB
- 平均FLOPS：3.2TFLOPS
- 平均吞吐率：8k tokens/s

**过往实验数据**

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
