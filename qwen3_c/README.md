## llama2.c

Forked from https://github.com/karpathy/llama2.c

Qwen3 inference

Export `qwen3-tokenizer.bin`:

```
python tokenizer.py
```

Export `qwen3-0b6.bin`:

```
python export.py qwen3-0b6.bin --hf /home/bd4sur/ai/_model/Qwen3/Qwen3-0.6B
```

Compile:

```
make
```

Run:

```
# Qwen3-0.6B
OMP_NUM_THREADS=4 ./qwen3 qwen3-0b6.bin qwen3-tokenizer.bin -f qwen -i "人类的本质是复读机吗？"
```

Requirements:

```
numpy==1.23.5
pytest==7.4.0
Requests==2.31.0
sentencepiece==0.1.99
torch==2.0.1
tqdm==4.64.1
wandb==0.15.5
```
