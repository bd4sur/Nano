import torch
from tokenizer import Tokenizer

MODEL_VERSION  = "2024.10"
input_path     = "/home/bd4sur/ai/Nano/checkpoint/nano_56m_20241027_sft_118000.pt"
output_path    = "/home/bd4sur/ai/Nano/checkpoint/nano_56m_20241027_sft_118000_new.pt"
tokenizer_path = "/home/bd4sur/ai/Nano/tokenizer/tokenizer_16384.json"

ckpt = torch.load(input_path, map_location="cuda")

tokenizer = Tokenizer()
tokenizer.load_from_config_file(tokenizer_path)

new_ckpt = {
    "version":          MODEL_VERSION,
    "is_lora":          False,
    "model":            ckpt["model"],
    "optimizer":        ckpt["optimizer"],
    "step_count":       ckpt["step_count"],
    "train_config":     ckpt["train_config"],
    "model_config":     ckpt["model_config"],
    "tokenizer_config": tokenizer.config
}

torch.save(new_ckpt, output_path)

print("Done.")
