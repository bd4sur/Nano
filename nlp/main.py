import os
import json
import torch
from data import generate_text, generate_problem_q, generate_sorting
from train import TrainGPT
from inference import InferenceGPT

CONFIG_JSON = "train_nlg.json"

def main():
    print(f"PyTorch version: {torch.__version__}")

    generate_text("psycho.txt", data_dir="dataset", block_size=128, overlap_ratio=0.1)
    # generate_problem_q("dataset")
    # generate_sorting("dataset")

    with open(os.path.join(os.path.dirname(__file__), CONFIG_JSON), "r", encoding="utf-8") as f:
        config = json.load(f)
        trainer = TrainGPT(config, max_iters=1000)
        trainer.start()

    infer = InferenceGPT()
    infer.inference_nlg()
    # infer.inference_q()
    # infer.inference_sorting()

if __name__ == "__main__":
    main()