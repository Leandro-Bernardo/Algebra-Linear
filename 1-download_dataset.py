from datasets import load_dataset
from huggingface_hub import login
import os

# Login opcional
token = input("Token do Hugging Face (pressione Enter para continuar sem login):\n").strip()

if token:
    login(token)

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "Raw_dataset")
dataset = load_dataset("project-oceania/planktonzilla-17M")#, split="train[:5000000]")
dataset.save_to_disk(save_dir)
