import os
import pandas as pd
from datasets import load_from_disk
import torch

def load_raw_data(raw_dir: str):
    """Carga datos crudos desde CSV en raw_dir y retorna un dict de DataFrames."""
    data = {}
    for split in ["train", "validation", "test"]:
        path_csv = os.path.join(raw_dir, f"mrpc_{split}.csv")
        data[split] = pd.read_csv(path_csv)
    return data


def load_processed_hf(proc_dir: str):
    """Carga dataset tokenizado guardado con Hugging Face save_to_disk."""
    return load_from_disk(proc_dir)


def load_torch_tensors(tensor_dir: str):
    """Carga splits guardados como archivos .pt y retorna un dict de tensores."""
    return {
        "train": torch.load(os.path.join(tensor_dir, "train_tokenized.pt")),
        "validation": torch.load(os.path.join(tensor_dir, "val_tokenized.pt")),
        "test": torch.load(os.path.join(tensor_dir, "test_tokenized.pt"))
    }