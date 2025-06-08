import pytest
import os
import pandas as pd
from src.data.loader import load_raw_data, load_processed_hf, load_torch_tensors

def test_load_raw_data(tmp_path):
    # Crear archivos CSV de ejemplo
    df = pd.DataFrame({'sentence1': ['a'], 'sentence2': ['b'], 'label': [1]})
    for split in ['train', 'validation', 'test']:
        df.to_csv(tmp_path / f"mrpc_{split}.csv", index=False)
    data = load_raw_data(str(tmp_path))
    assert set(data.keys()) == {'train', 'validation', 'test'}
    assert isinstance(data['train'], pd.DataFrame)
    assert 'sentence1' in data['train'].columns

def test_load_processed_hf(tmp_path):
    from datasets import DatasetDict, Dataset
    # Crear dataset de ejemplo y guardarlo con save_to_disk
    ds = DatasetDict({
        'train': Dataset.from_dict({'sentence1': ['a'], 'sentence2': ['b'], 'label': [0]}),
        'validation': Dataset.from_dict({'sentence1': ['c'], 'sentence2': ['d'], 'label': [1]}),
        'test': Dataset.from_dict({'sentence1': ['e'], 'sentence2': ['f'], 'label': [0]})
    })
    out = tmp_path / 'hf'
    ds.save_to_disk(str(out))
    loaded = load_processed_hf(str(out))
    assert hasattr(loaded, 'keys')
    assert 'train' in loaded

def test_load_torch_tensors(tmp_path):
    import torch
    # Crear tensores de ejemplo y guardarlos
    dummy = [{'input_ids': torch.tensor([1]), 'attention_mask': torch.tensor([1]), 'label': torch.tensor(0)}]
    torch.save(dummy, tmp_path / 'train_tokenized.pt')
    torch.save(dummy, tmp_path / 'val_tokenized.pt')
    torch.save(dummy, tmp_path / 'test_tokenized.pt')
    loaded = load_torch_tensors(str(tmp_path))
    assert 'train' in loaded and isinstance(loaded['train'], list)