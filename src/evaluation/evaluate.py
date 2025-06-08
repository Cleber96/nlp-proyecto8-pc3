import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score


def evaluate_model(model, tokenizer_name: str = "distilbert-base-uncased"):
    """Evalúa un modelo en el split de validación MRPC y retorna accuracy."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("glue", "mrpc", split="validation")
    def preprocess(batch):
        return tokenizer(
            batch["sentence1"], batch["sentence2"], truncation=True, padding="max_length"
        )
    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model.eval()
    preds, labels = [], []
    for batch in ds:
        inputs = {k: batch[k].unsqueeze(0) for k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs
        preds.append(logits.argmax(dim=-1).item())
        labels.append(batch["label"].item())
    acc = accuracy_score(labels, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    return acc