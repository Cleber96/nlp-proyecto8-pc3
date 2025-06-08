import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    """Clase envoltorio para un Transformer + capa de clasificación."""
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return (loss, logits) if loss is not None else logits