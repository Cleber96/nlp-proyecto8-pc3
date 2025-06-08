from transformers import AutoTokenizer
from datasets import DatasetDict

def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Devuelve un tokenizador pre-entrenado de Hugging Face."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int = 128):
    """Aplica tokenización a todas las particiones de un DatasetDict."""
    def preprocess(batch):
        return tokenizer(
            batch["sentence1"], batch["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    tokenized = dataset.map(preprocess, batched=True)
    return tokenized