import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from src.data.tokenizer import get_tokenizer, tokenize_dataset
from src.models.model import TransformerClassifier
from src.training.train_utils import compute_metrics


def main():
    model_name = os.getenv("MODEL_NAME", "distilbert-base-uncased")
    num_labels = int(os.getenv("NUM_LABELS", "2"))
    output_dir = os.getenv("OUTPUT_DIR", "outputs/models/finetuned")

    tokenizer = get_tokenizer(model_name)
    dataset = load_dataset("glue", "mrpc")
    tokenized = tokenize_dataset(dataset, tokenizer)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = TransformerClassifier(model_name, num_labels)
    args = TrainingArguments(
        output_dir=output_dir,
        eval_steps=500,  # Evaluar cada 500 pasos
        save_steps=500,  # Guardar el modelo cada 500 pasos
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()