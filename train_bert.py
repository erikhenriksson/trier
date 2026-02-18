"""
ModernBERT register label classifier.
Reads simple jsonl files with keys "text" and "register".

Usage:
    python train_classifier.py --train splits/train.jsonl \
                               --dev   splits/dev.jsonl   \
                               --test  splits/test.jsonl
"""

import argparse
import json

import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "answerdotai/ModernBERT-large"
MAX_LEN = 512
SEED = 42


def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            examples.append((obj["text"], obj["register"]))
    return examples


def build_label_maps(examples):
    labels = sorted({label for _, label in examples})
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return l2i, i2l


def to_hf_dataset(examples, l2i, tokenizer):
    texts = [t for t, _ in examples]
    labels = [l2i[l] for _, l in examples]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)
    enc["labels"] = labels
    return Dataset.from_dict(enc)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output_dir", default="./register_model")
    args = parser.parse_args()

    print("Loading data...")
    train_raw = load_jsonl(args.train)
    dev_raw = load_jsonl(args.dev)
    test_raw = load_jsonl(args.test)

    l2i, i2l = build_label_maps(train_raw)
    print(f"Labels ({len(l2i)}): {l2i}")
    print(f"Train: {len(train_raw)}  Dev: {len(dev_raw)}  Test: {len(test_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(l2i)
    )

    train_ds = to_hf_dataset(train_raw, l2i, tokenizer)
    dev_ds = to_hf_dataset(dev_raw, l2i, tokenizer)
    test_ds = to_hf_dataset(test_raw, l2i, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nTraining...")
    trainer.train()

    for split_name, ds, raw in [("DEV", dev_ds, dev_raw), ("TEST", test_ds, test_raw)]:
        preds_out = trainer.predict(ds)
        preds = np.argmax(preds_out.predictions, axis=-1)
        true = [l2i[l] for _, l in raw]
        target_names = [i2l[i] for i in sorted(i2l)]
        print(f"\n{'=' * 50}\n{split_name} RESULTS\n{'=' * 50}")
        print(classification_report(true, preds, target_names=target_names, digits=4))


if __name__ == "__main__":
    main()
