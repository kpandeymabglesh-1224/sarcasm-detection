"""Pipeline B: fine-tuned DistilBERT (reconstructed - no original code survives).

    python -m src.distilbert_model --smoke   # ~1 min sanity check on a small subset
    python -m src.distilbert_model           # full run (use this for the comparison)

Settings below are standard DistilBERT fine-tuning choices, NOT recovered from
the original project (slide 4 only says "pre-trained DistilBERT, fine-tuned on
the News Headlines dataset"):
  - model: distilbert-base-uncased (headlines are all lowercase, so the uncased
    model loses nothing)
  - input: the RAW headline. No stopword removal / punctuation stripping as in
    the Stacked ML pipeline: DistilBERT needs words like "not" and full word
    order, which is the whole point of slide 6.
  - max_length 64 (95% of headlines are <=16 words; only one outlier is cut)
  - 3 epochs, lr 2e-5, batch 32, weight decay 0.01, 10% warmup, fp16 on GPU
  - 10% of the TRAIN set is held out as validation to pick the best epoch, so
    the test set is used exactly once, at the end.
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

from src.config import MODELS_DIR, SEED
from src.data import get_split, load_dataset
from src.metrics import evaluate

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64


class HeadlineDataset(torch.utils.data.Dataset):
    """Tokenised headlines + labels; padding is done per batch by the collator."""

    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def val_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset, 1 epoch: checks the code runs end to end")
    ap.add_argument("--model", default=MODEL_NAME, help="HF model name or local folder")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    set_seed(SEED)
    use_gpu = torch.cuda.is_available()
    print(f"Device: {torch.cuda.get_device_name(0) if use_gpu else 'CPU'}")

    train_df, test_df = get_split(load_dataset(dedupe=True))
    tr_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED,
                                     stratify=train_df["label"])
    name = "distilbert"
    if args.smoke:
        tr_df, val_df, test_df = tr_df[:512], val_df[:128], test_df[:256]
        args.epochs, name = 1, "distilbert_smoke"
    print(f"train {len(tr_df)}, val {len(val_df)}, test {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2,
        id2label={0: "not sarcastic", 1: "sarcastic"},
        label2id={"not sarcastic": 0, "sarcastic": 1})

    train_ds = HeadlineDataset(tr_df["text"], tr_df["label"], tokenizer)
    val_ds = HeadlineDataset(val_df["text"], val_df["label"], tokenizer)
    test_ds = HeadlineDataset(test_df["text"], test_df["label"], tokenizer)

    out_dir = MODELS_DIR / name
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0.1,                 # float < 1 = ratio of total steps (transformers 5.x)
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,      # keep the epoch with best validation F1
        metric_for_best_model="f1",
        fp16=use_gpu,                     # half precision: faster, less VRAM
        logging_steps=50,
        dataloader_num_workers=0,         # safest on Windows
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=val_metrics,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f} s")

    logits = trainer.predict(test_ds).predictions
    evaluate(test_df["label"].tolist(), np.argmax(logits, axis=-1).tolist(), name,
             extra={"train_time_s": round(train_time, 1), "epochs": args.epochs,
                    "lr": args.lr, "batch": args.batch, "max_len": MAX_LEN,
                    "base_model": args.model})

    # Save final (best) model + tokenizer for reuse
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"Saved model + tokenizer -> models/{name}/final/")

    # Same two sample sentences as the Stacked ML script
    samples = ["Oh great, another Monday morning!",
               "The government passed a new law today."]
    model.eval()
    with torch.no_grad():
        enc = tokenizer(samples, return_tensors="pt", padding=True).to(model.device)
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist()
    for s, p in zip(samples, probs):
        print(f"  [{'sarcastic' if p > 0.5 else 'not sarcastic'} p={p:.2f}] {s}")


if __name__ == "__main__":
    main()
