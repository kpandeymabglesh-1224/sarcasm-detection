"""Stage 5: side-by-side comparison of the two pipelines.

    python -m src.compare

Reads results/*.json written by stacked_ml.py and distilbert_model.py and
prints / saves (results/comparison.md) a table next to the slide-5 numbers.
"""
import json

from src.config import RESULTS_DIR

# Reported on slide 5 (sarcastic-class precision/recall/F1)
SLIDE = {
    "Stacked ML": {"accuracy": 0.803, "precision": 0.80, "recall": 0.79, "f1": 0.79},
    "DistilBERT": {"accuracy": 0.927, "precision": 0.93, "recall": 0.92, "f1": 0.92},
}

RUNS = [  # (results file, label, slide row it should be compared with)
    ("stacked_ml_original", "Stacked ML - exact Colab replica*", "Stacked ML"),
    ("stacked_ml", "Stacked ML - corrected", "Stacked ML"),
    ("distilbert", "DistilBERT - fine-tuned", "DistilBERT"),
]


def load(name):
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    # Sarcastic-class scores from the confusion matrix (works for older files too)
    (tn, fp), (fn, tp) = r["confusion_matrix"]
    p, rec = tp / (tp + fp), tp / (tp + fn)
    r.update(precision_sarcastic=p, recall_sarcastic=rec, f1_sarcastic=2 * p * rec / (p + rec))
    return r


def main():
    rows = []
    header = "| Model | Accuracy | Precision | Recall | F1 | Macro F1 | Test size | Train time |"
    rows += [header, "|" + "---|" * 8]
    for slide_name, s in SLIDE.items():
        rows.append(f"| {slide_name} - slide 5 (reported) | {s['accuracy']:.1%} | {s['precision']:.2f} "
                    f"| {s['recall']:.2f} | {s['f1']:.2f} | - | - | - |")
    results = {}
    for name, label, _ in RUNS:
        r = load(name)
        if r is None:
            print(f"(skipping {name}: results/{name}.json not found)")
            continue
        results[name] = r
        rows.append(f"| {label} | {r['accuracy']:.2%} | {r['precision_sarcastic']:.3f} "
                    f"| {r['recall_sarcastic']:.3f} | {r['f1_sarcastic']:.3f} | {r['f1_macro']:.3f} "
                    f"| {r['n_test']} | {r.get('train_time_s', 0):.0f} s |")

    notes = ["",
             "Precision / recall / F1 are for the sarcastic class (as on slide 5).",
             "*The Colab replica fits TF-IDF on all data before splitting and keeps duplicates, "
             "and uses a different test split; it is only there to check the reported 80.3%."]
    if "stacked_ml" in results and "distilbert" in results:
        a, b = results["stacked_ml"], results["distilbert"]
        gap = (b["accuracy"] - a["accuracy"]) * 100
        err_a, err_b = 1 - a["accuracy"], 1 - b["accuracy"]
        notes += ["",
                  f"Same {a['n_test']} test headlines for both corrected runs.",
                  f"Accuracy gap (DistilBERT - Stacked ML): {gap:.1f} percentage points "
                  f"(slide 5 reports {(0.927 - 0.803) * 100:.1f}).",
                  f"Errors: Stacked ML {err_a * a['n_test']:.0f}, DistilBERT {err_b * b['n_test']:.0f} "
                  f"-> DistilBERT makes {1 - err_b / err_a:.0%} fewer mistakes."]

    text = "\n".join(rows + notes)
    print(text)
    (RESULTS_DIR / "comparison.md").write_text("# Stacked ML vs DistilBERT\n\n" + text + "\n")
    print("\nSaved -> results/comparison.md")


if __name__ == "__main__":
    main()
