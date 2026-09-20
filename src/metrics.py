"""Metrics shared by both pipelines, saved in the same JSON format."""
import json

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from src.config import RESULTS_DIR


def evaluate(y_true, y_pred, name: str, extra: dict | None = None) -> dict:
    """Print a report and save metrics to results/<name>.json."""
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    # Sarcastic-class (label 1) scores: these are what the slides report
    p_s, r_s, f_s, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    res = {
        "model": name,
        "n_test": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_sarcastic": p_s, "recall_sarcastic": r_s, "f1_sarcastic": f_s,
        "precision_macro": p_mac, "recall_macro": r_mac, "f1_macro": f_mac,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # rows=true, cols=pred
        **(extra or {}),
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(classification_report(y_true, y_pred, digits=4,
                                target_names=["not sarcastic", "sarcastic"]))
    print("Confusion matrix [rows=true, cols=pred]:", res["confusion_matrix"])

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(f"Saved -> results/{name}.json")
    return res
