"""
evaluate.py
-----------
Shared evaluation helpers so every member reports metrics the SAME way.

For multi-label classification the key metrics are:
    - Subset accuracy : fraction of rows where ALL labels are predicted correctly (strict)
    - Hamming loss    : fraction of individual label predictions that are wrong (lower = better)
    - Precision / Recall / F1 with 'micro' and 'macro' averaging
        micro = aggregate over all label decisions (favours frequent labels)
        macro = average of per-label scores (treats every label equally)

`evaluate_model` returns a dict of the headline numbers and prints a full report.
`save_result` appends a model's headline numbers to a shared CSV so
compare_models.py can build the final comparison table.
"""

import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
    classification_report,
)

RESULTS_CSV = os.path.join("results", "model_scores.csv")


def evaluate_model(name, y_true, y_pred, label_names):
    """Print a full multi-label report and return headline metrics as a dict."""
    subset_acc = accuracy_score(y_true, y_pred)
    hloss = hamming_loss(y_true, y_pred)

    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*60}\n  RESULTS: {name}\n{'='*60}")
    print(f"Subset accuracy (all labels correct): {subset_acc:.4f}")
    print(f"Hamming loss (lower is better)       : {hloss:.4f}")
    print(f"Micro  -> P: {p_mi:.4f}  R: {r_mi:.4f}  F1: {f_mi:.4f}")
    print(f"Macro  -> P: {p_ma:.4f}  R: {r_ma:.4f}  F1: {f_ma:.4f}")
    print("\nPer-label report:")
    print(classification_report(y_true, y_pred, target_names=label_names,
                                zero_division=0))

    return {
        "model": name,
        "subset_accuracy": round(subset_acc, 4),
        "hamming_loss": round(hloss, 4),
        "precision_micro": round(p_mi, 4),
        "recall_micro": round(r_mi, 4),
        "f1_micro": round(f_mi, 4),
        "precision_macro": round(p_ma, 4),
        "recall_macro": round(r_ma, 4),
        "f1_macro": round(f_ma, 4),
    }


def save_result(result: dict, path: str = RESULTS_CSV):
    """Append (or replace) one model's scores in the shared results CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["model"] != result["model"]]   # replace old entry if re-run
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = pd.DataFrame([result])
    df.to_csv(path, index=False)
    print(f"\nSaved scores for '{result['model']}' -> {path}")
