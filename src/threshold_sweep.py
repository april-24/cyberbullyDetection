"""
threshold_sweep.py
-------------------
Finds each trained model's own optimal detection threshold (the one that
maximizes micro-F1 on the held-out test set), rather than assuming one
shared threshold works equally well for every model.

Why this matters: different algorithms produce probability-like scores on
different natural scales, even when equally correct. Forcing every model to
share one threshold (e.g. 0.60) can badly under-serve some of them - on this
project's models, Random Forest's F1 dropped from 0.70 (at its own best
threshold) to 0.56 when forced to share a 0.60 cutoff with the others.

Run this any time you retrain a model - the right threshold is specific to
that exact trained model, not to the algorithm in general, so it can shift
if you retrain (e.g. after merging in more data).

Run from the project root:
    python -m src.threshold_sweep
"""

import numpy as np
from sklearn.metrics import f1_score
from .common import prepare_data
from .predictor import available_models, load_model, _label_probs


def sweep(step=0.02, lo=0.20, hi=0.86):
    Xtr, Xte, ytr, yte, labels = prepare_data(verbose=False)

    print(f"{'Model':22s} {'Best threshold':15s} {'F1 @ best':10s} "
          f"{'F1 @ 0.50':10s} {'F1 @ 0.60':10s}")

    results = {}
    for name, path in available_models().items():
        bundle = load_model(path)
        P = _label_probs(bundle["pipeline"], list(Xte))

        best_th, best_f1 = 0.5, 0.0
        for th in np.arange(lo, hi, step):
            pred = (P >= th).astype(int)
            f1 = f1_score(yte.values, pred, average="micro", zero_division=0)
            if f1 > best_f1:
                best_th, best_f1 = round(float(th), 2), f1

        f1_50 = f1_score(yte.values, (P >= 0.5).astype(int), average="micro", zero_division=0)
        f1_60 = f1_score(yte.values, (P >= 0.6).astype(int), average="micro", zero_division=0)
        results[name] = best_th
        print(f"{name:22s} {best_th:<15.2f} {best_f1:<10.4f} {f1_50:<10.4f} {f1_60:<10.4f}")

    print("\nCopy this into DEFAULT_THRESHOLDS in src/config.py:")
    print("DEFAULT_THRESHOLDS = {")
    for name, th in results.items():
        print(f'    "{name}": {th},')
    print("}")
    return results


if __name__ == "__main__":
    sweep()
