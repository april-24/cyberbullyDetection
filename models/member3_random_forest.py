"""
member3_random_forest.py   (MEMBER 3's solution)
================================================
Method : Random Forest + TF-IDF features
Wrapper: OneVsRestClassifier (one forest per label -> multi-label)

A Random Forest is an ensemble of decision trees; each tree votes and the forest
averages them, which reduces overfitting. Unlike the linear models it can capture
non-linear combinations of words. It provides predict_proba (used for confidence)
and per-feature importances (used to highlight influential words).

Parameters are tuned to keep the saved model small (~7 MB) while staying accurate.

Run from the project root:
    python -m models.member3_random_forest
    python -m models.member3_random_forest --sample 20000
"""

import os
import sys
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_utils import train_and_save

MODEL_NAME = "Random Forest"
MODEL_PATH = "results/model_rf.joblib"


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 1),
                                  min_df=3, sublinear_tf=True)),
        ("clf", OneVsRestClassifier(RandomForestClassifier(
            n_estimators=120, max_depth=45, min_samples_leaf=3,
            n_jobs=-1, class_weight="balanced", random_state=42))),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    train_and_save(MODEL_NAME, build_pipeline(), MODEL_PATH, sample=args.sample)
