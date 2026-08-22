"""
naive_bayes_extra.py   (OPTIONAL 4th model - for testing/comparison)
======================================================================
Method : Multinomial Naive Bayes + TF-IDF features (word bigrams + character n-grams)
Wrapper: OneVsRestClassifier (one NB per label -> multi-label)

This is NOT one of the assignment's three required methods (those are
Logistic Regression, Linear SVM, and Random Forest - see member1/2/3). It's
added here so you can compare a fourth approach and decide for yourselves
whether to swap it in for one of the three, based on real results rather
than guessing.

Why try it: Naive Bayes is specifically well-suited to word/n-gram COUNT data
like TF-IDF (it's a direct probabilistic model over feature frequencies, with
no ensemble-voting dilution the way Random Forest has on sparse text). In
testing, it caught cases Random Forest missed (slurs, profanity, obvious
insults) with much better-separated confidence scores, and trains in ~2
seconds instead of ~90.

BE HONEST ABOUT THE TRADE-OFF IN YOUR REPORT: Naive Bayes had HIGHER recall
on the core "abusive" flag than Random Forest, but LOWER recall specifically
on the minority target categories (Gender, Sexual Orientation, Miscellaneous)
- it's better at catching cyberbullying in general, worse at correctly
identifying which group is targeted in these harder categories. Also, Naive
Bayes' confidence scores tend to run to extremes (very close to 0% or 100%)
due to its feature-independence assumption - a well-known property, not a
sign of unusually high certainty. Don't quote its confidence percentages as
precisely calibrated probabilities.

Run from the project root:
    python -m models.naive_bayes_extra
    python -m models.naive_bayes_extra --sample 20000
"""

import os
import sys
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_utils import train_and_save, build_word_char_features

MODEL_NAME = "Naive Bayes"
MODEL_PATH = "results/model_nb.joblib"


def build_pipeline():
    return Pipeline([
        ("features", build_word_char_features(word_max_features=15000,
                                              char_max_features=8000)),
        ("clf", OneVsRestClassifier(MultinomialNB(alpha=0.3))),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    train_and_save(MODEL_NAME, build_pipeline(), MODEL_PATH, sample=args.sample)
