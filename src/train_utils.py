"""
train_utils.py
--------------
Shared routine used by all three member scripts: load data -> fit the pipeline ->
evaluate -> save the model. Keeps each member's file focused on just their model.
"""

import os
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from .common import prepare_data
from .evaluate import evaluate_model, save_result


def build_word_char_features(word_max_features=30000, char_max_features=10000,
                             word_ngram_range=(1, 2)):
    """
    A FeatureUnion of WORD-level and CHARACTER-level TF-IDF, used by all three
    models so the comparison stays fair.

    Why character n-grams too, not just words: word-level TF-IDF only
    recognizes a word if it appears EXACTLY as seen during training. That
    makes it easy to dodge with obfuscation the text-cleaning step doesn't
    already catch (unusual substitutions, deliberate misspellings, etc.).
    Character n-grams (3-5 letter chunks, e.g. "idi", "dio", "iot" from
    "idiot") are much harder to evade, because a lightly disguised word still
    shares most of its character substrings with the original - this is a
    standard technique in hate-speech/toxic-comment classification literature
    for exactly this robustness reason. It also gives models with a smaller
    word vocabulary (like the Random Forest here) meaningfully more signal to
    work with, since sub-word patterns repeat across many different words.

    Word-level features remain primary (larger max_features, unigrams +
    bigrams) since they carry most of the interpretable signal used for
    highlighting influential words in the app.
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(max_features=word_max_features,
                                 ngram_range=word_ngram_range,
                                 min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(max_features=char_max_features,
                                 analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=2, sublinear_tf=True)),
    ])


def train_and_save(model_name, pipeline, model_path, sample=None):
    """Fit `pipeline`, print/save metrics, and save the model bundle to disk."""
    X_train, X_test, y_train, y_test, labels = prepare_data(sample=sample)

    print(f"\nTraining {model_name} ...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = pipeline.predict(X_test)
    predict_time = time.time() - t0

    result = evaluate_model(model_name, y_test.values, y_pred, labels,
                            train_time=train_time, predict_time=predict_time)
    save_result(result)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # compress=3 keeps Random Forest files small (~6 MB instead of ~170 MB)
    joblib.dump({"pipeline": pipeline, "labels": labels}, model_path, compress=3)
    print(f"Saved trained model -> {model_path}")
    return pipeline
