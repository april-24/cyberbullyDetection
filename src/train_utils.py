"""
train_utils.py
--------------
Shared routine used by all three member scripts: load data -> fit the pipeline ->
evaluate -> save the model. Keeps each member's file focused on just their model.
"""

import os
import joblib
from .common import prepare_data
from .evaluate import evaluate_model, save_result


def train_and_save(model_name, pipeline, model_path, sample=None):
    """Fit `pipeline`, print/save metrics, and save the model bundle to disk."""
    X_train, X_test, y_train, y_test, labels = prepare_data(sample=sample)

    print(f"\nTraining {model_name} ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    result = evaluate_model(model_name, y_test.values, y_pred, labels)
    save_result(result)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # compress=3 keeps Random Forest files small (~6 MB instead of ~170 MB)
    joblib.dump({"pipeline": pipeline, "labels": labels}, model_path, compress=3)
    print(f"Saved trained model -> {model_path}")
    return pipeline
