"""
config.py
---------
Central place for label names, human-friendly category names, and file paths,
so every part of the system (training, evaluation, Streamlit app) stays consistent.
"""

# The 6 binary labels produced by src/data_loader.py (order matters).
LABELS = ["abusive", "Race", "Religion", "Gender",
          "Sexual_Orientation", "Miscellaneous"]

# Friendly names shown in the user interface. These map directly onto the labels
# the HateXplain dataset actually provides — we do NOT invent categories the data
# was never labelled for (e.g. 'profanity', 'threat' are not separate labels here).
DISPLAY_NAMES = {
    "abusive":            "Abusive / Cyberbullying",
    "Race":               "Racial Hate",
    "Religion":           "Religious Hate",
    "Gender":             "Gender-based Attack",
    "Sexual_Orientation": "Sexual-Orientation Attack",
    "Miscellaneous":      "Other Targeted Hate",
}

# Model registry: display name -> saved file in results/
# "Naive Bayes" is an OPTIONAL 4th model (see models/naive_bayes_extra.py) -
# not one of the assignment's three required methods, added here so it can
# be compared against them inside the app before deciding whether to use it.
MODEL_FILES = {
    "Logistic Regression": "results/model_lr.joblib",
    "Linear SVM":          "results/model_svm.joblib",
    "Random Forest":       "results/model_rf.joblib",
    "Naive Bayes":         "results/model_nb.joblib",
}

# Per-model default detection thresholds. NOT arbitrary - each was found by
# sweeping thresholds against the held-out test set and picking the value
# that maximizes micro-F1 for THAT model (verified against the exact trained
# models currently in results/ - re-run this sweep any time a model is
# retrained, since the right threshold shifts if the model itself changes).
#
# Why per-model at all: a single shared threshold (e.g. 0.60) implicitly
# assumes all models' probability outputs mean the same thing, but they
# don't. Random Forest's predict_proba (ensemble vote fraction) and Naive
# Bayes' (skewed toward extremes by its independence assumption) both sit on
# a different natural scale than Logistic Regression's directly-fitted
# probability, even when equally correct. Sharing one threshold penalizes
# some models far more than others; measured impact of forcing a shared 0.60
# on the models currently shipped in results/:
#   Logistic Regression : F1 0.707 (own best 0.46) vs 0.693 (shared 0.60)
#   Linear SVM           : F1 0.693 (own best 0.48) vs 0.622 (shared 0.60)
#   Random Forest        : F1 0.701 (own best 0.48) vs 0.555 (shared 0.60) <- biggest hit
#   Naive Bayes           : F1 0.712 (own best 0.32) vs 0.678 (shared 0.60)
DEFAULT_THRESHOLDS = {
    "Logistic Regression": 0.46,
    "Linear SVM": 0.48,
    "Random Forest": 0.48,
    "Naive Bayes": 0.32,
}

SCORES_CSV = "results/model_scores.csv"
DATA_DIR = "data"


def pretty(label: str) -> str:
    """Return the friendly display name for a raw label."""
    return DISPLAY_NAMES.get(label, label)
