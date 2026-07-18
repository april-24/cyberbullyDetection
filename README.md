# 🛡️ CyberShield — Multi-Model Cyberbullying Detection System

AI Assignment — **Title 4: Natural Language Processing** — TARUMT Session 202605.

A Streamlit web application that detects cyberbullying in online comments,
identifies **which group is targeted**, reports a **confidence score**, highlights
the **influential words**, and **compares three machine-learning models**.

| Member | Method | Script |
|--------|--------|--------|
| Member 1 | **Logistic Regression** + TF-IDF | `models/member1_logistic_regression.py` |
| Member 2 | **Linear SVM** + TF-IDF | `models/member2_svm.py` |
| Member 3 | **Random Forest** + TF-IDF | `models/member3_random_forest.py` |

---

## Quick start

```bash
pip install -r requirements.txt      # 1. install
streamlit run app.py                 # 2. launch the web app
```

The three models are **already trained and included**, so the app works
immediately. To retrain them yourself:

```bash
python -m models.member1_logistic_regression
python -m models.member2_svm
python -m models.member3_random_forest
python compare_models.py
```

Or open **`Cyberbully_Detection.ipynb`** in Jupyter/Colab and Run All.

---

## Application pages

| Page | What it does |
|------|--------------|
| **Home** | Project intro, objectives, supported categories |
| **Text Detection** | Analyse one comment or many (one per line) |
| **Social Media Detection** | Fetch + analyse comments from a YouTube or Reddit URL |
| **Batch File Detection** | Upload a CSV/TXT of comments and analyse in bulk |
| **Model Comparison** | Run all three models on the same input side by side |
| **Dataset Statistics** | Dataset size, class balance, lengths, word frequencies |
| **Model Evaluation** | Accuracy, Precision, Recall, F1, confusion matrices |

Every result includes: prediction, confidence score, model used, processing time,
category breakdown with progress bars, highlighted offensive words, and a
plain-English explanation.

---

## What it detects

The model outputs **six labels at once** (a comment can carry several):

| Label | Displayed as |
|-------|--------------|
| `abusive` | Abusive / Cyberbullying |
| `Race` | Racial Hate |
| `Religion` | Religious Hate |
| `Gender` | Gender-based Attack |
| `Sexual_Orientation` | Sexual-Orientation Attack |
| `Miscellaneous` | Other Targeted Hate |

> **Note on categories:** these are the categories the **HateXplain dataset is
> actually labelled for**. Categories such as "threat" or "profanity" are *not*
> separate labels in this dataset, so the system does not claim to predict them —
> a model cannot reliably predict a class it never saw during training.

---

## Social media support — important

| Platform | Supported? | Method |
|----------|-----------|--------|
| **YouTube** | ✅ Yes | Official **YouTube Data API v3** (needs your own free key) |
| **Reddit** | ✅ Yes | Public `.json` endpoints |
| Facebook / Instagram / X / TikTok | ❌ No | Their Terms of Service **prohibit** automated comment collection, and they block it technically |

The app states this clearly rather than pretending otherwise. For unsupported
platforms, copy comments manually into the **Text Detection** page. A **demo
mode** with built-in sample comments is provided so the feature can be shown
without an API key — the app always labels demo data as such.

**To enable YouTube:** get a free key from Google Cloud Console (enable
"YouTube Data API v3"), then paste it into the sidebar box on the Social Media page.

---

## Results (held-out 20% test set, threshold 0.50)

| Model | Micro-F1 | Macro-F1 | Subset Acc. | Hamming Loss |
|-------|:--------:|:--------:|:-----------:|:------------:|
| **Logistic Regression** | **0.704** | **0.679** | 0.363 | 0.165 |
| Random Forest | 0.692 | 0.678 | 0.351 | 0.165 |
| Linear SVM | 0.688 | 0.654 | 0.348 | 0.172 |

**Discussion points for your report:**
- Logistic Regression wins on **recall** (0.72) — it catches the most abuse.
- Random Forest wins on **precision** (0.70) — fewer false alarms.
- All three are close, which is itself a finding: on TF-IDF features the choice
  of classifier matters less than the quality of the features.
- **Subset accuracy looks low (~0.35)** because it demands *all six* labels be
  correct simultaneously. That is normal and expected for multi-label tasks —
  Hamming loss (~0.17, i.e. ~83% of individual label decisions correct) is the
  fairer read.

---

## Known limitations (be honest about these in your report — it earns marks)

1. **Dataset domain bias.** HateXplain was collected from Twitter and Gab, which
   are hate-speech-heavy sources. Its "normal" class is therefore not everyday
   polite conversation, so ordinary benign sentences can score near the decision
   boundary. This is why the app defaults to a **0.60 threshold** instead of 0.50
   — it noticeably reduces false positives on benign text at a cost of only
   ~0.02 micro-F1. The sidebar slider lets you explore this tradeoff live, and
   reported metrics use the standard 0.50 for comparability.
2. **Minority categories are harder.** "Gender" and "Miscellaneous" have the
   lowest F1 — fewer training examples means weaker performance.
3. **No sarcasm or context understanding.** TF-IDF is bag-of-words; it cannot
   read intent, irony, or conversation history.
4. **English only.**
5. **Not a moderation authority.** Predictions are statistical and can be wrong.
   Human review should always precede any consequential action.

---

## Project structure

```
cyberbully_detection/
├── app.py                      Streamlit web application (7 pages)
├── Cyberbully_Detection.ipynb  notebook version (train + evaluate)
├── data/
│   └── final_hateXplain.csv    dataset (20,109 comments)
├── src/
│   ├── config.py               labels + friendly display names
│   ├── data_loader.py          builds the 6 binary labels
│   ├── preprocessing.py        cleaning, tokenisation, lemmatisation
│   ├── common.py               load + clean + train/test split
│   ├── train_utils.py          shared train/evaluate/save routine
│   ├── evaluate.py             multi-label metrics
│   ├── predictor.py            confidence, word highlighting, explanations
│   └── social.py               YouTube / Reddit fetching
├── models/
│   ├── member1_logistic_regression.py
│   ├── member2_svm.py
│   └── member3_random_forest.py
├── run_eda.py                  EDA charts
├── compare_models.py           comparison table + chart
├── requirements.txt
└── results/                    trained models, scores, charts
```

---

## How it meets the NLP assignment requirements (a–g)

- **(a) NLP task identified** — text classification for cyberbullying detection.
- **(b) Background study** — for the Part 1 documentation.
- **(c) Crawler *or* reliable dataset** — we use the public HateXplain dataset
  (the specification allows either), *plus* official-API comment retrieval for
  YouTube and Reddit in the app.
- **(d) Preprocessing** — cleaning (URLs, mentions, punctuation, special
  characters), **tokenisation**, stop-word removal, **lemmatisation**, and
  **feature extraction** via TF-IDF (unigrams + bigrams).
- **(e) Each member a different method** — Logistic Regression / Linear SVM /
  Random Forest.
- **(f) Compare & evaluate** — Accuracy, Precision, Recall, F1 (micro + macro),
  Hamming loss, and confusion matrices.
- **(g) Reliable dataset source** — HateXplain, a peer-reviewed academic dataset.

---

## Troubleshooting

- **"No trained models found"** — run the three training commands above, and make
  sure you launched Streamlit from the project root (the folder containing `app.py`).
- **Models don't appear after retraining** — stop Streamlit (`Ctrl+C`) and start it
  again; it caches loaded models per process.
- **NLTK download errors** — the code falls back to a built-in stop-word list and
  still runs.
- **Reddit fetch fails** — Reddit rate-limits automated requests; wait and retry.
