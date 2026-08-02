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

Navigate using the **top navigation bar** (minimal text links, not a sidebar) —
click any page name to switch. Inline **help tooltips** (the small (?) icon)
sit right next to each input, so guidance is where you need it instead of in
a separate box.

| Page | What it does |
|------|--------------|
| **Home** | Project intro, objectives, NLP task, implemented models, dataset summary |
| **Dataset Statistics** | Dataset overview/info, class & category distribution, word cloud, NLP workflow diagram |
| **Data Preprocessing** | Live before/after demo of every cleaning step, quality checks, TF-IDF preview |
| **Cyberbully Detection** | Three tabs: Enter Comment, Import CSV, Social Media URL |
| **Model Evaluation** | Accuracy (per-label, paper-comparable), Precision/Recall/F1, confusion matrices, classification report, training/prediction time |

Every result includes: prediction, confidence score, model used, processing time,
category breakdown with progress bars, highlighted offensive words, a
plain-English explanation, and a **suggested next step** (e.g. save evidence,
report to the platform, escalate if it's a repeated pattern).

**A note on "Accuracy":** the headline **Accuracy** metric is the *per-label*
average (each of the 6 categories scored as its own yes/no question, then
averaged) — this is the number comparable to what most classification papers
report, typically 82–84% here. **Subset Accuracy** is also shown, but it's a
much stricter "all 6 labels correct at once" measure and will always look
lower for any multi-label system — that's expected, not a sign of a weak model.

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
| **X / Twitter** | ❌ No | As of Feb 2026, X removed its free API tier entirely — it's now pay-per-use with no free allowance (~$0.005/read, ~2M read/month cap). Some published research papers used X data collected before this change, or under special academic agreements no longer available to new developers. |
| Facebook / Instagram / TikTok | ❌ No | Their Terms of Service **prohibit** automated comment collection from arbitrary posts, and they block it technically. Meta's official Graph API is designed for managing pages/ads you own, not scraping public content. |

The app states this clearly rather than pretending otherwise. For unsupported
platforms, copy comments manually into the **Cyberbully Detection → Enter Comment** tab. A **demo
mode** with built-in sample comments is provided so the feature can be shown
without an API key — the app always labels demo data as such.

**To enable YouTube:** get a free key from Google Cloud Console (enable
"YouTube Data API v3"), then paste it into the box on the Cyberbully Detection page's Social Media URL tab — it's completely free, no credit card required.

---

## Results (held-out 20% test set, threshold 0.50)

| Model | Accuracy | Micro-F1 | Macro-F1 | Subset Acc. | Train Time | Predict Time |
|-------|:--------:|:--------:|:--------:|:-----------:|:----------:|:------------:|
| **Random Forest** | **0.849** | **0.709** | 0.689 | 0.392 | 87.0s | 1.80s |
| Logistic Regression | 0.836 | 0.706 | **0.679**† | 0.369 | 5.8s | 0.48s |
| Linear SVM | 0.829 | 0.688 | 0.652 | 0.352 | 8.5s | 0.48s |

†Weighted-F1 tells a slightly different story per model — see `results/model_scores.csv` for the full metric set (precision/recall/F1 in micro, macro, *and* weighted averaging).

**Discussion points for your report:**
- All three models use the **same feature set**: word-level TF-IDF (unigrams +
  bigrams) *plus* character-level TF-IDF (3-5 letter n-grams) — see "Evasion
  & obfuscation detection" below for why.
- Random Forest edges out the linear models on raw accuracy and F1 here, but
  takes ~10-15x longer to train and predict, and produces a much larger model
  file (~33 MB vs ~2 MB) — a real accuracy/cost tradeoff worth discussing.
- **Random Forest's confidence scores run more conservative** than the linear
  models', even on comments it classifies correctly — a well-documented
  structural property of ensemble voting over sparse, high-dimensional text
  (each split only sees a random subset of features, so short comments often
  don't reach the words that matter in many trees). This is *not* a bug; it's
  a legitimate, citable limitation for your report's model comparison.
- **Subset accuracy looks low (~0.35-0.39)** because it demands *all six*
  labels be correct simultaneously. That is normal and expected for
  multi-label tasks — **Accuracy** (per-label average, ~83-85%) is the fairer,
  paper-comparable read.

---

## Evasion & obfuscation detection

People trying to slip an offensive word past a filter typically do one of
three things — all handled by `src/preprocessing.py` before the "strip
everything but letters" step, so the underlying word survives instead of
being destroyed:

| Trick | Example | Normalized to |
|-------|---------|---------------|
| Leetspeak substitution | `n1gg4`, `sh1t`, `@sshole` | `nigga`, `shit`, `asshole` |
| Spaced-out letters | `n.i.g.g.a`, `s h i t` | `nigga`, `shit` |
| Repeated-character spam | `stuuuupid` | `stupid` |

This is **not a complete solution** — evasion detection is fundamentally an
arms race, and new obfuscation tricks will always exist. As a second,
complementary layer, every model also uses **character n-gram features**
(3-5 letter chunks), which can catch obfuscation patterns the explicit rules
above don't know about, because a lightly disguised word still shares most of
its character substrings with the original. Both points are worth stating
plainly in your report's limitations section — being upfront about what a
detection system *can't* fully solve is a stronger academic position than
implying it's solved.

---

## Known limitations (be honest about these in your report — it earns marks)

1. **Dataset domain bias.** HateXplain was collected from Twitter and Gab, which
   are hate-speech-heavy sources. Its "normal" class is therefore not everyday
   polite conversation, so ordinary benign sentences can score near the decision
   boundary. This is why the app defaults to a **0.60 threshold** instead of 0.50
   — it noticeably reduces false positives on benign text at a cost of only
   ~0.02 micro-F1. The sensitivity slider on each detection page lets you explore this tradeoff live, and
   reported metrics use the standard 0.50 for comparability.
2. **Minority categories are harder.** "Gender" and "Miscellaneous" have the
   lowest F1 — fewer training examples means weaker performance.
3. **No sarcasm or context understanding.** TF-IDF is bag-of-words; it cannot
   read intent, irony, or conversation history.
4. **English only.**
5. **Not a moderation authority.** Predictions are statistical and can be wrong.
   Human review should always precede any consequential action.

---

## Want more data? Collect Malaysian comments (optional)

The `crawler/` folder has tools to legally collect and label more comment
data, with a focus on Malaysia — useful if you want the models to reflect
local slang/context, or as an "extra effort" feature for a higher grade.

```bash
python crawler/youtube_batch_crawler.py     # collect (official YouTube API)
streamlit run crawler/annotate_data.py      # label what you collected
python crawler/merge_datasets.py            # merge into training data
python -m models.member1_logistic_regression  # retrain
```

**Read `crawler/CRAWLING_GUIDE.md` first** — it covers what's legal to crawl
(and what isn't), how to get free API access, and the full workflow.

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
├── crawler/
│   ├── CRAWLING_GUIDE.md          full guide: legal notes, setup, workflow
│   ├── youtube_batch_crawler.py   collect comments (official YouTube API)
│   ├── reddit_crawler.py          collect comments (official Reddit API)
│   ├── generic_crawler.py         robots.txt-respecting template for other sites
│   ├── annotate_data.py           Streamlit tool to label crawled comments
│   └── merge_datasets.py          combine labeled data into training set
├── run_eda.py                  EDA charts
├── compare_models.py           comparison table + chart
├── requirements.txt
└── results/                    trained models, scores, charts
```

---

## How it meets the NLP assignment requirements (a–g)

- **(a) NLP task identified** — text classification for cyberbullying detection.
- **(b) Background study** — for the Part 1 documentation.
- **(c) Crawler *or* reliable dataset** — the assignment allows either. We use
  both: the public **HateXplain** dataset (a reliable, peer-reviewed source)
  as the base, *plus* actual crawler tools (`crawler/`) to collect additional
  real-world Malaysian comments via official APIs (YouTube, Reddit) which you
  can label and merge in — see `crawler/CRAWLING_GUIDE.md`.
- **(d) Preprocessing** — cleaning (URLs, mentions, punctuation, special
  characters), **tokenisation**, stop-word removal, **lemmatisation** (used in
  place of stemming, since it produces real dictionary words), and
  **feature extraction** via TF-IDF (unigrams + bigrams) — one of the
  assignment's named example methods.
- **(e) Each member a different method** — mapped directly onto the
  assignment's suggested method list:
  - Member 1: **Logistic Regression** (explicitly named in the spec)
  - Member 2: **Linear SVM** (explicitly named in the spec)
  - Member 3: **Random Forest** — an ensemble of **Decision Trees**, the third
    method family the spec names
- **(f) Compare & evaluate** — Accuracy, Precision, Recall, F1 (micro + macro),
  Hamming loss, and confusion matrices — all the metrics the spec requires,
  displayed live in the **Model Evaluation** and **Model Comparison** pages.
- **(g) Reliable dataset source** — HateXplain, a peer-reviewed academic dataset.

---

## Ideas for pushing toward an Excellent grade

Beyond what's already built, these are realistic, scoped additions if you
have time left before the deadline — roughly ordered by effort:

**Low effort, real payoff:**
- **Manual data augmentation for minority categories.** Gender and
  Miscellaneous have the weakest F1 scores because they have the fewest
  training examples — labeling even 100-200 more examples specifically for
  these categories (via `crawler/annotate_data.py`) would likely move these
  numbers more than any hyperparameter tweak.
- **Inter-annotator agreement.** If two group members independently label the
  same 50-100 crawled comments, compute agreement (e.g. Cohen's Kappa) between
  them. This is a genuine, citable data-quality metric that most student
  projects skip.
- **Error analysis section.** Pull 10-15 misclassified test-set comments
  (false positives and false negatives) and discuss *why* the model got them
  wrong — sarcasm, coded language, short context, etc. This is exactly the
  kind of critical analysis the rubric's "Results & Discussion" section rewards.

**Medium effort:**
- **A fourth model for contrast.** Everything here is TF-IDF-based; adding
  one model that uses a fundamentally different representation (e.g. word
  embeddings via Word2Vec/GloVe, or a small pretrained transformer like
  DistilBERT) gives a much stronger "compare different feature extraction
  approaches" narrative than three TF-IDF variants. Fair warning: a
  transformer model is a real jump in setup complexity and training time —
  scope it carefully against your remaining time.
- **Threshold tuning per label, not just globally.** Right now one threshold
  applies to all 6 categories. Tuning a separate optimal threshold per
  category (maximizing F1 per label on a validation split) is a legitimate
  technique and would likely raise macro-F1 measurably.
- **Malay-language preprocessing.** The current pipeline is English-tuned
  (English stopwords/lemmatizer). Adding a Malay stopword list and even basic
  Malay-aware tokenization would directly strengthen the Malaysia angle of
  this project, and ties directly into the crawler/labeling work already built.

**Higher effort, strong differentiator:**
- **User study / usability evaluation.** Have a few classmates or friends use
  the app and rate the explanations/suggested actions for clarity and
  trustworthiness. Real user feedback data is uncommon in student NLP
  projects and stands out.
- **Deployment monitoring angle.** Discuss (even without fully building it)
  how the system would need to be retrained/monitored over time as language
  and evasion tactics evolve — ties directly into the evasion-detection
  limitations already documented above, and shows systems-level thinking
  beyond just model accuracy.

---

## Troubleshooting

- **"No trained models found"** — run the three training commands above, and make
  sure you launched Streamlit from the project root (the folder containing `app.py`).
- **Models don't appear after retraining** — stop Streamlit (`Ctrl+C`) and start it
  again; it caches loaded models per process.
- **NLTK download errors** — the code falls back to a built-in stop-word list and
  still runs.
- **Reddit fetch fails** — Reddit rate-limits automated requests; wait and retry.
