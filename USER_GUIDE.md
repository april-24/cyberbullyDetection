# 🛡️ CyberShield — User Guide

This guide walks through every page of the app: what it's for, what to type in,
and how to read the results. For installation and setup, see `README.md`
instead — this document assumes the app is already running
(`streamlit run app.py`).

---

## Before you start: the sidebar controls (apply to every page)

On the left side of the app, two settings affect **every page** that makes a prediction:

| Control | What it does |
|---------|---------------|
| **Model** | Choose which of the three trained models (Logistic Regression, Linear SVM, Random Forest) is used for detection. |
| **Detection sensitivity (threshold)** | How confident the model must be before it flags something as cyberbullying. **Lower** = flags more comments (catches more, but more false alarms). **Higher** = stricter (fewer false alarms, may miss some). Default is 0.60. |

You don't need to touch these to use the app — the defaults work well — but
they're there if you want to explore how model choice or strictness changes
the outcome.

---

## 1. Home Page

**What it's for:**
The landing page. It introduces the project, explains what the system detects,
and gives a quick orientation before you start using it.

**What's on it:**
- Project title and short description
- Quick stats: how many models, how many categories, how many training comments
- The list of categories the system can detect (e.g. Racial Hate, Religious Hate)
- The project's objectives
- A short "how to use" summary of the other pages

**How to use it:**
There's nothing to click here — just read it, then use the **sidebar** on the
left to jump to whichever page you need (usually **Text Detection** first).

---

## 2. Text Detection

**What it's for:**
The core feature. Type or paste any comment(s) and find out instantly whether
they contain cyberbullying, which category they target, and why.

**Who it's for:** anyone testing a single sentence quickly — parents checking a
message, moderators screening a comment, or you demonstrating the system.

**How to use it:**
1. Type or paste your comment into the text box.
   - **One comment** → type a single sentence.
   - **Multiple comments** → paste several, **one per line** (press Enter between each).
2. (Optional) Click **"Load example"** to auto-fill a sample set of comments —
   useful for a quick demo without typing anything.
3. Click **"Analyze"**.
4. Read the result:
   - **Single comment** → you'll see a full result card: a clear verdict
     (⚠️ Cyberbullying Detected / ✅ No cyberbullying), which category(ies) it
     matched, a confidence percentage for each category (shown as progress
     bars), the specific words that drove the decision **highlighted in red**
     in your original text, and a plain-English explanation of the reasoning.
   - **Multiple comments** → you'll see a summary table (flagged vs clean,
     category, confidence for each line), a pie chart and bar chart summarising
     the batch, and a **download button** to save the results as a CSV file.
5. Click **"Clear"** to empty the box and start over.

**Tip:** if a comment gets flagged and you're not sure why, check the
highlighted words and the explanation text underneath the result — that's
exactly what they're there for.

---

## 3. Social Media Detection

**What it's for:**
Instead of copying comments manually, this page can pull comments **directly
from a social media link** and analyze all of them at once.

### What you must know before using it

- ✅ **Supported platforms: YouTube and Reddit.** These are fetched through
  each platform's **official public API** — a legitimate, sanctioned way to
  read public comments.
- ❌ **Not supported: Facebook, Instagram, X/Twitter, TikTok.** These platforms'
  Terms of Service prohibit automated comment collection, and they actively
  block it. The app will **not** attempt to scrape them — it will tell you
  clearly and suggest pasting those comments manually into **Text Detection**
  instead.
- 🔑 **YouTube needs a free API key; Reddit does not.** Without a key, YouTube
  fetching won't work — see the box below for how to get one. Reddit works with
  no key at all.
- 🧪 **No key? Use Demo Mode.** A "Use demo comments" button loads a small set
  of built-in sample comments so you can try out (and demonstrate) this feature
  without any setup. The app always clearly labels this as demo data, never as
  real fetched content.

### Getting a free YouTube API key (optional, only needed for YouTube)

1. Go to [console.cloud.google.com](https://console.cloud.google.com/) and sign
   in with a Google account.
2. Create a new project (or use an existing one).
3. Search for **"YouTube Data API v3"** in the API library and click **Enable**.
4. Go to **Credentials → Create Credentials → API key**. Copy the key.
5. Paste it into the **"YouTube API key"** box in the app's sidebar (this box
   only appears on the Social Media Detection page).

Reddit doesn't require any key or account — it works out of the box.

### What to input

| Field | What to put in it |
|-------|--------------------|
| **Social media URL** | A YouTube video link (e.g. `https://www.youtube.com/watch?v=...`) or a Reddit thread link (e.g. `https://www.reddit.com/r/.../comments/...`) |
| **Max comments to fetch** | How many comments to pull (10–100). Higher = more thorough, but slower. |

### How to use it

1. (If using YouTube) paste your API key into the sidebar box.
2. Paste the URL into the **"Social media URL"** field.
3. Set how many comments to fetch with the slider.
4. Click **"Fetch comments"** — or click **"Use demo comments"** if you don't
   have a key handy and just want to try the feature.
5. Once comments are retrieved, you'll see how many were found and can expand
   **"Preview retrieved comments"** to skim them first.
6. Click **"Analyze comments"** to run detection on all of them at once.
7. Review the summary metrics, the results table, the charts, and download the
   full results as a CSV if needed.

**Common issue:** if a Reddit fetch fails, it's usually a temporary rate limit
— wait a few seconds and try again. If a YouTube fetch fails, double-check the
API key is pasted correctly and that "YouTube Data API v3" is enabled on your
Google Cloud project.

---

## 4. Batch File Detection

**What it's for:**
Analyze a large number of comments at once from a file, instead of pasting them
by hand — useful if you already have comments exported to a spreadsheet or text file.

**How to use it:**
1. Click **"Choose a file"** and upload either:
   - A **CSV file** — any spreadsheet with a column of text (e.g. exported
     comments, survey responses).
   - A **TXT file** — a plain text file with **one comment per line**.
2. If you uploaded a CSV, a preview of the first few rows appears — use the
   dropdown to tell the app **which column holds the comment text**.
3. Click **"Analyze file"**.
4. Review the summary metrics (total, flagged, flag rate), the full results
   table (with a confidence score per category for every row), and the summary
   charts.
5. Click **"Download full results (CSV)"** to save everything, including every
   category's confidence score per comment — useful for further analysis in
   Excel or for your report appendix.

---

## 5. Model Comparison

**What it's for:**
See how the three different models (Logistic Regression, SVM, Random Forest)
judge the *exact same* comment side by side — useful for demonstrating that
different algorithms can reach different conclusions, and for your assignment's
model-comparison requirement.

**How to use it:**
1. Type a comment into the **"Comment to compare"** box.
2. Click **"Compare models"**.
3. Read the comparison table: each model's verdict, matched categories, top
   confidence score, and how long it took to process.
4. Check the bar chart underneath to see each model's confidence per category
   side by side.
5. A message at the bottom tells you whether all three models **agree** or
   **disagree** on this comment — disagreements are especially useful to quote
   in your report as an example of how model choice affects results.

---

## 6. Dataset Statistics

**What it's for:**
Understand the data the models were trained on — useful for your
documentation's "Dataset" section, and for building trust in the results by
seeing where they come from.

**What's on it (no input needed — just browse):**
- Total number of training comments, number of categories, average comment
  length, and how many comments carry more than one label
- A bar chart of how many comments fall into each category
- A histogram showing how long comments typically are
- A bar chart of the most frequently occurring words after cleaning
- A preview table of sample rows from the raw dataset

**How to use it:** simply open the page and scroll — everything loads
automatically.

---

## 7. Model Evaluation

**What it's for:**
The formal performance report: how accurate each model actually is, using the
metrics required for the assignment (Accuracy, Precision, Recall, F1, Confusion
Matrix).

**How to use it:**
1. Open the page — a table of all three models' scores (Accuracy, Precision,
   Recall, F1, Hamming Loss) loads automatically, along with a bar chart
   comparing their F1 scores.
2. To see **confusion matrices** (how often each category was correctly vs.
   incorrectly predicted), click **"Compute confusion matrices"**. This runs
   the model currently selected in the sidebar over the test set and shows one
   small matrix per category — it takes a few seconds to compute.
3. Scroll down to **"Notes for the report"** for a plain-English explanation of
   what Micro-F1, Macro-F1, and Subset Accuracy mean — handy if you need to
   explain these terms in your documentation.

---

## Quick reference: which page do I need?

| I want to... | Use this page |
|---|---|
| Test one sentence quickly | Text Detection |
| Check a list of comments I already have | Text Detection (paste one per line) |
| Pull comments from a YouTube video or Reddit thread | Social Media Detection |
| Analyze a spreadsheet or text file of comments | Batch File Detection |
| See if different models agree on a comment | Model Comparison |
| Learn about the training data | Dataset Statistics |
| Get official accuracy/precision/recall/F1 numbers | Model Evaluation |
