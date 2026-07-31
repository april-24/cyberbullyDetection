# 🇲🇾 Collecting Malaysian Cyberbullying Data — Full Guide

This guide covers how to legally collect more comment data (with a focus on
Malaysia), label it, and merge it into your training set to make the models
more locally relevant.

**Workflow:** Crawl (unlabeled) → Annotate (your group labels it) → Merge →
Retrain.

---

## 1. What's legal, and what isn't

| Source | Status | Why |
|--------|--------|-----|
| **YouTube** (official API) | ✅ Recommended | Official, sanctioned, free, instant key |
| **Reddit** (official API) | ⚠️ Usable, but slow to set up | As of Nov 2025, Reddit requires manual pre-approval for API access — even the free tier — which can take **2–4 weeks**. Apply early if you want this. |
| **Existing Reddit datasets on Kaggle** | ✅ Fast alternative | A pre-collected r/malaysia dataset already exists on Kaggle (search "Reddit r/Malaysia Subreddit Dataset") — skips the approval wait entirely |
| **Public forums / news comments** (generic crawler) | ⚠️ Case-by-case | Only if the site's `robots.txt` allows it AND their Terms of Service don't prohibit it. Check both yourself before crawling any specific site. |
| **Facebook, Instagram, X/Twitter, TikTok** | ❌ Do not scrape | Their Terms of Service explicitly prohibit automated data collection, and they block it technically. This isn't just a policy choice here — it's a real legal/ToS risk for you as a student. If you need comments from these platforms, copy them manually into the app's Cyberbully Detection → Enter Comment tab (small amounts only, for demo purposes). |

**Why this matters for your assignment specifically:** the marking scheme
checks for originality and academic integrity. Scraper code that breaches a
platform's ToS is a bad look in a report that's also graded on ethics — and
it's genuinely unnecessary, since YouTube alone gives you plenty of real
Malaysian discourse to work with.

**Context worth citing in your documentation:** Malaysia's **Online Safety Act
2025** came into force on 1 January 2026, extending the Malaysian
Communications and Multimedia Commission's (MCMC) oversight to platforms with
over 8 million Malaysian users. It's directly relevant background for why a
cyberbullying detection tool for the Malaysian context matters right now.

---

## 2. Recommended path: YouTube (start here)

**Setup (2 minutes, no waiting):**
1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Create/select a project
3. **APIs & Services → Library** → search "YouTube Data API v3" → **Enable**
4. **APIs & Services → Credentials → Create Credentials → API key** → copy it

**Run the crawler:**
1. Open `crawler/youtube_batch_crawler.py`
2. Paste your key into `YOUTUBE_API_KEY` (or set it as an environment variable)
3. Adjust `SEARCH_QUERIES` if you want — the defaults mix English and Malay
   queries ("Malaysia news debate", "berita Malaysia terkini", etc.) chosen to
   be **neutral topic searches** rather than anything designed to bait hateful
   replies
4. Run:
   ```bash
   python crawler/youtube_batch_crawler.py
   ```
5. Output: `data/crawled/youtube_malaysia_raw.csv` — **unlabeled**, ready for
   the annotation step below.

You can re-run it any time — it automatically skips comments it already has.

---

## 3. Optional: Reddit

Two options, pick based on your timeline:

**Option A — fast, no live crawling needed:** Download the existing "Reddit
r/Malaysia Subreddit Dataset" from Kaggle and adapt it to the raw-comment CSV
format the annotation tool expects (just a `comment` column is enough).

**Option B — live crawling via the official API:**
1. Apply for Reddit API access now if you want to go this route — approval
   can take 2–4 weeks under their current policy, so don't leave it to the
   last minute.
2. `pip install praw`
3. Register an app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
   ("script" type), get your client ID and secret
4. Fill them into `crawler/reddit_crawler.py`, then:
   ```bash
   python crawler/reddit_crawler.py
   ```
5. Output: `data/crawled/reddit_malaysia_raw.csv`

---

## 4. Optional: other public sites (generic crawler)

For a forum or news site not covered by an official API:

1. Open the target page in your browser, right-click a comment → **Inspect**,
   and note the CSS class/tag wrapping the comment text.
2. Edit `crawler/generic_crawler.py` → add your URL and selector to `TARGETS`.
3. Run:
   ```bash
   pip install requests beautifulsoup4
   python crawler/generic_crawler.py
   ```

This script **automatically checks `robots.txt` first** and refuses to crawl
any page that disallows it. It still won't tell you whether the site's Terms
of Service allow scraping — that's on you to check before adding a URL.

---

## 5. Annotate: turn raw comments into labeled data

Crawled comments have no labels yet — you have to decide, per comment,
whether it's abusive and what it targets. This is normal for a hate-speech
dataset and is genuine, gradeable work for your documentation.

**You do not need to label everything you crawled.** Even 50–100 labeled
comments is a legitimate, honest extension — label an amount that fits your
timeline, not everything you collected.

```bash
streamlit run crawler/annotate_data.py
```

- Pick which raw file to label from the dropdown.
- For each comment: mark **Yes/No** for abusive, and tick any target
  categories that apply (Race, Religion, Gender, Sexual Orientation,
  Miscellaneous).
- Click **"Skip"** for anything you're unsure about or can't read (e.g.
  heavy slang, mixed language you're not confident on) — better to skip than
  guess.
- Progress saves after **every single comment**, so you can close it and
  resume anytime.
- **Split the work across your group.** Progress saves to the same output
  file after each comment, and the tool automatically skips anything already
  labeled — so one member can label 50, close it, and a teammate opens the
  same file afterwards and continues exactly where it was left off, with no
  overlap or duplicates. Just make sure everyone works from the same copy of
  the file (e.g. push to GitHub after your batch, teammate pulls before
  starting theirs).

Output: `data/crawled/<name>_labeled.csv`, already in the format the rest of
the project expects.

---

## 6. Merge into your training data

Once you've labeled at least one file:

```bash
python crawler/merge_datasets.py
```

This combines the original HateXplain data with everything in
`data/crawled/*_labeled.csv`, removes duplicates, shuffles, and saves
`data/combined_dataset.csv`. **No other code changes needed** — the loader
automatically prefers this combined file over the original once it exists.

## 7. Retrain

```bash
python -m models.member1_logistic_regression
python -m models.member2_svm
python -m models.member3_random_forest
python compare_models.py
```

Your models are now trained on a mix of the original English hate-speech
data plus real Malaysian comments (Malay, English, and Manglish) that your
group personally labeled.

---

## Notes for your documentation

- **This is a legitimate way to demonstrate "extra effort" for an Excellent
  grade** — it's genuinely new data, a new skill (data annotation, API
  integration), and shows the ability to extend a public dataset with
  locally-relevant examples.
- **Be honest about data quality in your report.** Manually labeled data from
  a small group has some subjectivity — mention this as a limitation.
  If time allows, having two members independently label the same subset and
  comparing agreement (inter-annotator agreement) is a strong bonus point.
- **Watch for class imbalance.** A handful of newly labeled Malaysian
  comments won't shift a 20,000-row dataset much — the more you label, the
  more it actually influences the trained model. Track this before/after in
  `merge_datasets.py`'s printed label counts.
- **Language note (measured, not just theoretical):** the current models were
  trained on English HateXplain data, so their TF-IDF vocabulary is almost
  entirely English. Testing the trained Logistic Regression model on genuine
  Malay insults showed only ~14% of the comment's words were even recognized,
  and it confidently (but wrongly) predicted "not abusive" both times. This is
  exactly why manually labeling real Malay/Manglish comments — not just
  running them through the existing model — is the part of this pipeline that
  actually matters. It's also a legitimate, citable limitation for your
  report: "the baseline model has near-zero signal on Malay-language input,
  which is precisely the gap this data collection effort addresses."
- **Preprocessing is English-tuned**, too (`src/preprocessing.py` uses English
  stopwords and an English lemmatizer). Malay/Manglish text still gets basic
  cleaning (URLs/punctuation stripped) but won't be stemmed/lemmatized
  correctly — worth flagging as a limitation, or as a "future work" item if
  your group wants to extend it with a Malay-language NLP library.
