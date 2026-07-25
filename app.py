"""
CyberShield - Multi-Model Cyberbullying Detection System
=========================================================
Streamlit application. Run from the project root:

    streamlit run app.py

Pages (top navigation bar): Home | Text Detection | Social Media |
       Batch File | Model Comparison | Dataset Statistics | Model Evaluation
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import LABELS, pretty, SCORES_CSV, DATA_DIR
from src.predictor import (available_models, load_model, predict,
                           explain, highlight_html, _label_probs)
from src.preprocessing import clean_text
from src import social

st.set_page_config(page_title="CyberShield", page_icon="🛡️", layout="wide")

DEFAULT_THRESHOLD = 0.60   # see README: reduces false positives on benign text
PAGES = ["Home", "Text Detection", "Social Media Detection", "Batch File Detection",
         "Model Comparison", "Dataset Statistics", "Model Evaluation"]
PAGE_ICONS = {"Home": "🏠", "Text Detection": "📝", "Social Media Detection": "🌐",
              "Batch File Detection": "📁", "Model Comparison": "⚖️",
              "Dataset Statistics": "📊", "Model Evaluation": "📈"}

NAVBAR_CSS = """
<style>
/* Bigger, easier-to-click buttons everywhere (nav row + in-page buttons) */
div.stButton > button {
    font-size: 16px;
    font-weight: 600;
    padding: 0.55rem 0.4rem;
    border-radius: 8px;
    width: 100%;
}
/* Slightly larger top padding so the nav bar breathes like a real navbar */
div[data-testid="stHorizontalBlock"] { gap: 0.4rem; }
</style>
"""

PAGE_GUIDES = {
    "Home": None,
    "Text Detection": (
        "**What this page does:** analyse one comment, or paste several "
        "(one per line) to check them all at once.\n\n"
        "**How to use it:** type or paste your comment(s) → click **Load "
        "example** if you want a quick demo instead → click **Analyze**. "
        "For a single comment you'll get a full breakdown (confidence, "
        "highlighted words, explanation, suggested action). For multiple "
        "comments you'll get a summary table and charts you can download."
    ),
    "Social Media Detection": (
        "**What this page does:** pulls public comments from a YouTube "
        "video or Reddit thread and analyses all of them at once.\n\n"
        "**How to use it:** paste a video/thread URL → click **Fetch "
        "comments** (or use **Demo comments** if you don't want to set up "
        "an API key) → click **Analyze comments**."
    ),
    "Batch File Detection": (
        "**What this page does:** analyse many comments at once from a "
        "file you already have.\n\n"
        "**How to use it:** upload a CSV (pick which column holds the "
        "text) or a TXT file (one comment per line) → click **Analyze "
        "file** → download the full results as CSV."
    ),
    "Model Comparison": (
        "**What this page does:** runs the *same* comment through all "
        "three models so you can see where they agree or disagree.\n\n"
        "**How to use it:** type a comment → click **Compare models**."
    ),
    "Dataset Statistics": (
        "**What this page does:** shows what's inside the training data — "
        "size, category balance, comment lengths, and common words. No "
        "input needed, just browse."
    ),
    "Model Evaluation": (
        "**What this page does:** the formal accuracy numbers (Accuracy, "
        "Precision, Recall, F1) for each model, plus confusion matrices.\n\n"
        "**How to use it:** the score table and F1 chart load automatically. "
        "Click **Compute confusion matrices** to see per-category detail "
        "for the model selected above."
    ),
}


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False)
def get_model(path):
    return load_model(path)


@st.cache_data(show_spinner=False)
def get_dataset():
    from src.data_loader import load_dataset
    df, text_col, labels = load_dataset(DATA_DIR, verbose=False)
    return df, text_col, labels


def analyze_many(bundle, texts, threshold):
    """Analyze a list of comments -> DataFrame of results."""
    cleaned = [clean_text(t) for t in texts]
    P = _label_probs(bundle["pipeline"], cleaned)
    labels = bundle["labels"]
    rows = []
    for i, t in enumerate(texts):
        probs = {l: float(P[i][j]) for j, l in enumerate(labels)}
        flagged = [l for l in labels if probs[l] >= threshold]
        rows.append({
            "Comment": t,
            "Cyberbullying": "YES" if flagged else "NO",
            "Categories": ", ".join(pretty(l) for l in flagged) or "-",
            "Confidence": round(max(probs.values()), 3),
            **{pretty(l): round(probs[l], 3) for l in labels},
        })
    return pd.DataFrame(rows)


def suggested_action(res):
    """Plain-English 'what should I do about this' guidance for one comment."""
    if not res["is_bully"]:
        return ("✅ **No action needed.** This comment doesn't cross the "
                "detection threshold. If the conversation continues, it may "
                "be worth a quick re-check later, especially if the tone shifts.")

    top_conf = max(res["probs"].values())
    cats = [pretty(l) for l in res["flagged"] if l != "abusive"]
    lines = []
    if top_conf >= 0.80:
        lines.append("⚠️ **High confidence detection** — this is worth acting on.")
    else:
        lines.append("🟡 **Moderate confidence** — have a human double-check "
                     "before acting, since the model isn't fully certain.")
    lines.append("**Suggested next steps:**")
    lines.append("- Save or screenshot the comment as evidence before it can "
                 "be edited or deleted.")
    lines.append("- Most platforms have a built-in **report** option for "
                 "harassment or hate speech — consider using it.")
    if cats:
        lines.append(f"- This appears to target **{', '.join(cats)}** — if "
                     "it's part of a repeated pattern against the same "
                     "person, consider escalating to a moderator, teacher, "
                     "or the platform's trust & safety team.")
    lines.append("- If it includes a direct threat of violence, treat it "
                 "seriously and contact local authorities.")
    lines.append("- In Malaysia, online harassment can also be reported to "
                 "the **Malaysian Communications and Multimedia Commission "
                 "(MCMC)** under the Online Safety Act 2025.")
    lines.append("\n_This is general guidance, not legal advice. For "
                 "situations involving real danger to someone, contact the "
                 "appropriate authorities directly._")
    return "\n\n".join(lines)


def batch_suggestion(df):
    """Plain-English guidance for a batch of analysed comments."""
    n = len(df)
    n_bad = int((df["Cyberbullying"] == "YES").sum())
    if n == 0:
        return ""
    rate = n_bad / n
    if n_bad == 0:
        return "✅ **No cyberbullying detected in this batch.** No action needed."
    msg = f"⚠️ **{n_bad} of {n} comments ({rate:.0%}) were flagged.**\n\n"
    if rate >= 0.3:
        msg += ("This is a high proportion — consider reviewing the source "
               "(video, thread, or file) more closely, and if it's an "
               "ongoing conversation, consider flagging it to a moderator "
               "or platform trust & safety team before it escalates.")
    else:
        msg += ("Review the flagged rows individually before taking action "
               "— sort by the **Confidence** column to prioritise the most "
               "serious ones first.")
    msg += "\n\nKeep evidence (screenshots/exports) of anything you plan to report."
    return msg


def result_card(res, threshold, model_name, elapsed, original_text):
    """Render one prediction result nicely."""
    if res["is_bully"]:
        st.error("### ⚠️ CYBERBULLYING DETECTED")
    else:
        st.success("### ✅ No cyberbullying detected")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model used", model_name)
    c2.metric("Top confidence", f"{max(res['probs'].values()):.1%}")
    c3.metric("Processing time", f"{elapsed*1000:.0f} ms")

    if res["flagged"]:
        st.write("**Categories detected:**")
        for l in res["flagged"]:
            st.markdown(f"- **{pretty(l)}** — {res['probs'][l]:.1%} confidence")

    st.write("**Confidence by category:**")
    for l in LABELS:
        p = res["probs"].get(l, 0.0)
        st.write(f"{pretty(l)} — {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))

    if res["words"]:
        st.write("**Influential words highlighted:**")
        st.markdown(
            f"<div style='padding:10px;border:1px solid #ddd;border-radius:6px'>"
            f"{highlight_html(original_text, res['words'])}</div>",
            unsafe_allow_html=True)

    st.info(f"**Why this result?** {explain(res)}")
    st.caption(f"Decision threshold: {threshold:.2f}. A category is flagged "
               f"when its score is at or above this value.")

    st.markdown("#### 🧭 Suggested next step")
    st.markdown(suggested_action(res))


def summary_charts(df):
    """Pie + bar charts summarising a batch of results."""
    c1, c2 = st.columns(2)
    with c1:
        counts = df["Cyberbullying"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=["#c0392b" if i == "YES" else "#27ae60" for i in counts.index])
        ax.set_title("Cyberbullying vs Clean")
        st.pyplot(fig); plt.close(fig)
    with c2:
        cat_cols = [pretty(l) for l in LABELS]
        flags = (df[cat_cols] >= 0.5).sum()
        fig, ax = plt.subplots(figsize=(5, 4))
        flags.plot(kind="barh", ax=ax, color="#c0392b")
        ax.set_title("Detections per category")
        st.pyplot(fig); plt.close(fig)


def page_guide(page_name):
    """Optional per-page 'how to use this' expander."""
    text = PAGE_GUIDES.get(page_name)
    if text:
        with st.expander("ℹ️ How to use this page"):
            st.markdown(text)


def page_controls(show_model=True, show_threshold=True, key_prefix=""):
    """Inline model/threshold controls - only rendered on pages that use them."""
    n = sum([show_model, show_threshold])
    if n == 0:
        return
    cols = st.columns(n)
    i = 0
    models_list = list(MODELS.keys())
    if show_model:
        with cols[i]:
            st.session_state.sel_model = st.selectbox(
                "🤖 Model", models_list,
                index=models_list.index(st.session_state.sel_model),
                key=f"{key_prefix}_model")
        i += 1
    if show_threshold:
        with cols[i]:
            st.session_state.sel_threshold = st.slider(
                "🎚️ Detection sensitivity", 0.30, 0.90,
                st.session_state.sel_threshold, 0.05, key=f"{key_prefix}_threshold",
                help="Lower = flags more comments (higher recall). Higher = "
                     "stricter (higher precision). Reported metrics use the "
                     "standard 0.50.")
    st.divider()


# ---------------------------------------------------------------- top navbar
st.markdown(NAVBAR_CSS, unsafe_allow_html=True)

st.markdown("## 🛡️ CyberShield")

MODELS = available_models()
if not MODELS:
    st.error("No trained models found. Train them first:\n\n"
             "```\npython -m models.member1_logistic_regression\n"
             "python -m models.member2_svm\n"
             "python -m models.member3_random_forest\n```")
    st.stop()

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "sel_model" not in st.session_state:
    st.session_state.sel_model = list(MODELS.keys())[0]
if "sel_threshold" not in st.session_state:
    st.session_state.sel_threshold = DEFAULT_THRESHOLD

nav_cols = st.columns(len(PAGES))
for i, p in enumerate(PAGES):
    with nav_cols[i]:
        active = st.session_state.page == p
        label = f"{PAGE_ICONS[p]} {p}"
        if st.button(label, key=f"nav_{p}", type="primary" if active else "secondary"):
            st.session_state.page = p
            st.rerun()
st.divider()

page = st.session_state.page


# ---------------------------------------------------------------- pages
if page == "Home":
    st.title("🛡️ CyberShield")
    st.subheader("Multi-Model Cyberbullying Detection System")
    st.write("""
CyberShield analyses online comments and flags cyberbullying before it spreads.
It uses **three different machine-learning models** trained on the public
**HateXplain** dataset, and reports not just *whether* a comment is abusive but
*which group it targets*, with a confidence score, an explanation, and a
suggested next step.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Models", len(MODELS))
    c2.metric("Categories", len(LABELS))
    c3.metric("Training comments", "20,109+")

    st.markdown("### What it detects")
    for l in LABELS:
        st.markdown(f"- **{pretty(l)}**")

    st.markdown("### Objectives")
    st.markdown("""
1. Detect cyberbullying in short online comments.
2. Identify the targeted category (race, religion, gender, etc.).
3. Compare three NLP models on the same data.
4. Explain each prediction and suggest what to do about it.
""")
    st.markdown("### How to use")
    st.markdown("""
- **Text Detection** — type or paste one or many comments.
- **Social Media Detection** — analyse a YouTube or Reddit link.
- **Batch File Detection** — upload a CSV/TXT of comments.
- **Model Comparison** — run all three models on the same text.
- **Dataset Statistics / Model Evaluation** — the data and the numbers.
""")
    st.info("Use the navigation bar above to switch pages. Each page has its "
            "own **ℹ️ How to use this page** guide if you need it.")
    st.caption("Educational project. Predictions are statistical and can be "
               "wrong — always apply human judgement before acting on a result.")


elif page == "Text Detection":
    st.title("Text Detection")
    page_guide(page)
    page_controls(show_model=True, show_threshold=True, key_prefix="text")
    bundle = get_model(MODELS[st.session_state.sel_model])
    threshold = st.session_state.sel_threshold
    model_name = st.session_state.sel_model

    st.write("Enter one comment, or paste several (one per line).")

    if "text_input" not in st.session_state:
        st.session_state.text_input = ""

    c1, c2 = st.columns([1, 1])
    if c1.button("Load example"):
        st.session_state.text_input = (
            "you are a stupid idiot nobody likes you\n"
            "Thanks for sharing, this was really useful!\n"
            "go back to your own country you don't belong here")
    if c2.button("Clear"):
        st.session_state.text_input = ""

    text = st.text_area("Comment(s)", key="text_input", height=160,
                        placeholder="Type a comment here...")

    if st.button("Analyze", type="primary"):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            st.warning("Please enter at least one comment.")
        elif len(lines) == 1:
            t0 = time.time()
            res = predict(bundle, lines[0], threshold=threshold)
            result_card(res, threshold, model_name, time.time() - t0, lines[0])
        else:
            st.write(f"Analyzing **{len(lines)}** comments with **{model_name}**...")
            df = analyze_many(bundle, lines, threshold)
            n_bad = (df["Cyberbullying"] == "YES").sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(df))
            c2.metric("Flagged", int(n_bad))
            c3.metric("Clean", int(len(df) - n_bad))
            st.dataframe(df[["Comment", "Cyberbullying", "Categories",
                             "Confidence"]], width="stretch")
            summary_charts(df)
            st.markdown("#### 🧭 Suggested next step")
            st.markdown(batch_suggestion(df))
            st.download_button("Download results (CSV)",
                               df.to_csv(index=False).encode(),
                               "cybershield_results.csv", "text/csv")


elif page == "Social Media Detection":
    st.title("Social Media Detection")
    page_guide(page)
    page_controls(show_model=True, show_threshold=True, key_prefix="social")
    bundle = get_model(MODELS[st.session_state.sel_model])
    threshold = st.session_state.sel_threshold

    st.info("""
**Supported:** YouTube and Reddit, through their **official public APIs**.

**Not supported:** Facebook, Instagram, X/Twitter and TikTok. Their Terms of
Service prohibit automated comment collection and they block it technically, so
this system does not attempt to scrape them. For those platforms, copy the
comments manually into the **Text Detection** page.
""")

    st.markdown("#### 🔑 About the YouTube API key")
    st.markdown(
        "**Is it free? Yes.** YouTube's official API has a generous free "
        "tier — no credit card, no charge, just a Google account and a few "
        "clicks. Reddit links need **no key at all**. Don't want to set "
        "anything up right now? Use **Demo comments** below instead — no "
        "key needed."
    )
    with st.expander("Get a free YouTube API key (one-time, ~2 minutes)"):
        st.markdown("""
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project (any name is fine)
3. Search **"YouTube Data API v3"** in the API Library → click **Enable**
4. Go to **Credentials → Create Credentials → API key** → copy it
5. Paste it in the box below
""")
    api_key = st.text_input(
        "YouTube API key (only needed for YouTube links — leave blank for "
        "Reddit or Demo mode)", type="password")

    url = st.text_input("Social media URL",
                        placeholder="https://www.youtube.com/watch?v=...  or  https://www.reddit.com/r/.../comments/...")
    n_max = st.slider("Max comments to fetch", 10, 100, 40, 10)

    c1, c2 = st.columns([1, 1])
    fetch_clicked = c1.button("Fetch comments", type="primary")
    demo_clicked = c2.button("Use demo comments (no API needed)")

    if demo_clicked:
        st.session_state.fetched = social.demo_comments()
        st.session_state.is_demo = True
    if fetch_clicked:
        if not url.strip():
            st.warning("Please paste a URL first.")
        else:
            with st.spinner("Fetching..."):
                comments, err, platform = social.fetch_comments(url, api_key, n_max)
            if err:
                st.error(err)
            else:
                st.session_state.fetched = comments
                st.session_state.is_demo = False
                st.success(f"Fetched {len(comments)} comments from {platform}.")

    comments = st.session_state.get("fetched", [])
    if comments:
        if st.session_state.get("is_demo"):
            st.warning("Showing **demo sample comments** — not real fetched data.")
        st.metric("Comments retrieved", len(comments))
        with st.expander("Preview retrieved comments"):
            for c in comments[:15]:
                st.write("-", c)

        if st.button("Analyze comments", type="primary"):
            df = analyze_many(bundle, comments, threshold)
            n_bad = (df["Cyberbullying"] == "YES").sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Analyzed", len(df))
            c2.metric("Flagged", int(n_bad))
            c3.metric("Flag rate", f"{n_bad/len(df):.0%}")
            st.dataframe(df[["Comment", "Cyberbullying", "Categories",
                             "Confidence"]], width="stretch")
            summary_charts(df)
            st.markdown("#### 🧭 Suggested next step")
            st.markdown(batch_suggestion(df))
            st.download_button("Download results (CSV)",
                               df.to_csv(index=False).encode(),
                               "social_results.csv", "text/csv")


elif page == "Batch File Detection":
    st.title("Batch File Detection")
    page_guide(page)
    page_controls(show_model=True, show_threshold=True, key_prefix="batch")
    bundle = get_model(MODELS[st.session_state.sel_model])
    threshold = st.session_state.sel_threshold

    st.write("Upload a **CSV** (with a text column) or a **TXT** file (one comment per line).")

    up = st.file_uploader("Choose a file", type=["csv", "txt"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                raw = pd.read_csv(up)
                st.write("Preview:")
                st.dataframe(raw.head(), width="stretch")
                col = st.selectbox("Which column holds the comment text?",
                                   list(raw.columns))
                texts = raw[col].dropna().astype(str).tolist()
            else:
                texts = [l.strip() for l in
                         up.read().decode("utf-8", errors="ignore").split("\n")
                         if l.strip()]
            st.success(f"Loaded {len(texts)} comments.")

            if st.button("Analyze file", type="primary"):
                with st.spinner("Analyzing..."):
                    df = analyze_many(bundle, texts, threshold)
                n_bad = (df["Cyberbullying"] == "YES").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(df))
                c2.metric("Flagged", int(n_bad))
                c3.metric("Flag rate", f"{n_bad/max(len(df),1):.0%}")
                st.dataframe(df, width="stretch")
                summary_charts(df)
                st.markdown("#### 🧭 Suggested next step")
                st.markdown(batch_suggestion(df))
                st.download_button("Download full results (CSV)",
                                   df.to_csv(index=False).encode(),
                                   "batch_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read that file: {e}")


elif page == "Model Comparison":
    st.title("Model Comparison")
    page_guide(page)
    page_controls(show_model=False, show_threshold=True, key_prefix="compare")
    threshold = st.session_state.sel_threshold

    st.write("Run **all three models** on the same comment and compare them.")

    text = st.text_input("Comment to compare",
                         "you are a stupid idiot nobody likes you")
    if st.button("Compare models", type="primary") and text.strip():
        rows = []
        for name, path in MODELS.items():
            b = get_model(path)
            t0 = time.time()
            r = predict(b, text, threshold=threshold)
            rows.append({
                "Model": name,
                "Prediction": "CYBERBULLYING" if r["is_bully"] else "Clean",
                "Categories": ", ".join(pretty(l) for l in r["flagged"]) or "-",
                "Top confidence": round(max(r["probs"].values()), 3),
                "Time (ms)": round((time.time() - t0) * 1000, 1),
                **{pretty(l): round(r["probs"][l], 3) for l in LABELS},
            })
        cmp = pd.DataFrame(rows)
        st.dataframe(cmp[["Model", "Prediction", "Categories",
                          "Top confidence", "Time (ms)"]], width="stretch")

        st.write("**Confidence per category, by model:**")
        chart_df = cmp.set_index("Model")[[pretty(l) for l in LABELS]]
        st.bar_chart(chart_df.T)

        if cmp["Prediction"].nunique() > 1:
            st.warning("The models disagree on this comment — a good example for "
                       "your report of how algorithm choice changes the outcome.")
        else:
            st.success("All three models agree on this comment.")


elif page == "Dataset Statistics":
    st.title("Dataset Statistics")
    page_guide(page)
    try:
        from src.data_loader import find_csv
        source_file = os.path.basename(find_csv(DATA_DIR))
    except Exception:
        source_file = "final_hateXplain.csv"
    st.write(f"Training data source: **`{source_file}`**")

    try:
        df, text_col, labels = get_dataset()
    except Exception as e:
        st.error(f"Could not load the dataset: {e}")
        st.stop()

    lengths = df[text_col].astype(str).str.split().apply(len)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comments", f"{len(df):,}")
    c2.metric("Categories", len(labels))
    c3.metric("Avg length", f"{lengths.mean():.1f} words")
    c4.metric("Multi-label", f"{(df[labels].sum(axis=1) >= 2).sum():,}")

    st.subheader("Comments per category")
    counts = df[labels].sum().sort_values(ascending=False)
    counts.index = [pretty(i) for i in counts.index]
    st.bar_chart(counts)

    st.subheader("Comment length distribution")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(lengths, bins=50, color="#2980b9")
    ax.set_xlim(0, lengths.quantile(0.99))
    ax.set_xlabel("Words per comment")
    st.pyplot(fig); plt.close(fig)

    st.subheader("Most frequent words (after cleaning)")
    from collections import Counter
    sample = df[text_col].sample(min(4000, len(df)), random_state=42)
    words = Counter(" ".join(clean_text(t) for t in sample).split())
    top = pd.Series(dict(words.most_common(25)))
    st.bar_chart(top)

    st.subheader("Sample rows")
    st.dataframe(df.head(20), width="stretch")


elif page == "Model Evaluation":
    st.title("Model Performance Evaluation")
    page_guide(page)
    page_controls(show_model=True, show_threshold=False, key_prefix="eval")
    bundle = get_model(MODELS[st.session_state.sel_model])

    st.write("Metrics computed on the held-out 20% test set at the standard "
             "0.50 threshold.")

    if not os.path.exists(SCORES_CSV):
        st.warning("No scores yet — train the models first.")
        st.stop()

    scores = pd.read_csv(SCORES_CSV)
    st.dataframe(scores, width="stretch")

    st.subheader("F1 comparison")
    chart = scores.set_index("model")[["f1_micro", "f1_macro"]]
    st.bar_chart(chart)

    st.subheader("Confusion matrix (per category)")
    st.caption("Computed live on the test set for the model selected above.")
    if st.button("Compute confusion matrices"):
        from sklearn.metrics import confusion_matrix
        from src.common import prepare_data
        with st.spinner("Running the model over the test set..."):
            X_train, X_test, y_train, y_test, labels = prepare_data(DATA_DIR, verbose=False)
            P = _label_probs(bundle["pipeline"], list(X_test))
            pred = (P >= 0.5).astype(int)
        cols = st.columns(3)
        for i, l in enumerate(labels):
            cm = confusion_matrix(y_test.values[:, i], pred[:, i])
            fig, ax = plt.subplots(figsize=(2.6, 2.4))
            ax.imshow(cm, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, cm[a, b], ha="center", va="center", fontsize=9)
            ax.set_title(pretty(l), fontsize=9)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            cols[i % 3].pyplot(fig); plt.close(fig)

    st.subheader("Notes for the report")
    st.markdown("""
- **Micro-F1** aggregates over all label decisions, so frequent labels dominate.
- **Macro-F1** averages each category equally, so rare categories matter more —
  it drops when a model handles minority categories poorly.
- **Subset accuracy** requires *every* one of the six labels to be right at once,
  which is why it looks low; that is normal for multi-label problems.
""")
