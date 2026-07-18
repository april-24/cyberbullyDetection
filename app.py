"""
CyberShield - Multi-Model Cyberbullying Detection System
=========================================================
Streamlit application. Run from the project root:

    streamlit run app.py

Pages: Home | Text Detection | Social Media | Batch File |
       Model Comparison | Dataset Statistics | Model Evaluation
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


def result_card(res, threshold, model_name, elapsed, original_text):
    """Render one prediction result nicely."""
    if res["is_bully"]:
        st.error(f"### ⚠️ CYBERBULLYING DETECTED")
    else:
        st.success(f"### ✅ No cyberbullying detected")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model used", model_name)
    c2.metric("Top confidence", f"{max(res['probs'].values()):.1%}")
    c3.metric("Processing time", f"{elapsed*1000:.0f} ms")

    # categories
    if res["flagged"]:
        st.write("**Categories detected:**")
        for l in res["flagged"]:
            st.markdown(f"- **{pretty(l)}** — {res['probs'][l]:.1%} confidence")

    # confidence bars for every label
    st.write("**Confidence by category:**")
    for l in LABELS:
        p = res["probs"].get(l, 0.0)
        st.write(f"{pretty(l)} — {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))

    # highlighted words
    if res["words"]:
        st.write("**Influential words highlighted:**")
        st.markdown(
            f"<div style='padding:10px;border:1px solid #ddd;border-radius:6px'>"
            f"{highlight_html(original_text, res['words'])}</div>",
            unsafe_allow_html=True)

    st.info(f"**Why this result?** {explain(res)}")
    st.caption(f"Decision threshold: {threshold:.2f}. A category is flagged when "
               f"its score is at or above this value.")


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


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ CyberShield")
page = st.sidebar.radio("Navigation", [
    "Home", "Text Detection", "Social Media Detection", "Batch File Detection",
    "Model Comparison", "Dataset Statistics", "Model Evaluation"])

MODELS = available_models()
if not MODELS:
    st.sidebar.error("No trained models found in results/.")
    st.error("No trained models found. Train them first:\n\n"
             "```\npython -m models.member1_logistic_regression\n"
             "python -m models.member2_svm\n"
             "python -m models.member3_random_forest\n```")
    st.stop()

st.sidebar.markdown("---")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
threshold = st.sidebar.slider(
    "Detection sensitivity (threshold)", 0.30, 0.90, DEFAULT_THRESHOLD, 0.05,
    help="Lower = flags more comments (higher recall). Higher = stricter "
         "(higher precision). Reported metrics use the standard 0.50.")
bundle = get_model(MODELS[model_name])


# ---------------------------------------------------------------- pages
if page == "Home":
    st.title("🛡️ CyberShield")
    st.subheader("Multi-Model Cyberbullying Detection System")
    st.write("""
CyberShield analyses online comments and flags cyberbullying before it spreads.
It uses **three different machine-learning models** trained on the public
**HateXplain** dataset, and reports not just *whether* a comment is abusive but
*which group it targets*, with a confidence score and an explanation.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Models", len(MODELS))
    c2.metric("Categories", len(LABELS))
    c3.metric("Training comments", "20,109")

    st.markdown("### What it detects")
    for l in LABELS:
        st.markdown(f"- **{pretty(l)}**")

    st.markdown("### Objectives")
    st.markdown("""
1. Detect cyberbullying in short online comments.
2. Identify the targeted category (race, religion, gender, etc.).
3. Compare three NLP models on the same data.
4. Explain each prediction so users can judge it for themselves.
""")
    st.markdown("### How to use")
    st.markdown("""
- **Text Detection** — type or paste one or many comments.
- **Social Media Detection** — analyse a YouTube or Reddit link.
- **Batch File Detection** — upload a CSV/TXT of comments.
- **Model Comparison** — run all three models on the same text.
- **Dataset Statistics / Model Evaluation** — the data and the numbers.
""")
    st.info("Use the sidebar to navigate. Pick your model and sensitivity there too.")
    st.caption("Educational project. Predictions are statistical and can be wrong — "
               "always apply human judgement before acting on a result.")


elif page == "Text Detection":
    st.title("Text Detection")
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
            st.download_button("Download results (CSV)",
                               df.to_csv(index=False).encode(),
                               "cybershield_results.csv", "text/csv")


elif page == "Social Media Detection":
    st.title("Social Media Detection")
    st.info("""
**Supported:** YouTube and Reddit, through their **official public APIs**.

**Not supported:** Facebook, Instagram, X/Twitter and TikTok. Their Terms of
Service prohibit automated comment collection and they block it technically, so
this system does not attempt to scrape them. For those platforms, copy the
comments manually into the **Text Detection** page.
""")

    api_key = st.sidebar.text_input("YouTube API key (optional)", type="password",
                                    help="Free key from Google Cloud Console with "
                                         "'YouTube Data API v3' enabled.")

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
            st.download_button("Download results (CSV)",
                               df.to_csv(index=False).encode(),
                               "social_results.csv", "text/csv")


elif page == "Batch File Detection":
    st.title("Batch File Detection")
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
                st.download_button("Download full results (CSV)",
                                   df.to_csv(index=False).encode(),
                                   "batch_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read that file: {e}")


elif page == "Model Comparison":
    st.title("Model Comparison")
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
    st.write("Training data: **HateXplain** (`final_hateXplain.csv`).")
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
    st.caption("Computed live on the test set for the model selected in the sidebar.")
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
