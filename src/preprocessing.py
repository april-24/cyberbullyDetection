"""
preprocessing.py
----------------
Shared text-cleaning pipeline used by ALL three models so the comparison is fair.

Pipeline steps:
    1. lowercase
    2. strip URLs, HTML, @mentions, #hashtag symbol, digits, punctuation
    3. tokenize
    4. remove stopwords
    5. lemmatize (reduce words to their base form)

It uses NLTK when available. If NLTK data can't be downloaded (common on locked-down
machines), it falls back to a built-in stopword list and skips lemmatization, so the
code still runs end-to-end.
"""

import re

# ----------------------------------------------------------------------------
# Try to set up NLTK; fall back gracefully if unavailable.
# ----------------------------------------------------------------------------
_USE_NLTK = True
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _USE_NLTK = False
    _LEMMATIZER = None
    # Minimal fallback stopword list
    _STOPWORDS = set("""
        a an the and or but if while is are was were be been being to of in on for with
        as by at from into this that these those it its i you he she we they them his her
        their our your my me him us do does did done have has had not no nor so than too
        very can will just don should now
    """.split())

# Pre-compiled regex patterns (compiled once for speed)
_RE_URL = re.compile(r"http\S+|www\.\S+")
_RE_HTML = re.compile(r"<.*?>")
_RE_MENTION = re.compile(r"@\w+")
_RE_NONALPHA = re.compile(r"[^a-z\s]")
_RE_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply the full cleaning + normalization pipeline to a single string."""
    text = str(text).lower()
    text = _RE_URL.sub(" ", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)
    text = text.replace("#", " ")
    text = _RE_NONALPHA.sub(" ", text)          # keep letters only
    text = _RE_SPACES.sub(" ", text).strip()

    # tokenize
    if _USE_NLTK:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()

    # remove stopwords + very short tokens, then lemmatize
    out = []
    for tok in tokens:
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        if _LEMMATIZER is not None:
            tok = _LEMMATIZER.lemmatize(tok)
        out.append(tok)
    return " ".join(out)


def preprocess_series(series):
    """Vectorized helper: clean an entire pandas Series of texts."""
    return series.astype(str).apply(clean_text)


if __name__ == "__main__":
    samples = [
        "You are SO stupid!!! @idiot check http://x.com #loser",
        "Have a great day everyone :)",
    ]
    for s in samples:
        print(f"{s!r}\n  -> {clean_text(s)!r}\n")
