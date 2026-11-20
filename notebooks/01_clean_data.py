"""
Section 1: Data cleaning and preprocessing
Fake Job Postings (EMSCAD) dataset

Reproduces the preprocessing in Section 3.1 of the dissertation:
  - de-duplication of reposted adverts
  - removal of attributes excluded from the analysis
  - replacement of remaining nulls with blank strings
  - extraction of URLs and hashtags into new columns
  - splitting location into a Country column
  - text normalisation: lowercasing, punctuation and stopword removal, Porter stemming
  - VADER sentiment polarity and label per posting (Hutto & Gilbert, 2014)

Input : data/fake_job_postings.csv
Output: data/cleaned_fake_job_postings.csv
        reports/01_cleaning_log.txt
"""
import html
import re
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "fake_job_postings.csv"
CLEAN_PATH = ROOT / "data" / "cleaned_fake_job_postings.csv"
LOG_PATH = ROOT / "reports" / "01_cleaning_log.txt"

TEXT_COLS = ["company_profile", "description", "requirements", "benefits"]

# Attributes excluded in Section 3.1 of the dissertation.
# telecommuting is 95.7% zeros (prevalence above 50%). The rest are listed as excluded
# in the dissertation; required_education is only 45% missing, so it does not meet the
# 50% rule and is excluded here to match the written method.
DROP_COLS = ["department", "salary_range", "telecommuting", "required_education"]

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#\w+")
TOKEN_RE = re.compile(r"[a-z]+")  # keeps letters only, which removes punctuation and digits

log = []

# ---------------------------------------------------------------
# Section 1.1: Load raw data
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
log.append(f"Raw rows: {len(df):,}")
log.append(f"Raw columns: {df.shape[1]}")
log.append(f"Fraudulent postings in raw data: {int(df['fraudulent'].sum()):,}")

# ---------------------------------------------------------------
# Section 1.2: De-duplicate reposted adverts
# Scammers repost identical adverts. job_id is unique by construction, so duplicates
# are detected on content instead.
# ---------------------------------------------------------------
content_cols = ["title", "location"] + TEXT_COLS
before = len(df)
df = df.drop_duplicates(subset=content_cols, keep="first").reset_index(drop=True)
log.append(f"Duplicate adverts removed: {before - len(df):,}")
log.append(f"Rows after de-duplication: {len(df):,}")

# ---------------------------------------------------------------
# Section 1.3: Drop excluded attributes
# ---------------------------------------------------------------
df = df.drop(columns=DROP_COLS)
log.append(f"Dropped columns: {', '.join(DROP_COLS)}")

# ---------------------------------------------------------------
# Section 1.4: Replace remaining nulls with blank strings
# ---------------------------------------------------------------
null_before = df.isnull().sum()
log.append("Nulls replaced with blanks (count per column):")
for col, n in null_before[null_before > 0].items():
    log.append(f"    {col}: {n:,}")
df = df.fillna("")

# ---------------------------------------------------------------
# Section 1.4b: Decode HTML entities
# The text contains entities such as &amp; (about 16,600 occurrences) and &nbsp;.
# Without decoding, 'amp' and 'nbsp' become frequent tokens in the analysis.
# This step was added after the first EDA run exposed the problem.
# ---------------------------------------------------------------
for col in TEXT_COLS:
    df[col] = df[col].apply(html.unescape)

# ---------------------------------------------------------------
# Section 1.5: Split location into Country
# Location looks like "US, NY, New York". The first comma-separated part is the country.
# ---------------------------------------------------------------
df["country"] = df["location"].str.split(",").str[0].str.strip()
log.append(f"Unique countries: {df['country'].replace('', np.nan).nunique()}")

# ---------------------------------------------------------------
# Section 1.6: Extract URLs and hashtags into new columns
# For each text column: the extracted values, plus counts of each.
# ---------------------------------------------------------------
for col in TEXT_COLS:
    df[f"{col}_urls"] = df[col].apply(lambda s: "|".join(URL_RE.findall(s)))
    df[f"{col}_hashtags"] = df[col].apply(lambda s: "|".join(HASHTAG_RE.findall(s)))
    df[f"{col}_url_count"] = df[col].apply(lambda s: len(URL_RE.findall(s)))
    df[f"{col}_hashtag_count"] = df[col].apply(lambda s: len(HASHTAG_RE.findall(s)))

# ---------------------------------------------------------------
# Section 1.7: Combine text columns and score sentiment
# The combined raw text (original casing, URLs and hashtags included) is scored with
# VADER, which is designed for unprocessed text (Hutto & Gilbert, 2014).
# ---------------------------------------------------------------
df["text_raw"] = df[TEXT_COLS].agg(" ".join, axis=1)

analyzer = SentimentIntensityAnalyzer()
df["sentiment_compound"] = df["text_raw"].apply(
    lambda t: analyzer.polarity_scores(t)["compound"]
)
df["sentiment_label"] = np.select(
    [df["sentiment_compound"] <= -0.05, df["sentiment_compound"] >= 0.05],
    ["negative", "positive"],
    default="neutral",
)

# ---------------------------------------------------------------
# Section 1.8: Text normalisation
# Order: remove URLs and hashtags (already extracted), lowercase, keep letters only,
# remove English stopwords, then Porter-stem each token.
# ---------------------------------------------------------------
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def stem(word):
    return stemmer.stem(word)


def clean_text(text):
    text = URL_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    tokens = TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(stem(t) for t in tokens)


df["text_clean"] = df["text_raw"].apply(clean_text)
log.append(
    f"Average tokens per posting after cleaning: "
    f"{df['text_clean'].str.split().str.len().mean():.1f}"
)

# ---------------------------------------------------------------
# Section 1.9: Save cleaned dataset
# Raw text columns are dropped from the output. The raw CSV in data/ keeps the originals.
# ---------------------------------------------------------------
df = df.drop(columns=["location", "text_raw"] + TEXT_COLS)
df.to_csv(CLEAN_PATH, index=False)

log.append(f"Cleaned dataset rows: {len(df):,}")
log.append(f"Cleaned dataset columns: {df.shape[1]}")
log.append(f"Fraudulent postings after cleaning: {int(df['fraudulent'].sum()):,}")
log.append(f"Sentiment label counts: {df['sentiment_label'].value_counts().to_dict()}")

LOG_PATH.parent.mkdir(exist_ok=True)
LOG_PATH.write_text("\n".join(log) + "\n")
print("\n".join(log))
print(f"\nSaved cleaned data to {CLEAN_PATH}")
