"""
Shared helpers for the analysis scripts (03_text_figures.py and 04_predictive.py).

Split: stratified 80/20 so both sets keep the fraud rate (seed 42).
Oversampling: random oversampling of the TRAINING set only, as described in
Section 3.2 of the dissertation. The test set always keeps the real class mix.
Features: TF-IDF of text_clean plus the numeric dataset features in FLAG_COLS.
"""
import html
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "fake_job_postings.csv"
DATA_PATH = ROOT / "data" / "cleaned_fake_job_postings.csv"
VIS = ROOT / "visuals"
REP = ROOT / "reports"
VIS.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

SEED = 42
TEXT_COLS = ["company_profile", "description", "requirements", "benefits"]
FLAG_COLS = (["has_company_logo", "has_questions"]
             + [f"{c}_url_count" for c in TEXT_COLS]
             + [f"{c}_hashtag_count" for c in TEXT_COLS])
CLASS_NAMES = {0: "Genuine", 1: "Fraudulent"}


def load_clean():
    df = pd.read_csv(DATA_PATH)
    df["text_clean"] = df["text_clean"].fillna("")
    return df


def split_data(df):
    return train_test_split(df, test_size=0.2, stratify=df["fraudulent"],
                            random_state=SEED)


def random_oversample(X, y):
    """Duplicate minority-class rows at random until both classes are equal in size."""
    rng = np.random.default_rng(SEED)
    idx_maj = np.flatnonzero(y == 0)
    idx_min = np.flatnonzero(y == 1)
    idx_min_up = rng.choice(idx_min, size=len(idx_maj), replace=True)
    idx = rng.permutation(np.concatenate([idx_maj, idx_min_up]))
    return X[idx], y[idx]


def build_matrix(text_matrix, frame):
    """Stack TF-IDF text features with the numeric dataset features."""
    flags = csr_matrix(frame[FLAG_COLS].values.astype(float))
    return hstack([text_matrix, flags]).tocsr()


def feature_names(vectorizer):
    return np.concatenate([vectorizer.get_feature_names_out(), FLAG_COLS])


_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(word):
    return _stemmer.stem(word)


def stem_display_lookup():
    """
    Map each stem to the most common original word that produces it, for chart labels
    only. Example: 'experi' is displayed as 'experience'. The analysis itself still uses
    stems, which is what the cleaned dataset contains.
    """
    nltk.download("stopwords", quiet=True)
    stop = set(stopwords.words("english"))
    raw = pd.read_csv(RAW_PATH)
    counts = defaultdict(Counter)
    for text in raw[TEXT_COLS].fillna("").agg(" ".join, axis=1):
        for word in re.findall(r"[a-z]+", html.unescape(text).lower()):
            if word not in stop:
                counts[_stem(word)][word] += 1
    return {stem: c.most_common(1)[0][0] for stem, c in counts.items()}
