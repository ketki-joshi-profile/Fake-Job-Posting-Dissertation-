"""
Section 2: Exploratory data analysis
Reproduces the exploratory analysis in Sections 3.2 and 5.1 of the dissertation:
  - class distribution
  - correlation of attributes with the Fraudulent target
  - word clouds with normalised term frequencies per class
  - term presence comparison for common and potentially deceptive terms
  - TF-IDF top terms per class
  - sentiment by class

Input : data/cleaned_fake_job_postings.csv
Output: visuals/*.png
        reports/02_*.csv
        reports/02_eda_summary.txt
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_fake_job_postings.csv"
VIS = ROOT / "visuals"
REP = ROOT / "reports"
VIS.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

TEXT_COLS = ["company_profile", "description", "requirements", "benefits"]
CLASS_NAMES = {0: "Genuine", 1: "Fraudulent"}
CLASS_COLORS = ["#4C72B0", "#C44E52"]

# Terms from Section 3.2 and 5.1 of the dissertation, stemmed to match text_clean
TERMS = ["experience", "business", "email", "rapid", "candidates", "bonus", "estate",
         "work", "customer", "team", "ability"]
stemmer = PorterStemmer()
TERM_STEMS = {t: stemmer.stem(t) for t in TERMS}

summary = []


def save_fig(fig, name):
    path = VIS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)}")


# ---------------------------------------------------------------
# Section 2.1: Load cleaned data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["text_clean"] = df["text_clean"].fillna("")
n_total = len(df)
summary.append(f"Postings analysed: {n_total:,}")

# ---------------------------------------------------------------
# Section 2.2: Class distribution
# ---------------------------------------------------------------
class_counts = df["fraudulent"].value_counts().sort_index()
class_pct = (class_counts / n_total * 100).round(2)
summary.append("\nClass distribution:")
for label, n in class_counts.items():
    summary.append(f"  {CLASS_NAMES[label]}: {n:,} ({class_pct[label]}%)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([CLASS_NAMES[i] for i in class_counts.index], class_counts.values, color=CLASS_COLORS)
for i, v in enumerate(class_counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_title("Class distribution of job postings")
ax.set_ylabel("Number of postings")
save_fig(fig, "01_class_distribution.png")

# ---------------------------------------------------------------
# Section 2.3: Correlation of attributes with the Fraudulent target
# URL and hashtag features are converted to presence (1 = at least one) for readability.
# ---------------------------------------------------------------
corr_df = pd.DataFrame({
    "has_company_logo": df["has_company_logo"],
    "has_questions": df["has_questions"],
})
for col in TEXT_COLS:
    corr_df[f"{col}_has_url"] = (df[f"{col}_url_count"] > 0).astype(int)
    corr_df[f"{col}_has_hashtag"] = (df[f"{col}_hashtag_count"] > 0).astype(int)
corr_df["sentiment_compound"] = df["sentiment_compound"]
corr_df["fraudulent"] = df["fraudulent"]

corr = corr_df.corr()
corr_with_target = corr["fraudulent"].drop("fraudulent").sort_values()
corr_with_target.round(4).to_csv(REP / "02_correlation_with_fraudulent.csv",
                                 header=["correlation_with_fraudulent"])

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("Correlation matrix (binary features, sentiment and target)")
save_fig(fig, "02_correlation_matrix.png")

summary.append("\nCorrelation with Fraudulent (sorted):")
for name, value in corr_with_target.items():
    summary.append(f"  {name}: {value:+.4f}")

# ---------------------------------------------------------------
# Section 2.4: Word clouds with normalised term frequencies per class
# Normalised frequency = share of postings in the class containing the term.
# Word clouds show stemmed tokens (for example "experi" for "experience").
# ---------------------------------------------------------------
def normalised_term_freq(texts):
    doc_freq = Counter()
    for text in texts:
        doc_freq.update(set(text.split()))
    n = len(texts)
    return {term: count / n for term, count in doc_freq.items()}


freqs = {}
for label in [0, 1]:
    texts = df.loc[df["fraudulent"] == label, "text_clean"]
    freqs[label] = normalised_term_freq(texts)
    wc = WordCloud(width=1200, height=600, background_color="white",
                   max_words=150, colormap="viridis")
    wc.generate_from_frequencies(freqs[label])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word cloud: {CLASS_NAMES[label]} postings")
    save_fig(fig, f"03_wordcloud_{CLASS_NAMES[label].lower()}.png")

# ---------------------------------------------------------------
# Section 2.5: Term presence by class
# Share of postings in each class that contain each term.
# ---------------------------------------------------------------
rows = []
for term in TERMS:
    stem_ = TERM_STEMS[term]
    row = {"term": term, "stem": stem_}
    for label in [0, 1]:
        share = freqs[label].get(stem_, 0.0) * 100
        row[f"{CLASS_NAMES[label].lower()}_pct_postings"] = round(share, 2)
    rows.append(row)
term_table = pd.DataFrame(rows)
term_table.to_csv(REP / "02_term_presence_by_class.csv", index=False)
summary.append("\nTerm presence (% of postings in each class):")
summary.append(term_table.to_string(index=False))

# ---------------------------------------------------------------
# Section 2.6: TF-IDF top terms per class
# Ten terms with the highest mean TF-IDF weight in each class.
# min_df=5 drops very rare tokens. This threshold is my choice; the dissertation
# does not state one.
# ---------------------------------------------------------------
vectorizer = TfidfVectorizer(min_df=5)
X = vectorizer.fit_transform(df["text_clean"])
vocab = np.array(vectorizer.get_feature_names_out())
print(f"TF-IDF vocabulary size: {len(vocab):,}")

tfidf_rows = []
top_terms = {}
for label in [0, 1]:
    mask = df["fraudulent"].values == label
    mean_weight = np.asarray(X[mask].mean(axis=0)).ravel()
    top_idx = mean_weight.argsort()[::-1][:10]
    top_terms[label] = (vocab[top_idx], mean_weight[top_idx])
    for rank, idx in enumerate(top_idx, start=1):
        tfidf_rows.append({
            "class": CLASS_NAMES[label],
            "rank": rank,
            "term": vocab[idx],
            "mean_tfidf": round(mean_weight[idx], 5),
        })
tfidf_table = pd.DataFrame(tfidf_rows)
tfidf_table.to_csv(REP / "02_top_tfidf_terms_by_class.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, label in zip(axes, [0, 1]):
    terms_, weights = top_terms[label]
    ax.barh(terms_[::-1], weights[::-1], color=CLASS_COLORS[label])
    ax.set_title(f"Top 10 TF-IDF terms: {CLASS_NAMES[label]}")
    ax.set_xlabel("Mean TF-IDF weight")
save_fig(fig, "04_top_tfidf_terms.png")

summary.append("\nTop 10 TF-IDF terms per class:")
for label in [0, 1]:
    summary.append(f"  {CLASS_NAMES[label]}: " + ", ".join(top_terms[label][0]))

# ---------------------------------------------------------------
# Section 2.6b: Most discriminative TF-IDF terms
# Ranking by mean weight favours words common to both classes. Ranking by the
# difference in mean weight (fraudulent minus genuine) shows which terms separate them.
# ---------------------------------------------------------------
mean_genuine = np.asarray(X[df["fraudulent"].values == 0].mean(axis=0)).ravel()
mean_fraud = np.asarray(X[df["fraudulent"].values == 1].mean(axis=0)).ravel()
diff = mean_fraud - mean_genuine
disc_rows = []
for label, idx_list in [("Fraudulent-leaning", diff.argsort()[::-1][:10]),
                        ("Genuine-leaning", diff.argsort()[:10])]:
    for rank, idx in enumerate(idx_list, start=1):
        disc_rows.append({
            "direction": label,
            "rank": rank,
            "term": vocab[idx],
            "mean_tfidf_genuine": round(mean_genuine[idx], 5),
            "mean_tfidf_fraudulent": round(mean_fraud[idx], 5),
            "difference": round(diff[idx], 5),
        })
disc_table = pd.DataFrame(disc_rows)
disc_table.to_csv(REP / "02_discriminative_tfidf_terms.csv", index=False)
summary.append("\nMost discriminative TF-IDF terms (difference in mean weight):")
for label in ["Fraudulent-leaning", "Genuine-leaning"]:
    terms_ = disc_table.loc[disc_table["direction"] == label, "term"].tolist()
    summary.append(f"  {label}: " + ", ".join(terms_))

# ---------------------------------------------------------------
# Section 2.7: Sentiment by class
# Mean VADER compound score and label shares, computed in 01_clean_data.py.
# ---------------------------------------------------------------
sent_mean = df.groupby("fraudulent")["sentiment_compound"].mean().rename(index=CLASS_NAMES)
sent_share = (pd.crosstab(df["fraudulent"], df["sentiment_label"], normalize="index") * 100)
sent_share.index = [CLASS_NAMES[i] for i in sent_share.index]
sent_out = sent_share.round(2).copy()
sent_out.insert(0, "mean_compound", sent_mean.round(4))
sent_out.to_csv(REP / "02_sentiment_by_class.csv")

fig, ax = plt.subplots(figsize=(7, 4))
sent_share[["negative", "neutral", "positive"]].plot(
    kind="barh", stacked=True, ax=ax,
    color=["#8C8C8C", "#D9D9D9", "#55A868"])
ax.set_xlabel("% of postings")
ax.set_title("Sentiment label share by class")
ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
save_fig(fig, "05_sentiment_by_class.png")

summary.append("\nSentiment by class:")
summary.append(sent_out.to_string())

# ---------------------------------------------------------------
# Section 2.8: Save EDA summary
# ---------------------------------------------------------------
summary_text = "\n".join(summary) + "\n"
(REP / "02_eda_summary.txt").write_text(summary_text)
print("\n" + summary_text)
