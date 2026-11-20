"""
Section 3: Word frequency and sentiment figures
Figures 3, 4, 6, 7 and 8 of the dissertation.

Figure 3: top 5 words in non-fraudulent postings (total occurrences)
Figure 4: top 5 words in fraudulent postings (total occurrences)
Figure 6: average sentiment (VADER compound score) per class
Figure 7: sentiment label share for genuine postings
Figure 8: sentiment label share for fraudulent postings

Input : data/cleaned_fake_job_postings.csv, data/fake_job_postings.csv (chart labels)
Output: visuals/fig03_*.png, fig04_*.png, fig06_*.png, fig07_*.png, fig08_*.png
        reports/03_*.csv
"""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from utils import CLASS_NAMES, REP, VIS, load_clean, stem_display_lookup

CLASS_COLORS = {"Genuine": "#4C72B0", "Fraudulent": "#C44E52"}
SENTIMENT_COLORS = {"negative": "#8C8C8C", "neutral": "#D9D9D9", "positive": "#55A868"}

df = load_clean()
display = stem_display_lookup()


def label(stem):
    return display.get(stem, stem)


def save(fig, name):
    fig.savefig(VIS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visuals/{name}")


# ---------------------------------------------------------------
# Section 3.1: Top 5 words by frequency (Figures 3 and 4)
# Frequency = total occurrences of each stemmed token in the class.
# ---------------------------------------------------------------
def top_words(texts, n=5):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    return counter.most_common(n)


word_rows = []
for cls, fig_no, caption in [(0, 3, "non-fraudulent"), (1, 4, "fraudulent")]:
    name = CLASS_NAMES[cls]
    top = top_words(df.loc[df["fraudulent"] == cls, "text_clean"])
    words = [label(w) for w, _ in top]
    counts = [c for _, c in top]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(words, counts, color=CLASS_COLORS[name])
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}", ha="center", va="bottom")
    ax.set_title(f"Top 5 words in {caption} postings")
    ax.set_ylabel("Frequency (total occurrences)")
    save(fig, f"fig{fig_no:02d}_top5_words_{name.lower()}.png")

    word_rows += [{"class": name, "rank": i, "word": label(w), "stem": w, "frequency": c}
                  for i, (w, c) in enumerate(top, start=1)]

pd.DataFrame(word_rows).to_csv(REP / "03_top5_words_by_class.csv", index=False)

# ---------------------------------------------------------------
# Section 3.2: Average sentiment per class (Figure 6)
# Scores come from VADER, computed in 01_clean_data.py.
# ---------------------------------------------------------------
mean_sent = df.groupby("fraudulent")["sentiment_compound"].mean()
names = [CLASS_NAMES[i] for i in mean_sent.index]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(names, mean_sent.values, color=[CLASS_COLORS[n] for n in names])
for bar, v in zip(bars, mean_sent.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
ax.set_ylim(-1, 1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean VADER compound score (-1 to +1)")
ax.set_title("Average sentiment of fraudulent and non-fraudulent postings")
save(fig, "fig06_average_sentiment.png")

# ---------------------------------------------------------------
# Section 3.3: Sentiment share per class (Figures 7 and 8)
# ---------------------------------------------------------------
share_rows = []
for cls, fig_no in [(0, 7), (1, 8)]:
    name = CLASS_NAMES[cls]
    shares = (df.loc[df["fraudulent"] == cls, "sentiment_label"]
              .value_counts(normalize=True)
              .reindex(["negative", "neutral", "positive"])
              .fillna(0) * 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, _ = ax.pie(
        shares.values,
        colors=[SENTIMENT_COLORS[k] for k in shares.index],
        autopct=lambda p: f"{p:.2f}%" if p >= 1 else "",
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    ax.legend(wedges, [f"{k} ({v:.2f}%)" for k, v in shares.items()],
              loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title(f"Sentiment percentage for {name.lower()} jobs")
    save(fig, f"fig{fig_no:02d}_sentiment_{name.lower()}.png")

    share_rows += [{"class": name, "sentiment": k, "percent": round(v, 2)}
                   for k, v in shares.items()]

pd.DataFrame(share_rows).to_csv(REP / "03_sentiment_shares.csv", index=False)
mean_sent.rename(index=CLASS_NAMES).round(4).to_csv(
    REP / "03_mean_sentiment.csv", header=["mean_compound"])
print("\nSection 3 figures written.")
