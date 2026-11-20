"""
Section 4: Keyword and predictive analysis
Figures 5, 9, 10, 11 and 12 of the dissertation.

Figure 5 : top 10 keywords by Logistic Regression coefficient size, then a Logistic
           Regression refit using only those 10 words. Unbalanced training set.
Figure 9 : same as Figure 5, on the randomly oversampled training set.
Figure 10: Lasso feature selection, keeping the 10 most significant features.
Figure 11: accuracy of Random Forest, Logistic Regression and Naive Bayes, unbalanced.
Figure 12: same models on the randomly oversampled training set.

Accuracy alone is misleading on imbalanced data (predicting "genuine" for every
posting scores about 95%), so precision, recall and F1 for the fraudulent class are
saved alongside it in reports/04_*.csv.

Input : data/cleaned_fake_job_postings.csv, data/fake_job_postings.csv (chart labels)
Output: visuals/fig05_*.png, fig09_*.png, fig10_*.png, fig11_*.png, fig12_*.png
        reports/04_*.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

from utils import (REP, SEED, VIS, build_matrix, feature_names, load_clean,
                   random_oversample, split_data, stem_display_lookup)

display = stem_display_lookup()


def label(name):
    return display.get(name, name)


def save(fig, name):
    fig.savefig(VIS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visuals/{name}")


def metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_fraudulent": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_fraudulent": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_fraudulent": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ---------------------------------------------------------------
# Section 4.1: Data preparation
# The TF-IDF vocabulary is fit on the training split only, which avoids leakage
# into the test set. Feature selection uses the training data only.
# ---------------------------------------------------------------
df = load_clean()
train_df, test_df = split_data(df)
y_train = train_df["fraudulent"].values
y_test = test_df["fraudulent"].values

vectorizer = TfidfVectorizer(min_df=5)
T_train = vectorizer.fit_transform(train_df["text_clean"])
T_test = vectorizer.transform(test_df["text_clean"])
words = vectorizer.get_feature_names_out()

X_train = build_matrix(T_train, train_df)   # text + numeric features
X_test = build_matrix(T_test, test_df)

T_train_bal, y_train_bal = random_oversample(T_train, y_train)
X_train_bal, _ = random_oversample(X_train, y_train)

print(f"Training: {len(y_train):,} rows, fraud rate {y_train.mean():.2%}")
print(f"After oversampling: {len(y_train_bal):,} rows, fraud rate {y_train_bal.mean():.2%}")


# ---------------------------------------------------------------
# Section 4.2: Top 10 keywords with Logistic Regression (Figures 5 and 9)
# Step 1: fit on all words and rank them by absolute coefficient.
# Step 2: refit on only those 10 words and report that model's output.
# Positive coefficient = leans fraudulent. Negative = leans genuine.
# ---------------------------------------------------------------
def keyword_model(T_tr, y_tr, fig_no, tag, title):
    full = LogisticRegression(max_iter=1000).fit(T_tr, y_tr)
    top_idx = np.argsort(np.abs(full.coef_[0]))[::-1][:10]

    top_model = LogisticRegression(max_iter=1000).fit(T_tr[:, top_idx], y_tr)
    scores = metrics(y_test, top_model.predict(T_test[:, top_idx]))

    coef = pd.Series(top_model.coef_[0], index=[label(w) for w in words[top_idx]])
    coef = coef.sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0" if c < 0 else "#C44E52" for c in coef.values]
    ax.barh(coef.index, coef.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic regression coefficient (positive = fraudulent-leaning)")
    ax.set_title(f"{title}\nTest accuracy using these 10 words: {scores['accuracy']:.2%}")
    save(fig, f"fig{fig_no:02d}_top10_keywords_{tag}.png")

    table = pd.DataFrame({
        "rank": range(1, 11),
        "word": [label(w) for w in words[top_idx]],
        "stem": words[top_idx],
        "coefficient": top_model.coef_[0].round(4),
    })
    return table, scores


table5, scores5 = keyword_model(
    T_train, y_train, 5, "before_balancing",
    "Top 10 keywords by significance (before balancing)")
table5.to_csv(REP / "04_fig05_keywords_before_balancing.csv", index=False)

table9, scores9 = keyword_model(
    T_train_bal, y_train_bal, 9, "after_balancing",
    "Top 10 keywords by significance (after balancing)")
table9.to_csv(REP / "04_fig09_keywords_after_balancing.csv", index=False)

print("Figure 5 keyword model:", scores5)
print("Figure 9 keyword model:", scores9)


# ---------------------------------------------------------------
# Section 4.3: Feature selection with Lasso (Figure 10)
# Lasso is a linear model, so it is fit to the binary target as described in
# Section 3.3. Features are scaled so the penalty treats them equally. The largest
# penalty that still keeps at least 10 non-zero features is used.
# ---------------------------------------------------------------
scaler = StandardScaler(with_mean=False).fit(X_train)
Xs_train = scaler.transform(X_train)
names = feature_names(vectorizer)

lasso = None
alpha_used = None
for alpha in np.logspace(-1, -4, 25):
    candidate = Lasso(alpha=alpha, max_iter=5000).fit(Xs_train, y_train)
    if np.count_nonzero(candidate.coef_) >= 10:
        lasso, alpha_used = candidate, alpha
        break
if lasso is None:  # fallback: smallest penalty in the grid
    alpha_used = 1e-4
    lasso = Lasso(alpha=alpha_used, max_iter=5000).fit(Xs_train, y_train)

top_idx = np.argsort(np.abs(lasso.coef_))[::-1][:10]
lasso_coef = pd.Series(lasso.coef_[top_idx], index=[label(n) for n in names[top_idx]])
lasso_coef = lasso_coef.sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#4C72B0" if c < 0 else "#C44E52" for c in lasso_coef.values]
ax.barh(lasso_coef.index, lasso_coef.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lasso coefficient on standardised features (positive = fraudulent-leaning)")
ax.set_title(f"Feature selection using Lasso (alpha = {alpha_used:.4f})")
save(fig, "fig10_lasso_feature_selection.png")

pd.DataFrame({
    "rank": range(1, 11),
    "feature": [label(n) for n in names[top_idx]],
    "stem_or_name": names[top_idx],
    "coefficient": lasso.coef_[top_idx].round(5),
}).to_csv(REP / "04_fig10_lasso_top10_features.csv", index=False)
print(f"Lasso alpha used: {alpha_used:.4f}, non-zero features: {np.count_nonzero(lasso.coef_)}")


# ---------------------------------------------------------------
# Section 4.4: Model accuracy before and after balancing (Figures 11 and 12)
# Models use the full feature set (text + numeric). Naive Bayes needs non-negative
# inputs, which all features here are.
# ---------------------------------------------------------------
def model_scores(X_tr, y_tr):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        results[name] = metrics(y_test, model.predict(X_test))
        print(f"  {name}: {results[name]}")
    return results


def accuracy_figure(results, fig_no, tag, title):
    names_ = list(results)
    values = [results[n]["accuracy"] * 100 for n in names_]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names_, values, color=["#4C72B0", "#55A868", "#C44E52"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    save(fig, f"fig{fig_no:02d}_accuracy_{tag}.png")


print("Before balancing:")
before = model_scores(X_train, y_train)
accuracy_figure(before, 11, "before_balancing",
                "Accuracy of predictive algorithms (before balancing)")

print("After balancing:")
after = model_scores(X_train_bal, y_train_bal)
accuracy_figure(after, 12, "after_balancing",
                "Accuracy of predictive algorithms (after balancing)")

rows = []
for stage, results in [("before_balancing", before), ("after_balancing", after)]:
    for model_name, scores in results.items():
        rows.append({"stage": stage, "model": model_name, **scores})
pd.DataFrame(rows).to_csv(REP / "04_model_scores.csv", index=False)
print("\nSection 4 figures and tables written.")
