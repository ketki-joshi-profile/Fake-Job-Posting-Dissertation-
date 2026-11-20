# Fake Job Posting Detection

I analysed 17,880 online job postings to find which features separate fraudulent adverts from genuine ones, and tested whether text, sentiment and employer metadata can flag scams before applicants are harmed.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Project Scope & Tools](#3-project-scope--tools)
4. [Repository Structure](#4-repository-structure)
5. [Data Workflow](#5-data-workflow)
6. [Data Model & Schema](#6-data-model--schema)
7. [Analysis & Metrics](#7-analysis--metrics)
8. [Key Insights](#8-key-insights)
9. [Recommendations](#9-recommendations)
10. [Assumptions & Limitations](#10-assumptions--limitations)
11. [Future Enhancements](#11-future-enhancements)
12. [Deliverables](#12-deliverables)
13. [Author](#13-author)

---

## 1. Project Overview

**Context:** Fake job postings collect applicants' personal data, waste their time, and sometimes lead to financial scams. Most postings are genuine, so fraud is a rare class. This project began as my MSc dissertation in Business Analytics at Queen's University Belfast (2024) and was rebuilt as a reproducible Python pipeline.

**Problem Statement:** Can fraudulent postings be identified from their text and metadata? Does the class imbalance (4.9% fraudulent) make accuracy a misleading measure of performance?

**Approach:** I cleaned the EMSCAD dataset, ran exploratory text and sentiment analysis, and compared Random Forest, Logistic Regression and Naive Bayes before and after random oversampling of the training data.

**Outcome:** Random Forest reaches about 98% accuracy, but 95% of postings are genuine, so a model that never flags fraud would score about 95%. Random Forest catches 63% of fraudulent postings. The strongest single signal is whether a posting has a company logo. Sentiment does not separate the two classes.

---

## 2. Objectives

- **Primary Objective:** Identify which posting features distinguish fraudulent adverts from genuine ones.
- **Secondary Objective 1:** Compare Random Forest, Logistic Regression and Naive Bayes on the same split, before and after oversampling.
- **Secondary Objective 2:** Test whether word patterns and sentiment differ between the two classes.
- **Secondary Objective 3:** Check whether headline accuracy holds up when measured with precision and recall for the fraud class.

---

## 3. Project Scope & Tools

### Scope

| Dimension | Details |
|---|---|
| **In Scope** | EMSCAD job postings: four text fields (company profile, description, requirements, benefits) and metadata (logo, questions, employment type, required experience, industry, function, location) |
| **Out of Scope** | Salary, department, telecommuting and required education, excluded as in the dissertation (Section 3.1). Job titles and the categorical fields are not used in the models. External data, SQL, dashboards and deployment. |
| **Time Period** | Dataset collection dates are not documented in the file. Analysis completed November 2026. |
| **Granularity** | One row per job posting. 17,525 postings after de-duplication. |

### Tools & Technologies

| Category | Tool(s) Used |
|---|---|
| **Data Storage** | CSV files (local) |
| **Data Processing** | Python 3, pandas, NumPy, SciPy, NLTK (stopwords, Porter stemmer) |
| **Analysis** | scikit-learn (TF-IDF, Logistic Regression, Lasso, Random Forest, Naive Bayes), VADER sentiment (`vaderSentiment`) |
| **Visualization** | Matplotlib, WordCloud |
| **Version Control** | Git / GitHub |
| **Documentation** | Markdown |
| **Other** | None |

---

## 4. Repository Structure

```
fake-job-detection/
│
├── data/
│   ├── fake_job_postings.csv           # Raw EMSCAD dataset, unchanged
│   └── cleaned_fake_job_postings.csv   # Output of scripts/01_clean_data.py
│
├── scripts/
│   ├── utils.py                        # Shared split, oversampling and feature code
│   ├── 01_clean_data.py                # Section 1: cleaning and preprocessing
│   ├── 02_eda.py                       # Section 2: exploratory analysis
│   ├── 03_text_figures.py              # Section 3: Figures 3, 4, 6, 7, 8
│   └── 04_predictive.py                # Section 4: Figures 5, 9, 10, 11, 12
│
├── reports/                            # CSV tables and text summaries written by the scripts
├── visuals/                            # Exported charts (PNG)
├── .gitignore
└── README.md                           # You are here
```

The template's `data/raw`, `data/processed` and `data/external` split is not used. The raw file and the cleaned file sit side by side in `data/`. The unused `queries/`, `notebooks/` and `docs/` folders were removed.

---

## 5. Data Workflow

```
[EMSCAD CSV, Kaggle]
        ↓
[Load and de-duplicate]
        ↓
[Clean and transform text and metadata]
        ↓
[EDA, text and sentiment analysis, classification]
        ↓
[Charts, tables and summaries]
```

**Source:** EMSCAD (Employment Scam Aegean Dataset) from Kaggle. 17,880 postings, 18 columns, 866 fraudulent (4.8%). The CSV is about 50 MB.

**Ingestion:** Loaded with pandas from `data/fake_job_postings.csv`.

**Cleaning:**
- Removed 355 duplicate adverts, matched on title, location and the four text fields. `job_id` is unique by construction, so it can't identify reposts.
- Dropped `department`, `salary_range`, `telecommuting` and `required_education`.
- Replaced remaining nulls with blank strings.
- Decoded HTML entities. `&amp;` alone appeared about 16,600 times, and left undecoded it became a top keyword.

**Transformation:**
- Split `location` into `country`, taking the first comma-separated part.
- Extracted URLs and hashtags from the four text fields into new columns, with values and counts for each.
- Combined the four text fields and scored each posting with VADER (compound score and label).
- Lowercased, kept letters only, removed English stopwords, and Porter-stemmed the tokens.
- Final dataset: 17,525 postings, 853 fraudulent (4.87%).

**Analysis:**
- Descriptive counts, correlations, word frequencies, TF-IDF terms and sentiment by class.
- Classification on a stratified 80/20 split (seed 42). TF-IDF is fit on the training split only.
- Random oversampling is applied to the training split only, so the test set keeps the real class mix.

**Output:** Charts in `visuals/`, CSV tables and an EDA summary in `reports/`.

**Run order:**

```bash
pip install pandas numpy scipy scikit-learn nltk vaderSentiment wordcloud matplotlib
python scripts/01_clean_data.py
python scripts/02_eda.py
python scripts/03_text_figures.py
python scripts/04_predictive.py
```

---

## 6. Data Model & Schema

**Dataset / Table:** `data/cleaned_fake_job_postings.csv`

| Field Name | Data Type | Description | Example Value |
|---|---|---|---|
| `job_id` | int | Unique posting identifier from the source data | `1` |
| `title` | string | Job title as posted | `Marketing Intern` |
| `country` | string | Country code, first part of the location field | `US` |
| `has_company_logo` | int | 1 if the posting has a company logo | `1` |
| `has_questions` | int | 1 if the posting includes screening questions | `0` |
| `employment_type` | string | Full-time, part-time, contract, etc. Blank if missing | `Full-time` |
| `required_experience` | string | Required experience level. Blank if missing | `Internship` |
| `industry` | string | Industry label. Blank if missing | `Marketing and Advertising` |
| `function` | string | Job function label. Blank if missing | `Marketing` |
| `fraudulent` | int | Target: 1 = fraudulent, 0 = genuine | `0` |
| `{field}_urls` | string | Pipe-separated URLs found in the field, for each of the four text fields | blank or `https://…` |
| `{field}_hashtags` | string | Pipe-separated hashtags found in the field, for each of the four text fields | blank or `#…` |
| `{field}_url_count` | int | Number of URLs in the field | `0` |
| `{field}_hashtag_count` | int | Number of hashtags in the field | `0` |
| `sentiment_compound` | float | VADER compound score for the combined text, from -1 to +1 | `0.9612` |
| `sentiment_label` | string | `positive`, `neutral` or `negative` (VADER thresholds ±0.05) | `positive` |
| `text_clean` | string | Stemmed, stopword-free tokens from the combined text | `work experi team manag …` |

Here `{field}` is one of `company_profile`, `description`, `requirements` or `benefits`, which gives 16 columns of this type.

**Row count:** 17,525 rows, 29 columns
**Date range:** Not documented in the source file
**Key join / relationship:** None. Single table. `job_id` is unique.

---

## 7. Analysis & Metrics

### Analytical Approach

I started with exploratory analysis to see which words, metadata and sentiment patterns differ by class. Then I moved to classification, comparing three models on the same split. I report results for the models before and after oversampling, using precision and recall for the fraud class alongside accuracy. Accuracy is reported but not treated as the main measure. The test set is never oversampled, so every score reflects the real class mix.

### Key Metrics Defined

| Metric | Plain-Language Definition | Why It Matters |
|---|---|---|
| **Accuracy** | Share of all postings classified correctly | Misleading here: the majority-class baseline is 95.1% |
| **Precision (fraud)** | Of postings flagged as fraudulent, the share that really are | Shows how often a fraud flag is wrong and an honest posting is penalised |
| **Recall (fraud)** | Of fraudulent postings, the share the model catches | Shows how many scams slip through |
| **F1 (fraud)** | Harmonic mean of precision and recall | Single number for the fraud class when both errors matter |
| **Mean VADER compound** | Average sentiment of a posting, from -1 (negative) to +1 (positive) | Tests whether tone separates the classes |

### Methods Used

- Descriptive statistics: class counts, null rates and duplicate detection
- Correlation analysis between binary posting features and the fraud label
- Normalised term frequency and TF-IDF weighting, compared across classes
- Sentiment analysis with VADER, by class
- Logistic Regression on the top 10 keywords by coefficient size, before and after balancing
- Lasso feature selection, with the largest penalty that still keeps 10 features
- Random Forest, Logistic Regression and Naive Bayes on a stratified 80/20 split, before and after random oversampling

### Results (test set: 3,505 postings, 171 fraudulent)

| Model | Stage | Accuracy | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|---|---|
| Random Forest | Before balancing | 98.17% | 99.08% | 63.16% | 77.14% |
| Random Forest | After balancing | 98.00% | 96.33% | 61.40% | 75.00% |
| Logistic Regression | Before balancing | 97.15% | 100.00% | 41.52% | 58.68% |
| Logistic Regression | After balancing | 96.78% | 61.69% | 89.47% | 73.03% |
| Naive Bayes | Before balancing | 95.46% | 92.86% | 7.60% | 14.05% |
| Naive Bayes | After balancing | 93.89% | 43.91% | 90.64% | 59.16% |

---

## 8. Key Insights

**Insight 1: Company logo presence is the strongest single signal.**
Postings with a logo have a correlation of -0.26 with fraud, the largest of any binary feature in the analysis. The Lasso step also gives `has_company_logo` the largest negative coefficient. This suggests that checking whether an employer has a verifiable identity is a reasonable first screen.

**Insight 2: High accuracy hides low recall.**
Random Forest looks strong at 98.2% accuracy and 99% precision, but before balancing it catches only 108 of the 171 fraudulent postings in the test set. Logistic Regression on the top 10 keywords, before balancing, catches just 6% of fraud. After balancing, Logistic Regression recovers 89% of fraud, but precision falls to 62%, and Naive Bayes reaches 91% recall with precision of 44%. The right model depends on how costly a missed scam is compared with a wrongly flagged genuine posting, so the decision threshold needs to be set deliberately.

**Insight 3: Frequent words are shared between classes, and only a few terms separate them.**
The most frequent words are almost the same in both classes, with work, experience, team and management in the top five of each. Ranking terms by the gap in TF-IDF weight shows a clearer picture. Fraud-leaning terms include *entry*, *earn*, *per*, *home*, *internet* and *position*, which fits common earnings and work-from-home promises. Genuine-leaning terms include *team*, *market*, *web*, *digital*, *client* and *design*, which describe established business work. Some terms in the fraud list, such as *oil* and *type*, are probably noise.

**Insight 4: Sentiment does not separate the classes.**
The mean VADER score is 0.95 for genuine postings and 0.91 for fraudulent ones. Almost all postings in both classes are positive: 99.1% of genuine and 97.7% of fraudulent. Tone on its own is not a useful signal, although the dissertation's sentiment figures differ from these (see Limitations).

---

## 9. Recommendations

| Priority | Recommendation | Based On | Suggested Owner |
|---|---|---|---|
| **High** | Report precision and recall for the fraud class alongside accuracy in every model comparison. Treat 98% accuracy as a baseline, not a result. | Insight 2 | Analytics / data science team |
| **High** | Before any deployment, tune the decision threshold to a chosen balance between wrongly flagged genuine postings and missed scams. | Insight 2 | Trust & safety team |
| **Medium** | Add a company identity check, such as logo or verifiable employer details, as a screening input. | Insight 1 | Trust & safety team |
| **Medium** | Verify the split used for the dissertation's keyword and balancing figures, then rerun with a clean pipeline (see Limitations). | Limitations | Project author |
| **Medium** | Remove near-duplicate postings, not just exact copies, and raise the minimum document frequency so employer-specific tokens drop out. | Limitations | Data engineering |
| **Low** | Test a sentence-embedding or transformer model on the same split as a stronger text baseline. | Insight 3 | Data science team |
| **Low** | Use group-based cross-validation by employer to measure how well the model generalises to employers it hasn't seen. | Limitations | Data science team |

---

## 10. Assumptions & Limitations

### Assumptions

- The EMSCAD labels are correct. I did not verify them against any independent source.
- Exact content duplicates are reposts of the same advert, so keeping the first copy is acceptable.
- Blank strings are an acceptable replacement for missing text, meaning "no content in this field".
- VADER is suitable for this text, although it was built for social media.
- The attributes dropped in the dissertation (department, salary, telecommuting and required education) carry no signal the analysis needs.

### Limitations

- **The dissertation's reported results do not match this reproduction, and the dissertation disagrees with itself.** Section 5.1 of the text gives accuracy of 90.58% for Logistic Regression after balancing and 50.04% for Naive Bayes. Table 4 and Figure 12 give 63.8% and 76.7% for the same models. Random Forest is 98.71% in the text and 98.3% in Figure 12. Figures 11 and 12 and Table 4 are the most consistent sources, so this README uses the outputs of `scripts/04_predictive.py` instead.
- **The test sets behind some dissertation figures appear to differ.** Figure 5 has 1,047 test rows with 31 fraudulent. Figure 9 has 5,232 test rows with 163 fraudulent. My split has 3,505 rows with 171 fraudulent. If oversampling was applied before the split, duplicated fraud postings could appear in both training and test data and inflate the scores. My pipeline splits first and oversamples only the training set. This needs to be checked against the original code.
- **Near-duplicates inflate some results.** Exact de-duplication doesn't catch similar adverts from the same employer. The keyword models rank employer-specific tokens, such as *accion*, *aker* and *aecom*, among their top features. These probably reflect a few employers rather than general patterns in fraud.
- **Feature sets differ from the dissertation.** The dissertation's Lasso output lists job-title indicators, which this reproduction does not use. The models here use TF-IDF text and numeric flags only. The categorical fields (employment type, required experience, industry and function) are not included, so the feature rankings aren't directly comparable.
- **Word counts don't match.** The dissertation's Table 2 shows "work" in 11,834 genuine postings. My reproduction finds it in about 13,600 documents, so the pipelines differ in ways I couldn't identify.
- **Sentiment scale differs.** The dissertation reports average sentiment of 0.19 to 0.21. VADER gives 0.91 to 0.95 on the same data, so the dissertation must have used a different scoring tool or scale.
- **Modelling is basic.** Random Forest uses 100 trees with default settings. There is one split and no cross-validation, and no hyperparameters were tuned.
- **Data provenance is undocumented.** The collection period, the employers covered and the labelling method are not described in the files I used.
- **Random oversampling duplicates rows.** Methods such as SMOTE or class weights could behave differently.

**What a more rigorous version would include:** a split that keeps each employer's postings in one fold, repeated cross-validation, near-duplicate removal, precision-recall curves with threshold selection, and documented data provenance.

---

## 11. Future Enhancements

- Remove near-duplicate postings using text similarity, then rerun all models. This addresses the employer-name keywords in Insight 3 and the Limitations section.
- Use group-based cross-validation by employer to measure generalisation to new employers, which the single split can't show.
- Add precision-recall curves and threshold tuning for each model, to address the recall gap in Insight 2.
- Add job titles and the categorical fields to the models, so the Lasso results can be compared with the dissertation's.
- Compare sentence embeddings with the TF-IDF features on the same split.

---

## 12. Deliverables

| Deliverable | Description | Location |
|---|---|---|
| Cleaned dataset | 17,525 postings, 29 fields, ready for analysis | `data/cleaned_fake_job_postings.csv` |
| Cleaning script | Section 1: de-duplication, cleaning, URL and hashtag extraction, sentiment, normalisation | `scripts/01_clean_data.py` |
| EDA script | Section 2: class distribution, correlations, word clouds, TF-IDF, sentiment | `scripts/02_eda.py` |
| Text and sentiment script | Figures 3, 4, 6, 7, 8 | `scripts/03_text_figures.py` |
| Predictive script | Figures 5, 9, 10, 11, 12 | `scripts/04_predictive.py` |
| Shared helpers | Split, oversampling and feature construction | `scripts/utils.py` |
| Cleaning log | Row counts and cleaning decisions | `reports/01_cleaning_log.txt` |
| EDA summary | Class counts, correlations, term tables, sentiment | `reports/02_eda_summary.txt` |
| Model scores | Accuracy, precision, recall and F1 before and after balancing | `reports/04_model_scores.csv` |
| Charts | EDA charts (01–05) and dissertation figures (03–12) | `visuals/` |

---

## 13. Author

**Ketki Joshi**
Data Analytics and Operations professional. MSc Business Analytics, Queen's University Belfast (2024).

🔗 LinkedIn: https://www.linkedin.com/in/ketki-joshi-31051999kpj/
💼 Portfolio / GitHub: https://github.com/ketki-joshi-profile
📧 ketkipjoshi31@gmail.com

Last updated: November 2026

