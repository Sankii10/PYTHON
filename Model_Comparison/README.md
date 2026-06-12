# EPL Match Outcome Prediction — Multi-Model Classifier Comparison

## Overview

This project builds and benchmarks three machine learning classification models to predict the outcome of English Premier League (EPL) matches — **Home Win**, **Draw**, or **Away Win** — using in-game performance statistics from the 2021–2024 seasons.

The core objective is not just to predict outcomes, but to **identify which algorithm generalises best** on multi-class sports analytics data, and to understand which in-game features drive predictive power.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle – EPL Stats 2021–2024](https://www.kaggle.com/datasets/mohamadsallah5/english-premier-league-stats20212024) |
| Records | 1,140 matches |
| Features | 30 in-game stats (possession, shots, passes, corners, fouls, cards, etc.) |
| Target | Match result: `h` = Home Win, `d` = Draw, `a` = Away Win |
| Missing Values | `year` and `month` columns — 560 nulls (imputed with mode) |
| Duplicates | 0 |

---

## Problem Statement

Predicting football match outcomes is a **3-class imbalanced classification problem**. Draws are underrepresented relative to wins, making standard accuracy misleading. This project addresses that with SMOTE oversampling and evaluates models using both accuracy and F1-Score.

---

## Project Workflow

```
Data Loading → EDA & Cleaning → Feature Engineering
→ Train-Test Split (80/20, stratified)
→ SMOTE (handle class imbalance on training set)
→ StandardScaler (feature normalisation)
→ Model Training (3 algorithms)
→ Evaluation (Accuracy, F1, Confusion Matrix)
→ 5-Fold Stratified Cross-Validation
→ Feature Importance Analysis
```

---

## Models Compared

| Model | Configuration |
|---|---|
| Logistic Regression | Multinomial, L2 penalty, C=1.0, max_iter=1000 |
| Random Forest | 500 trees, max_features='sqrt', balanced_subsample |
| XGBoost | 500 estimators, lr=0.05, max_depth=6, subsample=0.8 |

---

## Results

### Test Set Performance

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Logistic Regression** | **81.1%** | **0.79** | **0.82** |
| XGBoost | 68.0% | 0.59 | 0.66 |
| Random Forest | 64.0% | 0.56 | 0.61 |

### 5-Fold Stratified Cross-Validation

| Model | CV Accuracy |
|---|---|
| **Logistic Regression** | **81.9%** |
| XGBoost | 68.9% |
| Random Forest | 62.2% |

### Per-Class Performance — Logistic Regression (Best Model)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Home Win (0) | 0.90 | 0.83 | 0.86 | 99 |
| Draw (1) | 0.61 | 0.71 | 0.65 | 51 |
| Away Win (2) | 0.86 | 0.86 | 0.86 | 78 |

---

## Key Finding

**Logistic Regression outperformed both ensemble models by a significant margin (+13–17% accuracy).**

This is a counter-intuitive result worth noting. The likely explanation: after SMOTE and StandardScaler, the decision boundary between classes is approximately linear in feature space. XGBoost and Random Forest, being high-variance models, overfit on the SMOTE-synthetic samples during training, which hurts generalisation on real test data.

The **Draw class** remains the hardest to predict across all models (F1: 0.65 vs 0.86 for wins) — consistent with the broader sports analytics literature, where draws are structurally ambiguous events.

---

## Feature Importance

Feature importance was extracted from both Random Forest and XGBoost. Key predictive features:

- `home_on` / `away_on` — shots on target (strongest signal)
- `home_chances` / `away_chances` — clear goal-scoring opportunities
- `home_possessions` / `away_possessions` — territorial dominance
- `home_saves` / `away_saves` — goalkeeping intervention (correlates with pressure absorbed)

---

## Tech Stack

```
Python 3.13
pandas, numpy
scikit-learn (LogisticRegression, RandomForestClassifier, SMOTE, StratifiedKFold)
xgboost
matplotlib, seaborn
imbalanced-learn
```

---
## Author

**Sanket Gaikwad**  
[LinkedIn](http://www.linkedin.com/in/sanket-gaikwad-10) | [GitHub](https://github.com/Sankii10)
