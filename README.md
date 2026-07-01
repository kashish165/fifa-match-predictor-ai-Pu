# ⚽ FIFA International Match Outcome Prediction System

> A production-grade machine learning pipeline that predicts the outcome of any international football match using 154 years of historical data (1872–2026).

---

## 📌 Project Overview

This end-to-end data science project ingests four historical datasets covering 49,000+ international men's football matches, engineers 45 predictive features with strict temporal safety, trains and compares four classifiers with time-series cross-validation, and deploys a four-tab interactive Streamlit dashboard for live predictions and historical exploration.

**Core question:** Given any two national teams and match context (venue type, tournament tier), what are the probabilities of a Home Win, Draw, or Away Win?

---

## 📂 Project Architecture

```
fifa_prediction_system/
│
├── data/
│   ├── raw/                        ← Original CSVs (never modified)
│   │   ├── results.csv             ← 49,477 matches (1872–2026)
│   │   ├── goalscorers.csv         ← 47,690 goal-level records
│   │   ├── shootouts.csv           ← 678 penalty shootout results
│   │   └── former_names.csv        ← 36 team name change records
│   └── processed/                  ← Cleaned outputs (git-ignored)
│       ├── master.csv              ← Merged analytical dataframe
│       └── *.png                   ← EDA chart exports
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← Exploratory data analysis (6 blocks, 15+ charts)
│   ├── 02_feature_engineering.ipynb← Feature walkthrough & correlation analysis
│   ├── 03_model_training.ipynb     ← Training with TimeSeriesSplit CV
│   └── 04_model_evaluation.ipynb   ← Confusion matrix, ROC, calibration, SHAP
│
├── src/
│   ├── __init__.py
│   ├── config.py                   ← All magic numbers and hyperparameters
│   ├── utils.py                    ← Shared pure helper functions
│   ├── data_pipeline.py            ← DataPipeline class (load → clean → merge)
│   ├── feature_engineering.py      ← FeatureEngineer (sklearn TransformerMixin)
│   ├── model.py                    ← ModelTrainer, 4 classifiers, evaluation
│   └── updater.py                  ← ModelUpdater (dynamic post-match updates)
│
├── app/
│   └── streamlit_app.py            ← 4-tab interactive dashboard
│
├── models/
│   └── best_model.pkl              ← Serialised best model (git-ignored)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

| File | Rows | Description |
|---|---|---|
| `results.csv` | 49,477 | Every international match result 1872–2026 |
| `goalscorers.csv` | 47,690 | Individual goal records (scorer, minute, penalty, own goal) |
| `shootouts.csv` | 678 | Penalty shootout outcomes |
| `former_names.csv` | 36 | Historical → current team name mappings |

**Source:** [Kaggle — International Football Results from 1872 to 2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fifa-prediction-system.git
cd fifa-prediction-system
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place raw data

Download the four CSV files and place them in `data/raw/`:

```
data/raw/results.csv
data/raw/goalscorers.csv
data/raw/shootouts.csv
data/raw/former_names.csv
```

### 5. Run notebooks in order

```bash
jupyter notebook
```

Open and run:
1. `notebooks/01_EDA.ipynb` — Exploratory analysis
2. `notebooks/02_feature_engineering.ipynb` — Feature walkthrough
3. `notebooks/03_model_training.ipynb` — Train & save models
4. `notebooks/04_model_evaluation.ipynb` — Full evaluation suite

### 6. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The dashboard will open at `http://localhost:8501`

---

## 🤖 Model Architecture

### Feature Groups (45 total)

| Group | Features | Description |
|---|---|---|
| A — Historical | 16 | Career win %, goals scored/conceded avg, clean sheet %, shootout win rate |
| B — Recent Form | 17 | Rolling last-5 and last-10 form, momentum (win streak), rest days |
| C — Head-to-Head | 5 | Historical fixture-specific win %, draw %, avg goals |
| D — Contextual | 3 | Neutral venue flag, tournament tier, knockout stage flag |
| Differentials | 5 | Home minus away deltas for key metrics |

**Temporal safety:** All features use strict look-back — only data from before the match date is ever used. No data leakage.

### Models Trained

| Model | CV Accuracy | CV F1-Weighted | Notes |
|---|---|---|---|
| Logistic Regression | ~0.51 | ~0.52 | Baseline; multinomial via lbfgs |
| Random Forest | ~0.57 | ~0.54 | 300 trees, tuned with RandomizedSearchCV |
| Gradient Boosting | ~0.52 | ~0.51 | sklearn GradientBoostingClassifier |
| Extra Trees | ~0.56 | ~0.54 | Fastest; often best calibrated |

> **Note:** Predicting football matches is inherently uncertain. The draw class (~22.7% base rate) is the hardest to predict. Accuracy around 53–58% on a 3-class problem (random baseline = 33%) is consistent with published sports ML benchmarks.

### Training Strategy

- **TimeSeriesSplit** (5 folds) — prevents future data leaking into training
- **Sample weights** — Tier 1 matches (World Cup, continental championships) weighted 3×
- **Scoring metric** — weighted F1 to handle class imbalance

---

## 📱 Streamlit App — Tab Overview

### Tab 1 — ⚽ Match Predictor
- Select any two of 336 national teams
- Toggle neutral venue / choose tournament type
- Output: three probability gauges + confidence badge + last-5-H2H table + form badges

### Tab 2 — 📊 Team Analysis Dashboard
- All-time record (P/W/D/L/GF/GA/GD)
- Rolling annual win % trend chart (1950–2026)
- Goals scored vs conceded per year
- Top 10 opponents + performance by tier

### Tab 3 — 🏆 Tournament Bracket Predictor
- Select 8 teams for a knockout bracket
- Monte Carlo simulation (up to 10,000 iterations)
- Champion probability chart + QF/SF/Final progression rates
- Group stage simulation mode

### Tab 4 — 🗂️ Historical Explorer
- Most Successful Nations (filterable by decade + tier)
- Goal Trends Over Time (rolling average with annotations)
- Home Advantage Evolution (by decade with trend line)
- Top Individual Scorers (from goalscorers.csv)
- Penalty Shootout Specialists

---

## 🔍 Key Analytical Findings

From `notebooks/01_EDA.ipynb`:

- **Home advantage is real but declining:** Home win rate is ~49% at home grounds vs ~38% at neutral venues — a 11 percentage point edge. The decade-by-decade trend shows this gap has been narrowing since the 1980s (slope ≈ −0.3%/decade).

- **Elite matches produce fewer goals than friendlies:** Tier 1 (World Cup, continental championships) averages ~2.3 goals/match vs Tier 3 friendlies at ~2.9 goals/match, reflecting defensive organisation at high stakes.

- **Draws are more common in top-tier matches:** Elite competitions see ~25% draws vs ~22% in friendlies — pressure and defensive tactics dominate.

- **Goal scoring has declined since the 1950s peak:** Average goals peaked above 4.0/match in the 1950s and have stabilised around 2.5–2.8 since the 1980s, consistent with professionalisation and tactical evolution.

- **First-shooter advantage in shootouts:** Among the 256 shootouts with `first_shooter` data recorded, the team shooting first wins approximately 60% of the time — consistent with the psychological literature on sequential competition.

---

## 🔮 Dynamic Update System

After any new match is played, the `ModelUpdater` class:

1. Appends the result to the master dataset
2. Recomputes rolling/H2H features for both teams
3. Retrains the full pipeline
4. Serialises the updated model to `models/best_model.pkl`
5. Logs timestamp, match, and rolling accuracy on last 100 predictions

```python
from src.updater import ModelUpdater

updater = ModelUpdater(master_df, shootouts_df, model_bundle)
updater.update_after_match(
    date='2026-07-14',
    home_team='France',
    away_team='Brazil',
    home_score=2,
    away_score=1,
    tournament='FIFA World Cup',
)
```

---

## 📈 Future Work

| Enhancement | Description |
|---|---|
| **FIFA Rankings** | Integrate official FIFA ranking points as a feature — strong signal for strength of schedule |
| **Player-level features** | Squad average age, key player availability, injury reports |
| **Weather data** | Temperature, humidity, altitude — especially relevant for South American vs European teams |
| **Travel fatigue** | Time-zone distance and travel hours before match as rest-day refinement |
| **Odds calibration** | Calibrate predicted probabilities against historical betting market odds (Brier score optimisation) |
| **Poisson goal model** | Replace 3-class classifier with bivariate Poisson regression for exact score prediction |
| **Transformer model** | Sequence model over match history (LSTM/Transformer) to capture temporal momentum |
| **Real-time API** | Connect to a live scores API (e.g. Football-Data.org) for automatic daily updates |

---

## 🗂️ Git Commit Checklist

After running each phase, commit with:

```bash
# Phase 1
git add src/data_pipeline.py src/config.py src/utils.py
git commit -m "feat: Phase 1 — data pipeline with entity resolution and master merge"

# Phase 2
git add notebooks/01_EDA.ipynb data/processed/*.png
git commit -m "feat: Phase 2 — EDA with 6 analysis blocks and 15+ charts"

# Phase 3
git add src/feature_engineering.py notebooks/02_feature_engineering.ipynb
git commit -m "feat: Phase 3 — 45-feature engineering with temporal safety guarantees"

# Phase 4
git add src/model.py notebooks/03_model_training.ipynb
git commit -m "feat: Phase 4 — 4 models trained with TimeSeriesSplit CV + hyperparameter tuning"

# Phase 5
git add notebooks/04_model_evaluation.ipynb
git commit -m "feat: Phase 5 — model evaluation with calibration, ROC, SHAP, and feature importances"

# Phase 6
git add src/updater.py
git commit -m "feat: Phase 6 — dynamic post-match model updater with JSONL logging"

# Phase 7
git add app/streamlit_app.py
git commit -m "feat: Phase 7 — 4-tab Streamlit dashboard with Monte Carlo bracket simulator"
```

---

## 📸 App Screenshots

| Tab | Description |
|---|---|
| ![Match Predictor](data/processed/block_b_results.png) | **Tab 1** — Select teams, get win/draw/loss probabilities |
| ![Team Analysis](data/processed/block_d_teams.png) | **Tab 2** — Historical team performance dashboard |
| ![Tournament](data/processed/block_f_shootouts.png) | **Tab 3** — Monte Carlo bracket simulator |
| ![Explorer](data/processed/block_e_trends.png) | **Tab 4** — Historical statistics explorer |

---

## 📄 Licence

MIT — free to use, modify, and distribute with attribution.

---

*Built as a portfolio data science project demonstrating end-to-end ML engineering: data ingestion, feature engineering, model training, evaluation, and interactive deployment.*
