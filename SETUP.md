# ⚽ Setup Guide

Follow these steps to run the FIFA Match Outcome Prediction System after cloning this repository.

## 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/fifa-match-outcome-prediction-system.git
cd fifa-match-outcome-prediction-system
```

## 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows (PowerShell)
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Add the raw data (required — not included in the repo)

The four raw CSV files are **not tracked in git** (see `.gitignore`) because of their size.
Download them and place them here:

```
data/raw/results.csv
data/raw/goalscorers.csv
data/raw/shootouts.csv
data/raw/former_names.csv
```

Source: [Kaggle — International Football Results from 1872 to 2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

> If `data/raw/` doesn't exist yet, create it manually.

## 5. Run the app

```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`.
If it doesn't, open that URL manually in your browser.

## 6. (Optional) Run the notebooks

```bash
jupyter notebook
```

Run in order: `01_EDA.ipynb` → `02_feature_engineering.ipynb` → `03_model_training.ipynb` → `04_model_evaluation.ipynb`

---

### Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails with dependency conflicts | Make sure you're using the updated `requirements.txt` (no strict `==` pins) |
| `FileNotFoundError: ... data/raw/results.csv` | You skipped Step 4 — add the CSV files to `data/raw/` |
| `streamlit run app.py` says file not found | Make sure your terminal is inside the folder that actually contains `app.py` — run `dir` (Windows) or `ls` (Mac/Linux) to check |
