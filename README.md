# TrustLens — Decision Intelligence Platform

> Explainable AI for HR attrition prediction with per-prediction Trust Scoring and EU AI Act compliance.

---

## The problem it solves

Organizations have ML models that predict outcomes, but business leaders don't trust the outputs.
TrustLens sits on top of any HR attrition model and produces a single **Decision Trust Score (0–100)**
that tells HR exactly how much to trust each prediction — and why.

---

## Architecture

```
data/raw/hr_attrition.csv
        │
        ▼
modules/data_quality.py   ← ETL + quality scoring
        │
        ▼
modules/model.py          ← XGBoost training + inference
        │
        ▼
modules/explainer.py      ← SHAP feature attribution
        │
        ▼
modules/trust_score.py    ← Decision Trust Score (0–100)
        │
        ▼
modules/narrative.py      ← Rule-based plain-English report
        │
        ▼
app.py                    ← Streamlit dashboard
```

---

## Tech stack

| Layer         | Technology                        |
|---------------|-----------------------------------|
| Language      | Python 3.11                       |
| ML Model      | XGBoost                           |
| Explainability| SHAP                              |
| Report Engine | Rule-based (deterministic)        |
| Dashboard     | Streamlit                         |
| Dataset       | IBM HR Employee Attrition (1,470) |
| Compliance    | EU AI Act Art. 6(2) — HIGH RISK  |

---

## Trust Score formula

```
Trust Score = (Data Quality × 0.30) + (Model Confidence × 0.40) + (SHAP Clarity × 0.30)
```

| Score  | Interpretation         | Action                        |
|--------|------------------------|-------------------------------|
| 80–100 | RELIABLE               | Safe for decision support     |
| 60–79  | NEEDS REVIEW           | Human review recommended      |
| 0–59   | VERIFY FIRST           | Cross-check with manager      |

---

## Setup

```bash
git clone https://github.com/swarnadeepchatterjee02-glitch/TrustLens
cd TrustLens
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Project structure

```
TrustLens/
├── data/
│   ├── raw/          hr_attrition.csv (1,470 rows × 35 columns)
│   └── processed/    hr_processed.csv (feature-engineered)
├── models/           xgb_attrition.joblib, scaler.joblib
├── modules/
│   ├── data_quality.py
│   ├── model.py
│   ├── explainer.py
│   ├── trust_score.py
│   └── narrative.py
├── config.py
├── app.py
├── requirements.txt
└── README.md
```

---

## EU AI Act compliance

Employment decisions are classified **HIGH-RISK** under EU AI Act Article 6(2) / Annex III.
TrustLens enforces a mandatory human review gate before any employment action can be taken.

---

*Built by Swarnadeep Chatterjee — MBA Candidate, University of Louisville*
