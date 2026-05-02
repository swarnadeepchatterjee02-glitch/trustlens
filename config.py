"""
TrustLens — Central Configuration
All paths, constants, and thresholds live here.
Import this module instead of hardcoding values anywhere else.
"""

import os

# ── Paths ────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw",       "hr_attrition.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "hr_processed.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "models",            "xgb_attrition.joblib")
SCALER_PATH    = os.path.join(BASE_DIR, "models",            "scaler.joblib")

# ── Dataset ──────────────────────────────────────────────
TARGET_COLUMN  = "Attrition"           # Yes / No
POSITIVE_LABEL = "Yes"                 # the class we predict

# Features dropped before modelling (constants or leakage-prone)
DROP_COLUMNS = [
    "EmployeeCount",   # always 1
    "EmployeeNumber",  # ID — no predictive value
    "Over18",          # always 'Y'
    "StandardHours",   # always 80
]

# Numeric columns used as model features
NUMERIC_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# Categorical columns — will be one-hot encoded
CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]

# Human-readable display names for SHAP chart labels
FEATURE_DISPLAY_NAMES = {
    "OverTime_Yes":                      "Overtime",
    "JobSatisfaction":                   "Job Satisfaction",
    "WorkLifeBalance":                   "Work–Life Balance",
    "MonthlyIncome":                     "Monthly Income",
    "YearsAtCompany":                    "Tenure",
    "NumCompaniesWorked":                "Companies Worked",
    "DistanceFromHome":                  "Distance from Home",
    "Age":                               "Age",
    "TotalWorkingYears":                 "Total Working Years",
    "EnvironmentSatisfaction":           "Environment Satisfaction",
    "JobInvolvement":                    "Job Involvement",
    "MaritalStatus_Single":              "Marital Status (Single)",
    "BusinessTravel_Travel_Frequently":  "Frequent Business Travel",
    "StockOptionLevel":                  "Stock Options",
    "YearsWithCurrManager":              "Years with Manager",
    "JobLevel":                          "Job Level",
    "YearsInCurrentRole":                "Years in Role",
    "RelationshipSatisfaction":          "Relationship Satisfaction",
}

# ── Trust Score weights ───────────────────────────────────
TRUST_WEIGHT_DATA_QUALITY  = 0.30
TRUST_WEIGHT_CONFIDENCE    = 0.40
TRUST_WEIGHT_SHAP_CLARITY  = 0.30

# Thresholds for Trust Score interpretation
TRUST_RELIABLE   = 80   # >= 80 → RELIABLE
TRUST_REVIEW     = 60   # 60–79 → NEEDS REVIEW
                        # < 60  → VERIFY FIRST

# ── Model ────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5,     # handles class imbalance (16% positive)
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
}

TEST_SIZE      = 0.20
RANDOM_STATE   = 42

# ── Data Quality thresholds ──────────────────────────────
OUTLIER_IQR_MULTIPLIER = 3.0   # values beyond 3×IQR flagged
MAX_MISSING_RATE       = 0.05  # >5% missing in a column = warning

# ── EU AI Act ────────────────────────────────────────────
EU_AI_ACT_RISK_LEVEL = "HIGH"       # employment decisions → always HIGH
EU_AI_ACT_ARTICLE    = "Art. 6(2)"  # Annex III, point 4 — employment/workers management
EU_AI_ACT_NOTE       = (
    "This system makes predictions about employment-related decisions. "
    "Under EU AI Act Article 6(2) and Annex III, it is classified as HIGH-RISK. "
    "A qualified human must review and sign off on any action taken."
)
