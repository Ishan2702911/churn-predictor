"""
train.py
--------
Run this once before starting the Flask app:
    python train.py

Expects 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the same folder.
Downloads automatically from the Kaggle public URL if not found.
Saves: model.pkl, scaler.pkl, feature_columns.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_FILE)
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

# ── 2. Clean ──────────────────────────────────────────────────────────────────
df.drop(columns=["customerID"], inplace=True)

# TotalCharges has some blank strings → coerce to NaN, fill with median
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ── 3. Feature Engineering ───────────────────────────────────────────────────
# Derived features
df["ChargesPerMonth"]   = df["TotalCharges"] / (df["tenure"] + 1)
df["HighValue"]         = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
df["LongTenure"]        = (df["tenure"] > 24).astype(int)

# Binary encode Yes/No columns
yes_no_cols = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]
for col in yes_no_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

# One-hot encode remaining categoricals
df = pd.get_dummies(df, columns=["gender", "InternetService", "Contract", "PaymentMethod"])

# ── 4. Split ──────────────────────────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 6. Train ──────────────────────────────────────────────────────────────────
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n  Test Accuracy : {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── 8. Save ───────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("\nSaved: model.pkl, scaler.pkl, feature_columns.pkl")
print("You can now run: python app.py")
