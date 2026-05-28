"""
app.py
------
Flask application for Customer Churn Prediction.

Routes:
  GET  /           → Input form
  POST /predict    → Run prediction, log to DB, show result
  GET  /history    → Audit log of recent predictions

Run:
  python app.py
"""

import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from database import init_db, log_prediction, get_recent_predictions

app = Flask(__name__, static_folder='static')

# ── Load model artifacts ───────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Initialise DB
init_db()


def build_input_vector(form_data: dict) -> np.ndarray:
    """
    Convert raw form data into the exact feature vector the model expects.
    Mirrors the preprocessing in train.py.
    """
    tenure          = int(form_data["tenure"])
    monthly_charges = float(form_data["MonthlyCharges"])
    total_charges   = float(form_data["TotalCharges"])

    row = {col: 0 for col in feature_columns}

    # Numeric
    row["tenure"]         = tenure
    row["MonthlyCharges"] = monthly_charges
    row["TotalCharges"]   = total_charges
    row["SeniorCitizen"]  = int(form_data.get("SeniorCitizen", 0))

    # Derived
    row["ChargesPerMonth"] = total_charges / (tenure + 1)
    row["HighValue"]       = 1 if monthly_charges > 64.76 else 0   # dataset median
    row["LongTenure"]      = 1 if tenure > 24 else 0

    # Binary yes/no fields
    binary_fields = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for field in binary_fields:
        row[field] = 1 if form_data.get(field) == "Yes" else 0

    # One-hot: gender
    gender = form_data.get("gender", "Male")
    if f"gender_{gender}" in row:
        row[f"gender_{gender}"] = 1

    # One-hot: InternetService
    internet = form_data.get("InternetService", "No")
    key = f"InternetService_{internet}"
    if key in row:
        row[key] = 1

    # One-hot: Contract
    contract = form_data.get("Contract", "Month-to-month")
    key = f"Contract_{contract}"
    if key in row:
        row[key] = 1

    # One-hot: PaymentMethod
    payment = form_data.get("PaymentMethod", "Electronic check")
    key = f"PaymentMethod_{payment}"
    if key in row:
        row[key] = 1

    df_row = pd.DataFrame([row])[feature_columns]
    return scaler.transform(df_row)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    try:
        input_vector = build_input_vector(form_data)
        prediction_idx  = model.predict(input_vector)[0]
        probabilities   = model.predict_proba(input_vector)[0]
        churn_prob      = probabilities[1]
        prediction_label = "Churn" if prediction_idx == 1 else "No Churn"

        log_prediction(form_data, prediction_label, churn_prob)

        return render_template(
            "result.html",
            prediction=prediction_label,
            probability=round(churn_prob * 100, 1),
            tenure=form_data["tenure"],
            monthly=form_data["MonthlyCharges"],
            contract=form_data["Contract"]
        )
    except Exception as e:
        return render_template("index.html", error=str(e))


@app.route("/history")
def history():
    records = get_recent_predictions(limit=20)
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)
