"""
database.py
-----------
Handles SQLite logging of every prediction made via the Flask app.
Table: predictions
"""

import sqlite3
from datetime import datetime

DB_PATH = "predictions.db"


def init_db():
    """Create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            tenure          INTEGER,
            monthly_charges REAL,
            total_charges   REAL,
            contract        TEXT,
            internet_service TEXT,
            payment_method  TEXT,
            churn_prediction TEXT,
            churn_probability REAL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(input_data: dict, prediction: str, probability: float):
    """Log a single prediction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, tenure, monthly_charges, total_charges,
            contract, internet_service, payment_method,
            churn_prediction, churn_probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        input_data.get("tenure"),
        input_data.get("MonthlyCharges"),
        input_data.get("TotalCharges"),
        input_data.get("Contract"),
        input_data.get("InternetService"),
        input_data.get("PaymentMethod"),
        prediction,
        round(probability, 4)
    ))
    conn.commit()
    conn.close()


def get_recent_predictions(limit: int = 10):
    """Fetch the most recent predictions for the audit log page."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
