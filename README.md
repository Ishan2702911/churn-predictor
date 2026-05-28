# Customer Churn Prediction Web App

A Flask-based web app that predicts customer churn probability using a
Random Forest model trained on the Telco Customer Churn dataset.

## Project Structure

```
churn-app/
├── app.py                  Flask app + prediction routes
├── train.py                Train model, save artifacts
├── database.py             SQLite logging helpers
├── requirements.txt
├── render.yaml             Render deployment config
├── templates/
│   ├── index.html          Input form
│   ├── result.html         Prediction output
│   └── history.html        Audit log
└── static/
    └── style.css
```

## Setup (Local)

### 1. Get the dataset
Download from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Save the file as:
```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```
in the root of this project folder.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (run once)
```bash
python train.py
```
This creates: `model.pkl`, `scaler.pkl`, `feature_columns.pkl`

### 4. Run the app
```bash
python app.py
```
Open: http://127.0.0.1:5000

---

## Deploy to Render (Free)

1. Push this entire folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys

> Note: Add the CSV file to the repo OR modify `train.py` to
> download it automatically via the Kaggle API.

---

## Tech Stack
- Python, Flask, Scikit-learn, Pandas, NumPy
- Random Forest Classifier (300 estimators)
- SQLite for prediction logging
- Deployed on Render
# churn-predictor
