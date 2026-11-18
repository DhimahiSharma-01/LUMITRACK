from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_models():
    required = ["failure_model.pkl", "energy_model.pkl", "anomaly_model.pkl"]
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required):
        from train_model import train_all
        train_all()
    return (
        joblib.load(os.path.join(MODEL_DIR, "failure_model.pkl")),
        joblib.load(os.path.join(MODEL_DIR, "energy_model.pkl")),
        joblib.load(os.path.join(MODEL_DIR, "anomaly_model.pkl"))
    )

failure_model, energy_model, anomaly_model = load_models()

FEATURES = [
    "hours_used", "temperature", "voltage", "current",
    "motion_count", "brightness", "age_years"
]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    values = {
        "hours_used": "10",
        "temperature": "32",
        "voltage": "220",
        "current": "1.8",
        "motion_count": "35",
        "brightness": "80",
        "age_years": "3"
    }

    if request.method == "POST":
        try:
            values = {key: request.form.get(key, "") for key in FEATURES}
            row = pd.DataFrame([[float(values[k]) for k in FEATURES]], columns=FEATURES)

            failure_probability = float(failure_model.predict_proba(row)[0][1]) * 100
            failure_label = "High risk" if failure_probability >= 55 else (
                "Medium risk" if failure_probability >= 30 else "Low risk"
            )

            energy_prediction = max(0, float(energy_model.predict(row)[0]))
            anomaly = int(anomaly_model.predict(row)[0]) == -1

            if failure_label == "High risk":
                recommendation = "Schedule an inspection soon and check the electrical connections."
            elif anomaly:
                recommendation = "Unusual operating readings detected. Inspect this streetlight."
            else:
                recommendation = "Streetlight readings look normal. Continue routine monitoring."

            result = {
                "failure_probability": round(failure_probability, 1),
                "failure_label": failure_label,
                "energy": round(energy_prediction, 2),
                "anomaly": "Anomaly detected" if anomaly else "Normal pattern",
                "recommendation": recommendation
            }
        except (ValueError, TypeError):
            result = {"error": "Please enter valid numeric values in every field."}

    return render_template("index.html", result=result, values=values)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
