import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURES = [
    "hours_used", "temperature", "voltage", "current",
    "motion_count", "brightness", "age_years"
]

def create_dataset(n=2500, seed=42):
    """Create a reproducible demo dataset when no real sensor data is available."""
    rng = np.random.default_rng(seed)
    hours = rng.uniform(1, 16, n)
    temp = rng.uniform(10, 55, n)
    voltage = rng.normal(220, 16, n).clip(160, 280)
    current = rng.normal(1.7, 0.45, n).clip(0.3, 3.5)
    motion = rng.poisson(30, n).clip(0, 120)
    brightness = rng.uniform(20, 100, n)
    age = rng.uniform(0, 12, n)

    # Synthetic energy estimate in kWh for one operating period.
    energy = (hours * (brightness / 100) * (current * voltage / 1000) * 0.85)
    energy += rng.normal(0, 0.12, n)
    energy = energy.clip(0.05, None)

    # Transparent rule-based labels create training targets for this demo.
    risk_score = (
        (age > 7).astype(int) * 1.4 +
        (temp > 43).astype(int) * 1.0 +
        (voltage < 190).astype(int) * 1.1 +
        (voltage > 250).astype(int) * 1.0 +
        (current > 2.7).astype(int) * 1.0 +
        (hours > 13).astype(int) * 0.7 +
        rng.normal(0, 0.55, n)
    )
    failure = (risk_score > 1.55).astype(int)

    df = pd.DataFrame({
        "hours_used": hours,
        "temperature": temp,
        "voltage": voltage,
        "current": current,
        "motion_count": motion,
        "brightness": brightness,
        "age_years": age,
        "energy_kwh": energy,
        "failure": failure
    })
    return df

def train_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    data_path = os.path.join(DATA_DIR, "streetlight_data.csv")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = create_dataset()
        df.to_csv(data_path, index=False)

    X = df[FEATURES]
    y_failure = df["failure"]
    y_energy = df["energy_kwh"]

    X_train, X_test, yf_train, yf_test = train_test_split(
        X, y_failure, test_size=0.2, random_state=42, stratify=y_failure
    )
    failure_model = RandomForestClassifier(
        n_estimators=180, max_depth=10, random_state=42, class_weight="balanced"
    )
    failure_model.fit(X_train, yf_train)
    failure_accuracy = accuracy_score(yf_test, failure_model.predict(X_test))

    Xe_train, Xe_test, ye_train, ye_test = train_test_split(
        X, y_energy, test_size=0.2, random_state=42
    )
    energy_model = RandomForestRegressor(
        n_estimators=180, max_depth=12, random_state=42
    )
    energy_model.fit(Xe_train, ye_train)
    energy_mae = mean_absolute_error(ye_test, energy_model.predict(Xe_test))

    # Isolation Forest learns typical sensor patterns and flags unusual ones.
    anomaly_model = make_pipeline(
        StandardScaler(),
        IsolationForest(contamination=0.06, random_state=42)
    )
    anomaly_model.fit(X)

    joblib.dump(failure_model, os.path.join(MODEL_DIR, "failure_model.pkl"))
    joblib.dump(energy_model, os.path.join(MODEL_DIR, "energy_model.pkl"))
    joblib.dump(anomaly_model, os.path.join(MODEL_DIR, "anomaly_model.pkl"))

    print("Training completed.")
    print(f"Failure model accuracy on demo test split: {failure_accuracy:.2%}")
    print(f"Energy model MAE on demo test split: {energy_mae:.3f} kWh")
    print(f"Dataset saved at: {data_path}")

if __name__ == "__main__":
    train_all()
