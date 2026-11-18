LUMITRACK – SMART STREETLIGHT INTELLIGENCE
Machine Learning Academic Project

FEATURES
1. Failure prediction using Random Forest Classifier.
2. Energy consumption estimation using Random Forest Regressor.
3. Unusual sensor-pattern detection using Isolation Forest.
4. Flask dashboard to enter sample sensor values and view predictions.
5. About page explaining the workflow and limitations.

TECHNOLOGIES
Python, Flask, Pandas, NumPy, Scikit-learn, Joblib, HTML, CSS.

IMPORTANT
This is a software-based ML prototype. It uses a synthetic demonstration dataset
created by train_model.py. It is not connected to physical streetlights or live IoT
sensors. The predictions are for academic demonstration only.

HOW TO RUN (WINDOWS)
1. Extract the ZIP file.
2. Open the extracted LUMITRACK_ML_Project folder in VS Code.
3. Open Terminal in VS Code.
4. (Optional but recommended) Create a virtual environment:
      py -m venv venv
      .\\venv\\Scripts\\activate
5. Install required libraries:
      py -m pip install -r requirements.txt
6. Train the models:
      py train_model.py
   This creates the CSV dataset in data/ and trained models in models/.
7. Start the website:
      py app.py
8. Open the local address shown in the terminal:
      http://127.0.0.1:5000

If you skip step 6, app.py automatically trains the models the first time it runs.

PROJECT FILES
app.py                 Flask website and prediction logic
train_model.py         Dataset creation and ML model training
requirements.txt       Required Python libraries
data/                  Generated sample streetlight dataset
models/                Trained ML model files (created after training)
templates/             HTML pages
static/style.css       Website design

MODEL EXPLANATION
Random Forest Classifier:
Uses multiple decision trees to classify a streetlight's failure risk.

Random Forest Regressor:
Predicts a numeric energy-use value (kWh).

Isolation Forest:
Detects readings that look different from normal sensor patterns.

The synthetic target labels are created using illustrative rules. For a real project,
replace the generated data with verified sensor data and validate the models.
