# 🩺 Diabetes Prediction – Streamlit ML App

Clean separation of concerns:
- **`model.py`** — all ML logic (preprocessing, training, saving, loading, prediction)
- **`app.py`**  — Streamlit UI only (imports from `model.py`)

---

## 📁 Project Structure

```
diabetes_project/
├── app.py              # Streamlit UI
├── model.py            # ML logic (train / save / load / predict)
├── diabetes.csv        # Dataset ← place your file here
├── requirements.txt
├── README.md
└── saved_models/       # Auto-created after first run
    ├── random_forest_model.pkl
    ├── random_forest_scaler.pkl
    ├── logistic_regression_model.pkl
    ├── logistic_regression_scaler.pkl
    ├── svm_model.pkl
    └── svm_scaler.pkl
```

---

## 🚀 Setup in VS Code

### Step 1 — Open the folder in VS Code

```bash
cd diabetes_project
code .
```

### Step 2 — Create & activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Place the dataset

Copy `diabetes.csv` into the project root (same folder as `app.py`).

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — (Optional) Pre-train all models from terminal

```bash
python model.py
```

This trains all 3 models and saves them to `saved_models/`.

### Step 6 — Run the Streamlit app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 🧠 model.py — Public API

| Function | Description |
|----------|-------------|
| `preprocess(df)` | Replaces biological zeros with median |
| `train(df, model_name)` | Trains model, returns `{model, scaler, metrics}` |
| `save(model, scaler, model_name)` | Saves `.pkl` files to `saved_models/` |
| `load(model_name)` | Loads saved model+scaler, returns `(model, scaler)` |
| `predict(model, scaler, input_values)` | Returns `{prediction, probability}` |
| `AVAILABLE_MODELS` | Dict of all supported classifiers |
| `FEATURES` | List of 8 feature column names |

---

## 🎛️ App Pages

| Page | Content |
|------|---------|
| 🏠 Home | Dataset stats, class distribution pie chart |
| 📊 EDA | Distributions, correlation heatmap, boxplots, pairplot |
| 🤖 Model Performance | Accuracy / F1 / Precision / Recall, confusion matrix, ROC curve, feature importances |
| 🔮 Predict | Enter patient values → risk prediction + probability bar chart |

---

## 🛑 Disclaimer

For **educational purposes only**. Always consult a qualified medical professional.
