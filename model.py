"""
model.py — ML training, evaluation, and persistence.
Called by app.py; can also be run standalone to retrain:
    python model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, f1_score, precision_score, recall_score,
)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
MODEL_DIR = "saved_models"
MODEL_NAME = "Random Forest"

AVAILABLE_MODELS = {
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
}


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically impossible zeros with column median."""
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df = df.copy()
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df


# ── Train ─────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, model_name: str = "Random Forest") -> dict:
    
    """
    Train the chosen model on df.
    Returns
    -------
    dict with keys:
        model, scaler, metrics
    """

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(AVAILABLE_MODELS)}")

    df = preprocess(df)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Clone a fresh estimator so re-runs don't accumulate state
    import sklearn.base as skb
    model = skb.clone(AVAILABLE_MODELS[model_name])
    model.fit(X_train_sc, y_train)

    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy":     accuracy_score(y_test, y_pred),
        "f1":           f1_score(y_test, y_pred),
        "precision":    precision_score(y_test, y_pred),
        "recall":       recall_score(y_test, y_pred),
        "cm":           confusion_matrix(y_test, y_pred),
        "report":       classification_report(y_test, y_pred, output_dict=True),
        "fpr":          fpr,
        "tpr":          tpr,
        "auc":          roc_auc,
        "X_test":       X_test,
        "y_test":       y_test,
        "y_pred":       y_pred,
        "model_name":   model_name,
    }

    # Feature importances (Random Forest only)
    if hasattr(model, "feature_importances_"):
        metrics["feature_importances"] = pd.Series(
            model.feature_importances_, index=FEATURES
        ).sort_values(ascending=False)

    return {"model": model, "scaler": scaler, "metrics": metrics}


# ── Save / Load ───────────────────────────────────────────────────────────────
def _paths(model_name: str):
    safe = model_name.replace(" ", "_").lower()
    os.makedirs(MODEL_DIR, exist_ok=True)
    return (
        os.path.join(MODEL_DIR, f"{safe}_model.pkl"),
        os.path.join(MODEL_DIR, f"{safe}_scaler.pkl"),
    )


def save(model, scaler, model_name: str):
    """Persist model + scaler to disk."""
    model_path, scaler_path = _paths(model_name)
    with open(model_path,  "wb") as f: pickle.dump(model,  f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    print(f"[model.py] Saved → {model_path}, {scaler_path}")


def load(model_name: str):
    """Load previously saved model + scaler. Returns (model, scaler) or (None, None)."""
    model_path, scaler_path = _paths(model_name)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path,  "rb") as f: model  = pickle.load(f)
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
        return model, scaler
    return None, None


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model, scaler, input_values: list) -> dict:
    """
    Make a single prediction.

    Parameters
    ----------
    input_values : list of 8 floats in FEATURES order

    Returns
    -------
    dict with 'prediction' (0/1) and 'probability' ([prob_0, prob_1])
    """
    arr = np.array(input_values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred  = int(model.predict(arr_scaled)[0])
    proba = model.predict_proba(arr_scaled)[0].tolist()
    return {"prediction": pred, "probability": proba}


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = "diabetes.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'")

    df = pd.read_csv(DATA_PATH)

    print(f"\n{'='*50}")
    print(f"Training: {MODEL_NAME}")
    result = train(df, MODEL_NAME)
    m = result["metrics"]
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  ROC-AUC   : {m['auc']:.4f}")
    save(result["model"], result["scaler"], MODEL_NAME)

    print("\n✅ Random Forest model trained and saved to ./saved_models/")
