import os
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/training_data.csv"
MODEL_PATH = "models/fit_model.pkl"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}")
    df = pd.read_csv(path)
    required_cols = {"similarity", "skill_count", "experience_years", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    return df


def train_and_save_model():
    print(f"[INFO] Loading training data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)

    X = df[["similarity", "skill_count", "experience_years"]]
    y = df["label"].astype(int)

    # Train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Build a pipeline: Standardize features → Logistic Regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Test Accuracy: {acc:.3f}")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[SUCCESS] Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
