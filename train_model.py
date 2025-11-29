import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load training data
df = pd.read_csv("data/training_data.csv")

X = df[["similarity", "skill_count", "experience_years"]]
y = df["label"]

# Build pipeline (scaler + model)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "models/fit_model.pkl")

print("Model trained and saved to models/fit_model.pkl")
