print("Training AI model...")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os

# Create sample data
np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=200, freq="D")
prices = 100 + np.cumsum(np.random.randn(200))

df = pd.DataFrame({"Close": prices}, index=dates)
df["Returns"] = df["Close"].pct_change()
df["SMA_10"] = df["Close"].rolling(10).mean()
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

X = df[["Returns", "SMA_10"]].values
y = df["Target"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBClassifier(n_estimators=50, random_state=42)
model.fit(X_scaled, y)

print(f"✅ Model accuracy: {model.score(X_scaled, y):.1%}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(["Returns", "SMA_10"], "models/feature_names.pkl")

print("💾 Models saved to models/ folder")
