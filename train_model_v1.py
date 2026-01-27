import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("sales.csv")

# Basic cleaning (safe defaults)
df = df.dropna(subset=["Price", "Discount", "Units_Sold"])

# Discount is in %, create effective price after discount
df["net_price"] = df["Price"] * (1 - df["Discount"] / 100)

# v1 baseline features (simple pricing drivers)
X = df[["net_price", "Discount"]]
y = df["Units_Sold"]

# Train-test split (keep consistent for v1 vs v2 comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model_v1 = LinearRegression()
model_v1.fit(X_train, y_train)

# Evaluate (minimal)
y_pred = model_v1.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"[v1] MAE: {mae:.2f}")
print(f"[v1] R2:  {r2:.2f}")

# Save model
joblib.dump(model_v1, "units_sold_model_v1.pkl")
print("Saved: units_sold_model_v1.pkl")

