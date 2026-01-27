import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("sales.csv")

# Basic cleaning
needed = ["Price", "Discount", "Product_Category", "Customer_Segment", "Units_Sold"]
df = df.dropna(subset=needed)

# Ensure categorical columns are strings (avoids dtype issues)
df["Product_Category"] = df["Product_Category"].astype(str)
df["Customer_Segment"] = df["Customer_Segment"].astype(str)

# Discount is in %, create effective price after discount
df["net_price"] = df["Price"] * (1 - df["Discount"] / 100)

# Features and target
numeric_features = ["net_price", "Discount"]
categorical_features = ["Product_Category", "Customer_Segment"]

X = df[numeric_features + categorical_features]
y = df["Units_Sold"]

# Preprocess categoricals using one-hot encoding
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# v2 pipeline (ONLY model changed)
model_v2 = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", GradientBoostingRegressor(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3
    ))
])

# Train-test split (same settings as v1 for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model_v2.fit(X_train, y_train)

# Evaluate
y_pred = model_v2.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"[v2] MAE: {mae:.2f}")
print(f"[v2] R2:  {r2:.2f}")

# Save model
joblib.dump(model_v2, "units_sold_model_v2.pkl")
print("Saved: units_sold_model_v2.pkl")

