# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================
# 1️⃣ Load data
# =====================
data = pd.read_csv("data/pune_properties.csv")

# Drop null rows
data = data.dropna(subset=["price"]).reset_index(drop=True)

# =====================
# 2️⃣ Feature selection
# =====================
features = [
    "bhk", "carpetarea", "bathroom", "balconies", "floor", "totalfloor",
    "age", "opensides", "area", "locality", "ownership", "status",
    "neworold", "overlooking", "roadfaceing", "possesiondate"
]
target = "price"

# Keep only existing columns
features = [f for f in features if f in data.columns]

X = data[features].copy()
y = data[target]

# =====================
# 3️⃣ Encode categoricals
# =====================
categorical_cols = X.select_dtypes(include=["object"]).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# =====================
# 4️⃣ Scale numericals
# =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================
# 5️⃣ Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =====================
# 6️⃣ Random Forest (intensive)
# =====================
rf = RandomForestRegressor(n_jobs=-1, random_state=42)

param_dist = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [10, 20, 30, 40, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"]
}

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=30,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

best_rf = search.best_estimator_

# =====================
# 7️⃣ Evaluate
# =====================
y_pred = best_rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n🏁 Model Evaluation Results:")
print(f"R² Score  : {r2:.3f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")

# =====================
# 8️⃣ Save models
# =====================
joblib.dump(best_rf, "model/random_forest_model.pkl")
joblib.dump(label_encoders, "model/encoder.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n✅ Model, encoder, and scaler saved successfully!")
