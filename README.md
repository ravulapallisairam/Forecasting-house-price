import kagglehub
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Download Dataset
path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction")
print("Path to dataset files:", path)

# Step 2: Load the Dataset
# Note: Adjust the filename if needed based on actual file names
file_path = os.path.join(path, "Housing.csv")
df = pd.read_csv(file_path)

# Step 3: Explore the Data
print(df.head())
print(df.info())

# Step 4: Preprocess
# Convert categorical features using one-hot encoding
df = pd.get_dummies(df)

# Handle missing values (if any)
df = df.fillna(df.mean(numeric_only=True))

# Step 5: Split Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train Smart Regression Model (Gradient Boosting)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Step 9: Make Predictions and Evaluate
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Manually calculate Root Mean Squared Error
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
