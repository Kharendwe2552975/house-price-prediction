# House Price Prediction Project

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# 1. Load Dataset
# ---------------------------

# You can replace this with your dataset path
data = pd.read_csv('housing.csv')

# ---------------------------
# 2. Data Preprocessing
# ---------------------------

# Display basic info
print(data.info())

# Handle missing values
data = data.dropna()

# Encode categorical columns
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# ---------------------------
# 3. Feature Selection
# ---------------------------

# Target variable
target = 'price'  # change if dataset uses different name

X = data.drop(target, axis=1)
y = data[target]

# ---------------------------
# 4. Train-Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 5. Model Training
# ---------------------------

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ---------------------------
# 6. Predictions
# ---------------------------

lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# ---------------------------
# 7. Evaluation Function
# ---------------------------

def evaluate_model(name, y_test, preds):
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"\n{name} Performance:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

# Evaluate models
evaluate_model("Linear Regression", y_test, lr_preds)
evaluate_model("Random Forest", y_test, rf_preds)

# ---------------------------
# 8. Visualization
# ---------------------------

plt.scatter(y_test, rf_preds)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Random Forest)")
plt.show()
