import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import pathlib

# Define the paths based on project structure
here = pathlib.Path(__file__).parent
train_file_path = here / "../data/train_clean.csv"

# Load the training dataset
train_data = pd.read_csv(train_file_path)

# Split features and target
X = train_data.drop(columns=["SalePrice"])
y = train_data["SalePrice"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBRegressor model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, objective='reg:squarederror')

# Training with manual early stopping
min_rmse = float("inf")
best_iteration = 0
train_rmse_history = []
val_rmse_history = []

for i in range(1, model.n_estimators + 1):
    model.set_params(n_estimators=i)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Predict and calculate RMSE for both training and validation sets
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    train_rmse_history.append(train_rmse)
    val_rmse_history.append(val_rmse)

    if val_rmse < min_rmse:
        min_rmse = val_rmse
        best_iteration = i
    else:
        break

# Set the model to the best iteration
model.set_params(n_estimators=best_iteration)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_rmse_history, label="Training RMSE")
plt.plot(val_rmse_history, label="Validation RMSE")
plt.xlabel('Number of Estimators')
plt.ylabel('RMSE')
plt.title('Learning Curve: Training and Validation RMSE')
plt.legend()
plt.grid()
plt.show()

# Make predictions
y_pred = model.predict(X_val)

# Residual analysis
residuals = y_val - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Prices')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Analysis')
plt.grid()
plt.show()

# Prediction distribution
plt.figure(figsize=(10, 6))
plt.hist(y_val, bins=50, alpha=0.6, label='Actual Sale Prices')
plt.hist(y_pred, bins=50, alpha=0.6, label='Predicted Sale Prices')
plt.xlabel('Sale Prices')
plt.ylabel('Frequency')
plt.title('Actual vs. Predicted Sale Prices Distribution')
plt.legend()
plt.grid()
plt.show()

# Feature importance visualization
feature_importance = model.feature_importances_
plt.figure(figsize=(12, 6))
plt.barh(X_train.columns, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance from XGBoost Model')
plt.grid()
plt.show()

# Print model performance metrics
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
print(f"Validation RMSE: {rmse}")
print(f"Validation MAE: {mae}")
