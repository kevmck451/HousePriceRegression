import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Load datasets
dataset_train_filepath = '../data/train_clean.csv'
dataset_test_filepath = '../data/test_clean.csv'

train_df = pd.read_csv(dataset_train_filepath)
test_df = pd.read_csv(dataset_test_filepath)

# Prepare training and test data
X_train = train_df.drop("SalePrice", axis=1)
y_train = train_df["SalePrice"]
X_test = test_df.drop("Id", axis=1).copy()

# Ensure X_test columns match X_train
X_test = X_test[X_train.columns]

# Apply standard scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply PCA
pca = PCA(n_components=11)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(),
    'Neural Network': MLPRegressor(max_iter=1000)
}

# Hyperparameters for tuning
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 10]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'solver': ['adam', 'lbfgs']
    }
}

# Track model performance
results = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Check if hyperparameter tuning is needed
    if model_name in param_grids:
        param_grid = param_grids[model_name]
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"Best parameters for {model_name}: {search.best_params_}")
    else:
        model.fit(X_train, y_train)
        best_model = model

    # Make predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Evaluate performance
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    # Store results
    results[model_name] = {
        'model': best_model,
        'rmse_train': rmse_train,
        'r2_train': r2_train
    }

    print(f"{model_name} - RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")

# Rank models based on performance (e.g., RMSE)
ranked_models = sorted(results.items(), key=lambda x: x[1]['rmse_train'])

# Output the best model
best_model_name, best_model_info = ranked_models[0]
print(f"\nBest Model: {best_model_name} with RMSE: {best_model_info['rmse_train']:.4f}")

# Visualize model performance
model_names = [name for name, _ in ranked_models]
rmse_values = [info['rmse_train'] for _, info in ranked_models]

plt.figure(figsize=(10, 6))
plt.barh(model_names, rmse_values, color='skyblue')
plt.xlabel('RMSE')
plt.title('Model Performance Comparison')
plt.show()

# Optionally, save the trained models
for model_name, info in results.items():
    joblib.dump(info['model'], f'{model_name}_model.pkl')

# Optionally, save predictions
predictions = pd.DataFrame({
    'Id': test_df['Id'],
    'Predicted_SalePrice': best_model_info['model'].predict(X_test)
})
predictions.to_csv('house_price_predictions.csv', index=False)
