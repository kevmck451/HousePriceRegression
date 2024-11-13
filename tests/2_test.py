import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
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

# Define hyperparameter grids for RandomizedSearchCV
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
        'max_depth': [3, 5, 7]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
}

# To store model performance and the best models
best_model_info = None
model_performance = []

# Cross-validation and model selection
for model_name, model in models.items():
    print(f"Training {model_name}...")

    # If model has hyperparameters to tune
    if model_name in param_grids:
        search = RandomizedSearchCV(model, param_grids[model_name], n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
    else:
        # No hyperparameter tuning for simple models like Linear Regression
        best_model = model.fit(X_train, y_train)

    # Cross-validation score to prevent overfitting
    scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_cv_score = np.mean(np.sqrt(-scores))

    # Evaluate on training data
    y_train_pred = best_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2 = r2_score(y_train, y_train_pred)

    print(f"{model_name} - Train RMSE: {train_rmse}, Cross-Validation RMSE: {mean_cv_score}, R²: {r2}")

    # Save the best performing model based on CV score
    if best_model_info is None or mean_cv_score < best_model_info['cv_score']:
        best_model_info = {
            'model_name': model_name,
            'model': best_model,
            'cv_score': mean_cv_score
        }

    # Store model performance for ranking
    model_performance.append({
        'model': model_name,
        'train_rmse': train_rmse,
        'cv_rmse': mean_cv_score,
        'r2': r2
    })

# Output the best model based on cross-validation performance
print(f"\nBest model: {best_model_info['model_name']} with CV RMSE: {best_model_info['cv_score']}")

# Generate predictions using the best model
predictions = pd.DataFrame({
    'Id': test_df['Id'],
    'Predicted_SalePrice': best_model_info['model'].predict(X_test)
})
predictions.to_csv(f"{best_model_info['model_name']}_predictions_{best_model_info['cv_score']}.csv", index=False)

# Visualize model performance
performance_df = pd.DataFrame(model_performance)
performance_df.set_index('model', inplace=True)
performance_df[['train_rmse', 'cv_rmse']].plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance: Train vs CV RMSE')
plt.ylabel('RMSE')
plt.show()

# Save the best model to a file
joblib.dump(best_model_info['model'], 'best_model.pkl')
