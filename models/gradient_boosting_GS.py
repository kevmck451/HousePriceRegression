from models.save_submission import save_submission
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import pathlib

here = pathlib.Path(__file__)

def gradient_boosting(params):
    dataset_train_filepath = here.parent.parent / "data" / "train_clean.csv"
    dataset_test_filepath = here.parent.parent / "data" / "test_clean.csv"

    train_df = pd.read_csv(dataset_train_filepath)
    test_df = pd.read_csv(dataset_test_filepath)

    X_train = train_df.drop("SalePrice", axis=1)
    Y_train = train_df["SalePrice"]
    X_test = test_df.drop("Id", axis=1).copy()
    X_test = X_test[X_train.columns]

    # Scale and transform highly skewed features
    skewed_features = X_train.columns[X_train.skew() > 0.75]
    pt = PowerTransformer(method='yeo-johnson')  # Log-like transformation for skewed features
    X_train[skewed_features] = pt.fit_transform(X_train[skewed_features])
    X_test[skewed_features] = pt.transform(X_test[skewed_features])

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # XGBRegressor with GridSearchCV
    regressor = XGBRegressor(n_jobs=-1, random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=params,
        scoring='r2',
        cv=kfold,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, Y_train)

    best_model = grid_search.best_estimator_
    best_score = round(grid_search.best_score_ * 100, 2)
    best_params = grid_search.best_params_

    # Save best score and params to a text file
    with open("best_score_params.txt", "w") as f:
        f.write(f"Best Score: {best_score}\n")
        f.write(f"Best Params: {best_params}\n")

    print(f"Best Score: {best_score}, Best Params: {best_params}")

if __name__ == '__main__':
    param_grid = {
        'n_estimators': [100, 500, 1000, 1500, 2000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 10],
        'gamma': [0, 0.1, 0.5, 1],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],
        'reg_lambda': [0, 0.01, 0.1, 1, 10],
    }

    gradient_boosting(params=param_grid)
