from models.save_submission import save_submission
from models.load_data import load_data

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import pathlib


here = pathlib.Path(__file__)

def gradient_boosting(params):
    dataset_train_filepath = here.parent.parent/"data"/"train_clean.csv"
    dataset_test_filepath = here.parent.parent/"data"/"test_clean.csv"

    train_df = pd.read_csv(dataset_train_filepath)
    test_df = pd.read_csv(dataset_test_filepath)

    # train_df, test_df = load_data()

    X_train = train_df.drop("SalePrice", axis=1)
    Y_train = train_df["SalePrice"]
    X_test = test_df.drop("Id", axis=1).copy()
    X_test = X_test[X_train.columns]

    # Standard Scaling and Feature Engineering
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Feature Engineering: Polynomial Features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

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

    # Save best score and params to a text file
    with open("best_score_params.txt", "w") as f:
        f.write(f"Best Score: {best_score}\n")
        f.write(f"Best Params: {grid_search.best_params_}\n")

    print(f"Best Score: {best_score}, Best Params: {grid_search.best_params_}")

    if best_score >= 98:
        Y_pred = best_model.predict(X_test)
        save_submission(test_df=test_df, Y_pred=Y_pred, score=best_score, name='GB_Optimized_FE')

if __name__ == '__main__':
    param_grid = {
        'n_estimators': [500, 700, 1000],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 5, 6],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }

    gradient_boosting(params=param_grid)
