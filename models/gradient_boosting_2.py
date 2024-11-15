from models.save_submission import save_submission
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
import numpy as np
import pathlib
from xgboost import XGBRegressor

here = pathlib.Path(__file__)

def gradient_boosting():
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

    # Best parameters from previous GridSearchCV
    best_params = {
        'colsample_bytree': 0.8,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'n_estimators': 1000,
        'reg_alpha': 0.01,
        'reg_lambda': 1,
        'subsample': 0.8
    }

    # Train XGBRegressor with the best parameters
    best_model = XGBRegressor(n_jobs=-1, random_state=42, **best_params)
    best_model.fit(X_train, Y_train)

    best_score = best_model.score(X_train, Y_train) * 100  # Training R2 score

    # Save best score and params to a text file
    with open("best_score_params.txt", "w") as f:
        f.write(f"Best Score (Training): {round(best_score, 2)}\n")
        f.write(f"Best Params: {best_params}\n")

    print(f"Best Score (Training): {round(best_score, 2)}, Best Params: {best_params}")

    # Generate predictions and save submission if the score meets the threshold
    if best_score >= 82.0:  # Adjust threshold as needed
        Y_pred = best_model.predict(X_test)
        save_submission(test_df=test_df, Y_pred=Y_pred, score=round(best_score, 2), name='GB_Optimized_FE')

if __name__ == '__main__':
    gradient_boosting()
