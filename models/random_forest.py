
from models.save_submission import save_submission

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd


def random_forrest_model(num_principal_comps, n_estimators, random_state):
    # setup --------------------------------------------------------------
    dataset_train_filepath = '../data/train_clean.csv'
    dataset_test_filepath = '../data/test_clean.csv'

    train_df = pd.read_csv(dataset_train_filepath)
    test_df = pd.read_csv(dataset_test_filepath)

    # Drop target column 'SalePrice' for X_train
    X_train = train_df.drop("SalePrice", axis=1)
    Y_train = train_df["SalePrice"]

    # Drop 'Id' column from test data for X_test
    X_test  = test_df.drop("Id", axis=1).copy()

    # Ensure X_test columns match X_train
    X_test = X_test[X_train.columns]

    # Apply standard scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Principle Component Analysis
    pca = PCA(n_components = num_principal_comps)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)

    regressor.score(X_train, Y_train)
    score = round(regressor.score(X_train, Y_train) * 100, 2)

    if score >= 98.05:
        save_submission(test_df=test_df, Y_pred=Y_pred, score=score, name='RF')





if __name__ == '__main__':

    num_principal_comps = 11
    n_estimators = 200
    random_state = 0

    random_forrest_model(
        num_principal_comps=num_principal_comps,
        n_estimators=n_estimators,
        random_state=random_state
    )


