


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


def save_submission(test_df, Y_pred, score):
    dataframe = pd.DataFrame({
            "Id": test_df["Id"],
            "SalePrice": Y_pred
        })

    dataframe.to_csv(f'RF_{score}.csv', index=False)





# setup --------------------------------------------------------------
dataset_train_filepath = '../data/train_clean.csv'
dataset_test_filepath = '../data/test_clean.csv'

train_df = pd.read_csv(dataset_train_filepath)
test_df = pd.read_csv(dataset_test_filepath)


# 7 - 16
n_components = [x for x in range(7, 17)]

scores = []

for comp_num in n_components:

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

    # Output transformed data and target
    # print(X_train)
    # print(Y_train)

    # Principle Component Analysis
    pca = PCA(n_components = comp_num)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)


    ''' OR
    from xgboost import XGBRegressor
    regressor = XGBRegressor()
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    '''

    regressor.score(X_train, Y_train)
    score = round(regressor.score(X_train, Y_train) * 100, 2)
    print(f'Num: {comp_num} / Score: {score}')

    scores.append(score)

    if score >= 98.1:
        save_submission(test_df=test_df, Y_pred=Y_pred, score=score)


print(np.max(scores))



# save_submission(test_df=test_df, Y_pred=Y_pred)



