

import pandas as pd

def save_submission(test_df, Y_pred):
    dataframe = pd.DataFrame({
            "Id": test_df["Id"],
            "SalePrice": Y_pred
        })

    dataframe.to_csv('RandomForest.csv', index=False)