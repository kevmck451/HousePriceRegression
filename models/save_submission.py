
import pandas as pd

def save_submission(test_df, Y_pred, score, name):
    dataframe = pd.DataFrame({
            "Id": test_df["Id"],
            "SalePrice": Y_pred
        })

    dataframe.to_csv(f'submissions/{name}_{score}.csv', index=False)

