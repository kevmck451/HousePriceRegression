


import pandas as pd
import pathlib


here = pathlib.Path(__file__)


def load_data():
    dataset_train_filepath = here.parent.parent / "data" / "train_clean.csv"
    dataset_test_filepath = here.parent.parent / "data" / "test_clean.csv"

    train_df = pd.read_csv(dataset_train_filepath)
    test_df = pd.read_csv(dataset_test_filepath)

    return train_df, test_df







