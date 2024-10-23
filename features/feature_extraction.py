


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# setup --------------------------------------------------------------
dataset_train_filepath = '../data/train_clean.csv'
dataset_test_filepath = '../data/test_clean.csv'

train_df = pd.read_csv(dataset_train_filepath)
test_df = pd.read_csv(dataset_test_filepath)

# data type and missing values of each column
# train_df.info()
# test_df.info()
# train_df.describe()
# test_df.describe()
# train_df.head()
# test_df.head()


# visualize --------------------------------------------------------------
# plt.figure(figsize=(16, 4))  # Set the figure size in inches
# sns.histplot(train_df['SalePrice'], kde=True)
# plt.title('Distribution of Sale Prices')
# plt.xlabel('Sale Price')
# plt.ylabel('Frequency')
# plt.show()

plt.rcParams['figure.figsize'] = (9, 9)
numeric_train_df = train_df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_train_df.corr()
g = sns.heatmap(corr_matrix, annot=True, fmt=".1f", annot_kws={"size": 2})  # Adjust 'size' as needed
plt.title('Correlation Matrix')
plt.tight_layout(pad=1)
plt.show()

# sns.barplot(x='YearBuilt', y='SalePrice', data=train_df)
# plt.show()

# sns.barplot(x='SaleCondition', y='SalePrice', data=train_df)
# plt.show()

# sns.barplot(x='YrSold', y='SalePrice', data=train_df)
# plt.show()

# clean up --------------------------------------------------------------
# train_df=train_df.drop("Id",axis=1)
# train_df=train_df.drop("Alley",axis=1)
# train_df=train_df.drop("PoolQC",axis=1)
# train_df=train_df.drop("Fence",axis=1)
# train_df=train_df.drop("MiscFeature",axis=1)
# test_df=test_df.drop("Alley",axis=1)
# test_df=test_df.drop("PoolQC",axis=1)
# test_df=test_df.drop("Fence",axis=1)
# test_df=test_df.drop("MiscFeature",axis=1)
#
# train_df["LotFrontage"] = train_df["LotFrontage"].fillna(train_df["LotFrontage"].mean())
# train_df["MasVnrArea"] = train_df["MasVnrArea"].fillna(train_df["MasVnrArea"].mean())
# train_df["GarageYrBlt"] = train_df["GarageYrBlt"].fillna(2001)
#
# c = ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical", "FireplaceQu", "BsmtFinType1")
# for col in c:
#   if train_df[col].dtype == "object":
#     train_df[col] = train_df[col].fillna("None")
#
# ''' OR
# for col in ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical", "FireplaceQu", "BsmtFinType1"):
#   test_df[col] = test_df[col].fillna('None')
# '''
#
# test_df["LotFrontage"] = test_df["LotFrontage"].fillna(test_df["LotFrontage"].mean())
# test_df["MasVnrArea"] = test_df["MasVnrArea"].fillna(test_df["MasVnrArea"].mean())
# test_df["GarageYrBlt"] = test_df["GarageYrBlt"].fillna(2001)
# test_df["GarageCars"] = test_df["GarageCars"].fillna(0)
# test_df["GarageArea"] = test_df["GarageArea"].fillna(test_df["GarageArea"].mean())
# test_df["BsmtFullBath"] = test_df["BsmtFullBath"].fillna(0)
# test_df["BsmtHalfBath"] = test_df["BsmtHalfBath"].fillna(0)
# test_df["BsmtFinSF1"] = test_df["BsmtFinSF1"].fillna(test_df["BsmtFinSF1"].mean())
# test_df["BsmtFinSF2"] = test_df["BsmtFinSF2"].fillna(test_df["BsmtFinSF2"].mean())
# test_df["TotalBsmtSF"] = test_df["TotalBsmtSF"].fillna(test_df["TotalBsmtSF"].mean())
# test_df["BsmtUnfSF"] = test_df["BsmtUnfSF"].fillna(test_df["BsmtUnfSF"].mean())
#
# c = ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical","MSZoning","Utilities","Exterior1st","Exterior2nd","KitchenQual","Functional","FireplaceQu","SaleType", "BsmtFinType1")
# for col in c:
#   if test_df[col].dtype == "object":
#     test_df[col] = test_df[col].fillna("None")
#
# ''' OR
# for col in ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical","MSZoning","Utilities","Exterior1st","Exterior2nd","KitchenQual","Functional","FireplaceQu","SaleType", "BsmtFinType1"):
#   test_df[col] = test_df[col].fillna('None')
# '''

# train_df.info()
# test_df.info()








