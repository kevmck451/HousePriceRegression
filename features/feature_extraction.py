


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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









