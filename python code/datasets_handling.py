import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
import seaborn as sns

def vgsales_handling(ds):
    key = 'Other_Sales'
    # sample_length of null values
    length = 100
    df = pd.read_csv(ds).head(500)
    df_original = pd.read_csv(ds).head(500)
    # set random values to nan
    idx = np.random.choice(df.index, size=length, replace=False)
    df.loc[idx, 'Other_Sales'] = np.nan
    mask1_df = df["Other_Sales"].isnull()
    # drop null values from dataset to train
    df_copy1 = df.copy()
    df_copy1.dropna(axis=0, inplace=True)
    # X, Y and X_Test
    X = df_copy1.iloc[:, [6, 7, 8]]
    Y = df_copy1.iloc[:, 9]
    X_Test = df[mask1_df].iloc[:, [6, 7, 8]]
    # predicting the null values using linear regression
    reg = LinearRegression()
    reg.fit(X, Y)
    res = reg.predict(X_Test)
    # Add the predicted values to a new column in the original DataFrame
    df_original.loc[mask1_df, 'Other_Sales_predicted'] = res
    j = 0
    arr = df[df['Other_Sales'].isna()].index
    for i in arr:
        df.at[i, 'Other_Sales'] = float(res[j])
        j += 1
    # actual and predicted values
    y_actual = list(df_original.loc[idx, 'Other_Sales'])
    y_predicted = list(df.loc[idx, 'Other_Sales'])
    return df_original, df, idx, y_actual, y_predicted, key


def a():
    pass



