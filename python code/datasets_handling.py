import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# handling with vgsales dataset
def vgsales_handling(ds):
    # column that we want to predict
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
    # fill the null values with the predicted values
    for i in arr:
        df.at[i, 'Other_Sales'] = float(res[j])
        j += 1
    # actual and predicted values
    y_actual = list(df_original.loc[idx, 'Other_Sales'])
    y_predicted = list(df.loc[idx, 'Other_Sales'])
    return df_original, df, idx, y_actual, y_predicted, key

# handling with students performance dataset
def StudentsPerformance_handling(ds):
    # the column that we want to predict
    key = 'reading score'
    # sample_length of null values
    length = 100
    df = pd.read_csv(ds).head(500)
    df_original = pd.read_csv(ds).head(500)
    # set random values to nan
    idx = np.random.choice(df.index, size=length, replace=False)
    df.loc[idx, 'reading score'] = np.nan
    mask1_df = df["reading score"].isnull()
    # drop null values from dataset to train
    df_copy1 = df.copy()
    df_copy1.dropna(axis=0, inplace=True)
    # X, Y and X_Test
    X = df_copy1.iloc[:, [5, 7]]
    Y = df_copy1.iloc[:, 6]
    X_Test = df[mask1_df].iloc[:, [5, 7]]
    # predicting the null values using linear regression
    reg = LinearRegression()
    reg.fit(X, Y)
    res = reg.predict(X_Test)
    # Add the predicted values to a new column in the original DataFrame
    df_original.loc[mask1_df, 'Other_Sales_predicted'] = res
    j = 0
    arr = df[df['reading score'].isna()].index
    # filling the null values with the predicted values
    for i in arr:
        df.at[i, 'reading score'] = float(res[j])
        j += 1
    # actual and predicted values
    y_actual = list(df_original.loc[idx, 'reading score'])
    y_predicted = list(df.loc[idx, 'reading score'])
    return df_original, df, idx, y_actual, y_predicted, key

# handling with forecast dataset
def Mart_Sale_Forecast_handling(ds):
    # sample number of null values
    length = 150
    df = pd.read_csv(ds).head(700)
    df_original = pd.read_csv(ds).head(700)
    df_plot_org = df
    # set some null values
    idx = np.random.choice(df.index, size=length, replace=False)
    df.loc[idx, 'Outlet_Location_Type'] = np.nan
    mask1_df = df["Outlet_Location_Type"].isnull()
    # drop null values from dataset to train
    df_copy1 = df.copy()
    df_copy1.dropna(axis=0, inplace=True)
    # predicting the null values using random forest algorithm
    X = df_copy1.iloc[:, [1, 3, 5, 7, 10]]
    Y = df_copy1.iloc[:, 8]
    X_Test = df[mask1_df].iloc[:, [1, 3, 5, 7, 10]]
    X_Test = np.nan_to_num(X_Test)
    # using random forest classifier to predict values
    classifier = RandomForestClassifier()
    classifier = classifier.fit(X, Y)
    res = classifier.predict(X_Test)
    # Add the predicted values to a new column in the original DataFrame
    df_original.loc[mask1_df, 'Outlet_Location_Type_predicted'] = res
    # fill null values with predicted values
    j = 0
    arr = df[df['Outlet_Location_Type'].isna()].index
    for i in arr:
        df.at[i, 'Outlet_Location_Type'] = res[j]
        j += 1
    df_plot_after = df
    # actual values and predicted values as lists
    y_actual = list(df_original.loc[idx, 'Outlet_Location_Type'])
    y_predicted = list(df.loc[idx, 'Outlet_Location_Type'])
    counter = 0
    num = list(df_original['Outlet_Location_Type'])[0]
    # finding the most frequent class
    for i in list(df_original['Outlet_Location_Type']):
        curr_frequency = list(df_original['Outlet_Location_Type']).count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    most_freq = num
    # fill nulls with the most frequent value
    j = 0
    for i in arr:
        df.at[i, 'Outlet_Location_Type'] = most_freq
        j += 1
    return df_plot_org, df_plot_after, y_actual, y_predicted, most_freq

# handling with titanic dataset
def titanic_handling(ds):
    # sample number for null values
    length = 100
    # read the data
    df = pd.read_csv(ds)
    df_original = pd.read_csv(ds)
    df_plot_org = df
    # set some null values
    idx = np.random.choice(df.index, size=length, replace=False)
    df.loc[idx, 'Embarked'] = np.nan
    mask1_df = df["Embarked"].isnull()
    # drop null values from dataset to train
    df_copy1 = df.copy()
    df_copy1.dropna(axis=0, inplace=True)
    # predicting the null values using random forest algorithm
    X = df_copy1.iloc[:, [1, 2, 5, 6, 7, 9]]
    Y = df_copy1.iloc[:, 10]
    X_Test = df[mask1_df].iloc[:, [1, 2, 5, 6, 7, 9]]
    X_Test = np.nan_to_num(X_Test)
    # using random forest classifier to predict null values
    classifier = RandomForestClassifier()
    classifier = classifier.fit(X, Y)
    res = classifier.predict(X_Test)
    # Add the predicted values to a new column in the original DataFrame
    df_original.loc[mask1_df, 'Embarked_predicted'] = res
    # fill nan values with predicted values
    j = 0
    arr = df[df['Embarked'].isna()].index
    for i in arr:
        df.at[i, 'Embarked'] = res[j]
        j += 1
    df_plot_after = df
    # actual and predicted values as lists
    y_actual = list(df_original.loc[idx, 'Embarked'])
    y_predicted = list(df.loc[idx, 'Embarked'])
    counter = 0
    # finding the most frequent class
    num = list(df_original['Embarked'])[0]
    for i in list(df_original['Embarked']):
        curr_frequency = list(df_original['Embarked']).count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    most_freq = num
    # fill nulls with the most frequent value
    j = 0
    for i in arr:
        df.at[i, 'Embarked'] = most_freq
        j += 1
    return df_plot_org, df_plot_after, y_actual, y_predicted, most_freq
