# import numpy as np
# import pandas as pd
# import warnings
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
from datasets_handling import vgsales_handling
from empirical_experiments_utils import RMSE
from plotting_utils import plot1, plot2
# warnings.filterwarnings('ignore')
# df1 = pd.read_csv("HousingData.csv")
# mask1_df1 = df1["CRIM"].isnull()
# mask2_df1 = df1["ZN"].isnull()
# mask3_df1 = df1["INDUS"].isnull()
# mask4_df1 = df1["CHAS"].isnull()
# mask5_df1 = df1["AGE"].isnull()
# mask6_df1 = df1["LSTAT"].isnull()
# # print(df1.isnull().sum())
# df1_copy = df1.copy()
# df1_copy.dropna(axis=0, inplace=True)
# # For finding missing values of "CRIM"
# X = df1_copy.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
# Y = df1_copy.iloc[:, 0]
# X_Test = df1[mask1_df1].iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# # print("CRIM:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df1[df1['CRIM'].isna()].index
# j = 0
# for i in arr:
#     df1.at[i, 'CRIM'] = preds[j]
#     j += 1
# # For finding missing values of "ZN"
# X = df1_copy.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
# Y = df1_copy.iloc[:, 1]
# X_Test = df1[mask2_df1].iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# # print("ZN:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df1[df1['ZN'].isna()].index
# j = 0
# for i in arr:
#     df1.at[i, 'ZN'] = preds[j]
#     j += 1
# # For finding missing values of "INDUS"
# X = df1_copy.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
# Y = df1_copy.iloc[:, 2]
# X_Test = df1[mask3_df1].iloc[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# # print("INDUS:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df1[df1['INDUS'].isna()].index
# j = 0
# for i in arr:
#     df1.at[i, 'INDUS'] = preds[j]
#     j += 1
# # For finding missing values of "CHAS"
# X = df1_copy.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
# Y = df1_copy.iloc[:, 3]
# X_Test = df1[mask4_df1].iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# # print("CHAS:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df1[df1['CHAS'].isna()].index
# j = 0
# for i in arr:
#     df1.at[i, 'CHAS'] = preds[j]
#     j += 1
# # For finding missing values of "AGE"
# X = df1_copy.iloc[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]]
# Y = df1_copy.iloc[:, 6]
# X_Test = df1[mask5_df1].iloc[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# # print("AGE:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df1[df1['AGE'].isna()].index
# j = 0
# for i in arr:
#     df1.at[i, 'AGE'] = preds[j]
#     j += 1
# # For finding missing values of "LSTAT"
# X = df1_copy.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]]
# Y = df1_copy.iloc[:, 12]
# X_Test = df1[mask6_df1].iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# # print("LSTAT:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df1[df1['LSTAT'].isna()].index
# j = 0
# for i in arr:
#     df1.at[i, 'LSTAT'] = preds[j]
#     j += 1
#
# df2 = pd.read_csv("diamond_data_merged.csv")
# print(df2.isnull().sum())
# mask1_df2 = df2["inflation rate"].isnull()
# mask2_df2 = df2["interest rate"].isnull()
# mask3_df2 = df2["gold price"].isnull()
# df2_copy1 = df2.copy()
# df2_copy1.dropna(axis=0, inplace=True)
# # For finding missing values of "inflation rate"
# X = df2_copy1.iloc[:, [0, 2, 3, 4]]
# Y = df2_copy1.iloc[:, 1]
# X_Test = df2[mask1_df2].iloc[:, [0, 2, 3, 4]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# print("inflation rate:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df2[df2['inflation rate'].isna()].index
# j = 0
# for i in arr:
#     df2.at[i, 'inflation rate'] = preds[j]
#     j += 1
# print(df2.isnull().sum())
# # For finding missing values of "interest rate"
# df2_copy2 = df2.copy()
# df2_copy2.dropna(axis=0, inplace=True)
# X = df2_copy2.iloc[:, [0, 1, 3, 4]]
# Y = df2_copy2.iloc[:, 2]
# X_Test = df2[mask2_df2].iloc[:, [0, 1, 3, 4]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# print("interest rate:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df2[df2['interest rate'].isna()].index
# j = 0
# for i in arr:
#     df2.at[i, 'interest rate'] = preds[j]
#     j += 1
# print(df2.isnull().sum())
# # For finding missing values of "gold price"
# df2_copy3 = df2.copy()
# df2_copy3.dropna(axis=0, inplace=True)
# X = df2_copy3.iloc[:, [0, 1, 2, 3]]
# Y = df2_copy3.iloc[:, 4]
# X_Test = df2[mask3_df2].iloc[:, [0, 1, 2, 3]]
# # X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# print("gold price:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df2[df2['gold price'].isna()].index
# j = 0
# for i in arr:
#     df2.at[i, 'gold price'] = preds[j]
#     j += 1
# print(df2.isnull().sum())
#
# df3 = pd.read_csv("Es.csv")
# mask1_df3 = df3["E1"].isnull()
# mask2_df3 = df3["E2"].isnull()
# mask3_df3 = df3["E3"].isnull()
# mask4_df3 = df3["E4"].isnull()
# mask6_df3 = df3["E6"].isnull()
# print(df3.isnull().sum())
# df3_copy1 = df3.copy()
# df3_copy1.dropna(axis=0, inplace=True)
# # For finding missing values of "E1"
# X = df3_copy1.iloc[:, [1, 2, 3, 4, 5]]
# Y = df3_copy1.iloc[:, 0]
# X_Test = df3[mask1_df3].iloc[:, [1, 2, 3, 4, 5]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# print("E1:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df3[df3['E1'].isna()].index
# j = 0
# for i in arr:
#     df3.at[i, 'E1'] = preds[j]
#     j += 1
# print(df3.isnull().sum())
# # For finding missing values of "E2"
# df3_copy2 = df3.copy()
# df3_copy2.dropna(axis=0, inplace=True)
# X = df3_copy2.iloc[:, [0, 2, 3, 4, 5]]
# Y = df3_copy2.iloc[:, 1]
# X_Test = df3[mask2_df3].iloc[:, [0, 2, 3, 4, 5]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# print("E2:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df3[df3['E2'].isna()].index
# j = 0
# for i in arr:
#     df3.at[i, 'E2'] = preds[j]
#     j += 1
# print(df3.isnull().sum())
# # For finding missing values of "E3"
# df3_copy3 = df3.copy()
# df3_copy3.dropna(axis=0, inplace=True)
# X = df3_copy3.iloc[:, [0, 1, 3, 4, 5]]
# Y = df3_copy3.iloc[:, 2]
# X_Test = df3[mask3_df3].iloc[:, [0, 1, 3, 4, 5]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# print("E3:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df3[df3['E3'].isna()].index
# j = 0
# for i in arr:
#     df3.at[i, 'E3'] = preds[j]
#     j += 1
# print(df3.isnull().sum())
# # For finding missing values of "E4"
# df3_copy4 = df3.copy()
# df3_copy4.dropna(axis=0, inplace=True)
# X = df3_copy4.iloc[:, [0, 1, 2, 4, 5]]
# Y = df3_copy4.iloc[:, 3]
# X_Test = df3[mask4_df3].iloc[:, [0, 1, 2, 4, 5]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# print("E4:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df3[df3['E4'].isna()].index
# j = 0
# for i in arr:
#     df3.at[i, 'E4'] = preds[j]
#     j += 1
# print(df3.isnull().sum())
# # For finding missing values of "E4"
# df3_copy5 = df3.copy()
# df3_copy5.dropna(axis=0, inplace=True)
# X = df3_copy5.iloc[:, [0, 1, 2, 3, 4]]
# Y = df3_copy5.iloc[:, 5]
# X_Test = df3[mask6_df3].iloc[:, [0, 1, 2, 3, 4]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# print("E6:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df3[df3['E6'].isna()].index
# j = 0
# for i in arr:
#     df3.at[i, 'E6'] = preds[j]
#     j += 1
# print(df3.isnull().sum())
#
# df4 = pd.read_csv("titanic_train.csv")
# mask1_df4 = df4["Age"].isnull()
# mask2_df4 = df4["Embarked"].isnull()
# print(df4.isnull().sum())
# # For finding missing values of "Age"
# df4_copy1 = df4.copy()
# df4_copy1.dropna(axis=0, inplace=True)
# X = df4_copy1.iloc[:, [1, 2, 6, 7, 8]]
# Y = df4_copy1.iloc[:, 5]
# X_Test = df4[mask1_df4].iloc[:, [1, 2, 6, 7, 8]]
# X_Test = np.nan_to_num(X_Test)
# reg = LinearRegression()
# reg.fit(X, Y)
# print("Age:", reg.predict(X_Test))
# preds = reg.predict(X_Test)
# arr = df4[df4['Age'].isna()].index
# j = 0
# for i in arr:
#     df4.at[i, 'Age'] = abs(int(preds[j]))
#     j += 1
# print(df4.isnull().sum())
#
# # For finding missing values of "Embarked"
# df4_copy1 = df4.copy()
# df4_copy1.dropna(axis=0, inplace=True)
# X = df4_copy1.iloc[:, [0, 1, 2, 5, 6, 7, 8]]
# Y = df4_copy1.iloc[:, 9]
# X_Test = df4[mask2_df4].iloc[:, [0, 1, 2, 5, 6, 7, 8]]
# classifier = RandomForestClassifier()
# classifier = classifier.fit(X, Y)
# print(classifier.predict(X_Test))
# preds = classifier.predict(X_Test)
# arr = df4[df4['Embarked'].isna()].index
# j = 0
# for i in arr:
#     df4.at[i, 'Embarked'] = preds[j]
#     j += 1
# print(df4.isnull().sum())
df_original, df, idx, y_actual, y_predicted, key = vgsales_handling('Big Mart Sale Forecast.csv')
RMSE(y_actual, y_predicted)
plot1(df_original, df, idx, key)
plot2(df_original[key].mean(), idx, df, key)


