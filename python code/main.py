import warnings
import pandas as pd
from datasets_handling import *
from empirical_experiments_utils import RMSE, compare
from plotting_utils import plot1, plot2, org_distribution_graph, after_distribution_graph, corr_plot
warnings.filterwarnings('ignore')
FORECAST_DF = r"C:\Users\ahmad\PycharmProjects\tds-new\datasets\Big Mart Sale Forecast.csv"
TITANIC_DF = r"C:\Users\ahmad\PycharmProjects\tds-new\datasets\titanic.csv"
STUDENTS_DF = r"C:\Users\ahmad\PycharmProjects\tds-new\datasets\StudentsPerformance.csv"
VGSALES_DF = r"C:\Users\ahmad\PycharmProjects\tds-new\datasets\vgsales.csv"
# dataset 1
df_original, df, idx, y_actual, y_predicted, key = vgsales_handling(VGSALES_DF)
corr_plot(VGSALES_DF, 500, 'Other_Sales')
RMSE(y_actual, y_predicted, 'Linear Regression')
plot1(df_original, df, idx, key)
RMSE(y_actual, [df_original[key].mean()] * len(y_actual), 'Mean')
plot2(df_original[key].mean(), idx, df, key)

# dataset 2
df_original, df, idx, y_actual, y_predicted, key = StudentsPerformance_handling(STUDENTS_DF)
corr_plot(STUDENTS_DF, 700, 'reading score')
RMSE(y_actual, y_predicted, 'Linear Regression')
plot1(df_original, df, idx, key)
RMSE(y_actual, [df_original[key].mean()] * len(y_actual), 'Mean')
plot2(df_original[key].mean(), idx, df, key)

# dataset 3
df_plot_org, df_plot_after, y_actual, y_predicted, most_freq = Mart_Sale_Forecast_handling(FORECAST_DF)
compare(y_actual, y_predicted, 'Random Forest')
org_distribution_graph(df_plot_org, 'Outlet_Location_Type')
compare(y_actual, [most_freq] * len(y_actual), 'Most Frequent')
after_distribution_graph(df_plot_after, 'Outlet_Location_Type')

# dataset 4
df_plot_org, df_plot_after, y_actual, y_predicted, most_freq = titanic_handling(TITANIC_DF)
compare(y_actual, y_predicted, 'Random Forest')
org_distribution_graph(df_plot_org, 'Embarked')
compare(y_actual, [most_freq] * len(y_actual), 'Most Frequent')
after_distribution_graph(df_plot_after, 'Embarked')

















