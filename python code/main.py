import warnings
from datasets_handling import *
from empirical_experiments_utils import RMSE, compare
from plotting_utils import plot1, plot2, org_distribution_graph, after_distribution_graph
warnings.filterwarnings('ignore')
# dataset 1
# df_original, df, idx, y_actual, y_predicted, key = vgsales_handling(r'C:\Users\ahmad\PycharmProjects\tds-new\datasets\vgsales.csv')
# RMSE(y_actual, y_predicted, 'Linear Regression')
# plot1(df_original, df, idx, key)
# RMSE(y_actual, [df_original[key].mean()] * len(y_actual), 'Mean')
# plot2(df_original[key].mean(), idx, df, key)

# # dataset 2
# df_original, df, idx, y_actual, y_predicted, key = StudentsPerformance_handling(r'C:\Users\ahmad\PycharmProjects\tds-new\datasets\StudentsPerformance.csv')
# RMSE(y_actual, y_predicted, 'Linear Regression')
# plot1(df_original, df, idx, key)
# RMSE(y_actual, [df_original[key].mean()] * len(y_actual), 'Mean')
# plot2(df_original[key].mean(), idx, df, key)
#
# # dataset 3
# df_plot_org, df_plot_after, y_actual, y_predicted, most_freq = Mart_Sale_Forecast_handling(r'C:\Users\ahmad\PycharmProjects\tds-new\datasets\Big Mart Sale Forecast.csv')
# compare(y_actual, y_predicted, 'Random Forest')
# org_distribution_graph(df_plot_org, 'Outlet_Location_Type')
# compare(y_actual, [most_freq] * len(y_actual), 'Most Frequent')
# after_distribution_graph(df_plot_after, 'Outlet_Location_Type')
#
# # dataset 4
df_plot_org, df_plot_after, y_actual, y_predicted, most_freq = titanic_handling(r'C:\Users\ahmad\PycharmProjects\tds-new\datasets\titanic.csv')
RMSE(y_actual, y_predicted, 'Random Forest')
org_distribution_graph(df_plot_org, 'Embarked')
RMSE(y_actual, [most_freq] * len(y_actual), 'Most Frequent')
after_distribution_graph(df_plot_after, 'Embarked')

















