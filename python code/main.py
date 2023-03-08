import warnings
from datasets_handling import vgsales_handling
from empirical_experiments_utils import RMSE
from plotting_utils import plot1, plot2
warnings.filterwarnings('ignore')
# dataset 1
df_original, df, idx, y_actual, y_predicted, key = vgsales_handling('')
RMSE(y_actual, y_predicted, 'Linear Regression')
plot1(df_original, df, idx, key)
RMSE(y_actual, [df_original[key].mean()] * len(y_actual), 'Mean')
plot2(df_original[key].mean(), idx, df, key)

# dataset 2
df_original, df, idx, y_actual, y_predicted, key = vgsales_handling('')
RMSE(y_actual, y_predicted, 'Linear Regression')
plot1(df_original, df, idx, key)
RMSE(y_actual, [df_original[key].mean()] * len(y_actual), 'Mean')
plot2(df_original[key].mean(), idx, df, key)

# dataset 3




