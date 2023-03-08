import math
import numpy as np


def RMSE(y_actual, y_predicted):
    MSE = np.square(np.subtract(y_actual, y_predicted)).mean()
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error using linear regression method:")
    print(RMSE)
