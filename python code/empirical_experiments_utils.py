import math
import numpy as np


def RMSE(y_actual, y_predicted, key):
    MSE = np.square(np.subtract(y_actual, y_predicted)).mean()
    RMSE = math.sqrt(MSE)
    print(f"Root Mean Square Error using {key} method:")
    print(RMSE)
