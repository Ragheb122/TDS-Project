import math
import numpy as np


def RMSE(y_actual, y_predicted, key):
    MSE = np.square(np.subtract(y_actual, y_predicted)).mean()
    RMSE = math.sqrt(MSE)
    print(f"Root Mean Square Error using {key} method:")
    print(RMSE)

def compare(y_actual, y_predicted, key):
    sum = 0
    for i in range(len(y_actual)):
        if y_predicted[i] == y_actual[i]:
            sum += 1
    error_rate = 1 - (sum / len(y_actual))
    print(f"Error rate using {key}:")
    print(error_rate)
