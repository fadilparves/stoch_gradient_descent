import numpy as np
from math import sqrt
import math
from sklearn.metrics import mean_squared_error

def rmse(prediction, truth):
    prediction = prediction[truth.nonzero()].flatten()
    truth = truth[truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, truth))