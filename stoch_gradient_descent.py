import numpy as np
from math import sqrt
import math
from sklearn.metrics import mean_squared_error

def rmse(prediction, truth):
    prediction = prediction[truth.nonzero()].flatten()
    truth = truth[truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, truth))

class StochasticGradientDescent:
    def __init__(self, n_epochs=200, n_latent_features=3, lmbda=0.1, learning_rate=0.001):
        self.n_epochs = n_epochs
        self.n_latent_features = n_latent_features
        self.lmbda = lmbda
        self.learning_rate = learning_rate

    