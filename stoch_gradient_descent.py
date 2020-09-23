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

    def predictions(self, P, Q):
        return np.dot(P.T, Q)

    def fit(self, X_train, X_val):
        m, n  = X_train.shape

        self.P = 3 * np.random.rand(self.n_latent_features, m)
        self.Q = 3 * np.random.rand(self.n_latent_features, n)

        self.train_error = []
        self.val_error = []

        users, items = X_train.nonzero()

        for epoch in range(self.n_epochs):
            for u, i in zip(users, items):
                error = X_train[u, i] - self.predictions(self.P[:,u], self.Q[:,i])
                self.P[:,u] += self.learning_rate * (error * self.Q[:, i] - self.lmbda * self.P[:, u])
                self.Q[:,i] += self.learning_rate * (error * self.P[:, u] - self.lmbda * self.Q[:, i])

            train_rmse = rmse(self.predictions(self.P, self.Q), X_train)
            val_rmse = rmse(self.predictions(self.P, self.Q), X_val)
            self.train_error.append(train_rmse)
            self.val_error.append(val_rmse)
        
        return self

    def predict(self, X_train, user_index):
        y_hat = self.predictions(self.P, self.Q)
        pred_index = np.where(X_train[user_index, :] == 0)[0]
        return y_hat[user_index, pred_index].flatten()