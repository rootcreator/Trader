import keras
import numpy as np
from _testcapi import sequence_length
from keras import Model

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA  # For ARIMA model

from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense


class BaseModel:
    def train(self, X, y):
        raise NotImplementedError("Train method not implemented")

    def predict(self, X):
        raise NotImplementedError("Predict method not implemented")

    def evaluate(self, X, y):
        raise NotImplementedError("Evaluate method not implemented")


class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)


class DecisionTreeModel(BaseModel):
    def __init__(self, max_depth=None):
        self.model = DecisionTreeRegressor(max_depth=max_depth)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)


class SVMModel(BaseModel):
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVR(kernel=kernel, C=C)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)


# Assuming input X has shape (batch_size, 100, 1)
input_shape = (100, 1)


class ARIMAModel(BaseModel):
    def __init__(self, order=(1, 0, 0)):
        self.order = order
        self.model = None

    def train(self, X, y=None):
        self.model = ARIMA(X, order=self.order)
        self.model = self.model.fit()

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.forecast(steps=len(X))[0]

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)


# Example usage:

# Sample data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2.0 * X.squeeze() + np.random.randn(100)  # Linear relationship with noise

# Instantiate models
lr_model = LinearRegressionModel()
dt_model = DecisionTreeModel(max_depth=5)
rf_model = RandomForestModel(n_estimators=100)
svm_model = SVMModel(kernel='linear', C=1.0)
arima_model = ARIMAModel(order=(1, 0, 0))

# Train models
lr_model.train(X, y)
dt_model.train(X, y)
rf_model.train(X, y)
svm_model.train(X, y)
arima_model.train(y)

# Evaluate models
lr_loss = lr_model.evaluate(X, y)
dt_loss = dt_model.evaluate(X, y)
rf_loss = rf_model.evaluate(X, y)
svm_loss = svm_model.evaluate(X, y)
arima_loss = arima_model.evaluate(y, y)  # ARIMA evaluates differently

# Print example losses
print("Linear Regression Loss:", lr_loss)
print("Decision Tree Loss:", dt_loss)
print("Random Forest Loss:", rf_loss)
print("SVM Loss:", svm_loss)
print("ARIMA Loss:", arima_loss)
