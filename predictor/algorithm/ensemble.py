import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


class ModelEnsemble:
    def __init__(self, base_models=None, n_estimators=50, learning_rate=1.0):
        self.base_models = base_models if base_models else []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.boosted_models = []

    def add_base_model(self, model):
        self.base_models.append(model)

    def train(self, x_train, y_train):
        for base_model in self.base_models:
            boosted_model = AdaBoostRegressor(base_estimator=base_model,
                                              n_estimators=self.n_estimators,
                                              learning_rate=self.learning_rate)
            boosted_model.fit(x_train, y_train)
            self.boosted_models.append(boosted_model)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for boosted_model in self.boosted_models:
            predictions += boosted_model.predict(X)
        return predictions / len(self.boosted_models)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return mean_squared_error(y, predictions)

    def get_models(self):
        return self.boosted_models
