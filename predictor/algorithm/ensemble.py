import numpy as np


class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.tensordot(predictions, self.weights, axes=((0), (0)))
        return weighted_preds

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean((preds - y)**2)
