import abc

class BaseModel(abc.ABC):
    @abc.abstractmethod
    def train(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def evaluate(self, X, y):
        pass
