import numpy as np


class ExpectedValueModel:
    def __init__(self):
        pass

    def train(self, data):
        # No training required for expected value model
        pass

    def predict(self, data):
        return data.mean()  # Returns the mean of the data


class FixedFractionModel:
    def __init__(self, fraction):
        self.fraction = fraction

    def train(self, data):
        # No training required for fixed fraction model
        pass

    def predict(self, data):
        return self.fraction  # Always returns the fixed fraction


class KellyCriterionModel:
    def __init__(self):
        self.win_prob = None
        self.win_loss_ratio = None

    def train(self, data):
        # Calculate the win probability and win/loss ratio
        wins = data[data > 0]
        losses = data[data <= 0]
        self.win_prob = len(wins) / len(data)
        self.win_loss_ratio = abs(wins.mean() / losses.mean())

    def predict(self, data):
        if self.win_prob is None or self.win_loss_ratio is None:
            raise ValueError("Model has not been trained yet")

        # Kelly formula: f* = win_prob - (1 - win_prob) / win_loss_ratio
        kelly_fraction = self.win_prob - (1 - self.win_prob) / self.win_loss_ratio
        return kelly_fraction


# Example usage:

# Sample data representing returns
data = np.array([0.1, -0.05, 0.2, -0.1, 0.15, -0.02])

# Expected Value Model
ev_model = ExpectedValueModel()
ev_model.train(data)
print("Expected Value Prediction:", ev_model.predict(data))

# Fixed Fraction Model
ff_model = FixedFractionModel(fraction=0.1)
ff_model.train(data)
print("Fixed Fraction Prediction:", ff_model.predict(data))

# Kelly Criterion Model
kelly_model = KellyCriterionModel()
kelly_model.train(data)
print("Kelly Criterion Prediction:", kelly_model.predict(data))
