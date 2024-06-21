import numpy as np


class ExpectedValueModel:
    def __init__(self, win_prob=None, win_amount=None, loss_amount=None):
        self.win_prob = win_prob
        self.win_amount = win_amount
        self.loss_amount = loss_amount

    def train(self, data):
        pass  # If any training is required, otherwise keep it as a placeholder

    def predict(self, data):
        if self.win_prob is not None and self.win_amount is not None and self.loss_amount is not None:
            expected_value = (self.win_prob * self.win_amount) - ((1 - self.win_prob) * self.loss_amount)
            return expected_value
        else:
            return 0.0  # Default value if parameters are not set


class FixedFractionModel:
    def __init__(self, fraction):
        self.fraction = fraction

    def train(self, data):
        # No training required for fixed fraction model
        pass

    def predict(self, data):
        return self.fraction  # Always returns the fixed fraction


class KellyCriterionModel:
    def __init__(self, win_prob=None, win_loss_ratio=None):
        self.win_prob = win_prob
        self.win_loss_ratio = win_loss_ratio

    def train(self, data):
        pass  # If any training is required, otherwise keep it as a placeholder

    def predict(self, data):
        if self.win_prob is not None and self.win_loss_ratio is not None:
            kelly_fraction = self.win_prob - (1 - self.win_prob) / self.win_loss_ratio
            return kelly_fraction
        else:
            return 0.0  # Default value if parameters are not set


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
