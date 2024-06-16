import numpy as np


class MeanReversionModel:
    def __init__(self):
        pass

    def train(self, data):
        # Training can include calculating long-term mean, thresholds, etc.
        self.long_term_mean = data.mean()

    def predict(self, data):
        return self.long_term_mean


class CarryTradeModel:
    def __init__(self):
        self.interest_rate_diff = None

    def train(self, data):
        # Example data could include pairs of (domestic_rate, foreign_rate)
        domestic_rates = data[:, 0]
        foreign_rates = data[:, 1]
        self.interest_rate_diff = domestic_rates.mean() - foreign_rates.mean()

    def predict(self, data):
        if self.interest_rate_diff is None:
            raise ValueError("Model has not been trained yet")
        return self.interest_rate_diff


class VolatilityModel:
    def __init__(self):
        self.volatility = None

    def train(self, data):
        # Calculate the volatility (e.g., standard deviation of returns)
        self.volatility = np.std(data)

    def predict(self, data):
        if self.volatility is None:
            raise ValueError("Model has not been trained yet")
        return self.volatility


# Example usage:

# Sample data for Mean Reversion Model
mean_reversion_data = np.array([1, 2, 3, 2, 1, 2, 3])

# Mean Reversion Model
mr_model = MeanReversionModel()
mr_model.train(mean_reversion_data)
print("Mean Reversion Prediction:", mr_model.predict(mean_reversion_data))

# Sample data for Carry Trade Model (domestic_rate, foreign_rate)
carry_trade_data = np.array([[0.02, 0.01], [0.025, 0.015], [0.03, 0.02]])

# Carry Trade Model
ct_model = CarryTradeModel()
ct_model.train(carry_trade_data)
print("Carry Trade Prediction:", ct_model.predict(carry_trade_data))

# Sample data for Volatility Model
volatility_data = np.array([0.1, 0.2, -0.05, 0.3, -0.1])

# Volatility Model
vol_model = VolatilityModel()
vol_model.train(volatility_data)
print("Volatility Prediction:", vol_model.predict(volatility_data))
