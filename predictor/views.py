# Django imports
from django.shortcuts import render
from django.http import JsonResponse
from predictor.models import HistoricalData, CurrentTrendData, Prediction

# Data processing and machine learning imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom model imports
from predictor.algorithm.ensemble import ModelEnsemble
from predictor.algorithm.technical_models import (
    BollingerBands,
    MACD,
    RSI,
    SimpleMovingAverage
)
from predictor.algorithm.forex_models import (
    CarryTradeModel,
    VolatilityModel,
    MeanReversionModel
)
from predictor.algorithm.risk_management_models import (
    KellyCriterionModel,
    FixedFractionModel,
    ExpectedValueModel
)


def index(request):
    """Render the main page."""
    return render(request, 'index.html')


def fetch_historical_data(symbol):
    """Fetch historical data for a given symbol from the database."""
    data = HistoricalData.objects.filter(symbol=symbol).values('date', 'close')
    return pd.DataFrame(list(data))


def fetch_current_trend_data(symbol):
    """Fetch current trend data for a given symbol from the database."""
    data = CurrentTrendData.objects.filter(symbol=symbol).values('date', 'close')
    return pd.DataFrame(list(data))


def predict(request):
    """Handle prediction requests."""
    try:
        symbol = request.GET.get('symbol')  # Get symbol from request
        if not symbol:
            return JsonResponse({'error': 'Symbol parameter is required.'}, status=400)

        historical_data = fetch_historical_data(symbol)
        current_trend_data = fetch_current_trend_data(symbol)

        if historical_data.empty or current_trend_data.empty:
            return JsonResponse({'error': 'No data available for the provided symbol.'}, status=404)

        # Merge historical and current trend data
        X = pd.concat([historical_data.set_index('date'), current_trend_data.set_index('date')], axis=1, join='inner').dropna().reset_index()
        y = X.pop('close')

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize technical models
        bb_model = BollingerBands(window=20)
        macd_model = MACD(short_window=12, long_window=26, signal_window=9)
        rsi_model = RSI(window=14)
        sma_model = SimpleMovingAverage(window=3)

        # Initialize forex models
        carry_trade_model = CarryTradeModel()
        volatility_model = VolatilityModel()
        mean_reversion_model = MeanReversionModel()

        # Initialize risk management models
        kelly_criterion_model = KellyCriterionModel()
        fixed_fraction_model = FixedFractionModel()
        expected_value_model = ExpectedValueModel()

        # Train technical models
        bb_model.train(X_train['close'])
        macd_model.train(X_train['close'])
        rsi_model.train(X_train['close'])
        sma_model.train(X_train['close'])

        # Train forex models
        carry_trade_model.train(X_train['close'])
        volatility_model.train(X_train['close'])
        mean_reversion_model.train(X_train['close'])

        # Train risk management models
        kelly_criterion_model.train(X_train['close'])
        fixed_fraction_model.train(X_train['close'])
        expected_value_model.train(X_train['close'])

        # Define base models
        base_models = [
            bb_model, macd_model, rsi_model, sma_model,
            carry_trade_model, volatility_model, mean_reversion_model,
            kelly_criterion_model, fixed_fraction_model, expected_value_model
        ]

        # Initialize and train the ensemble model
        ensemble_model = ModelEnsemble(models=base_models)
        ensemble_model.train(X_train_scaled, y_train)

        # Evaluate the ensemble model
        mse = ensemble_model.evaluate(X_test_scaled, y_test)
        print(f"Ensemble MSE: {mse}")

        # Predict using the ensemble model
        predictions = ensemble_model.predict(X_test_scaled)
        results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        print(results)

        # Save the prediction to the database
        Prediction.objects.create(
            date=pd.Timestamp.now(),
            predicted_price=predictions[0],  # Assuming the first prediction is needed
            symbol=symbol
        )

        return JsonResponse({'prediction': predictions.tolist()})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
