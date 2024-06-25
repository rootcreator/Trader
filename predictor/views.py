from django.shortcuts import render
from django.http import JsonResponse
from predictor.models import HistoricalData, CurrentTrendData, Prediction
from predictor.algorithm.technical_models import SimpleMovingAverage, RSI, MACD, BollingerBands
from predictor.algorithm.machine_learning_models import LinearRegressionModel, DecisionTreeModel, RandomForestModel, \
    SVMModel, ARIMAModel
from predictor.algorithm.risk_management_models import FixedFractionModel, KellyCriterionModel, ExpectedValueModel
from predictor.algorithm.forex_models import MeanReversionModel, CarryTradeModel, VolatilityModel
from predictor.algorithm.ensemble import ModelEnsemble
import pandas as pd
import numpy as np

# Importing necessary scikit-learn models for AdaBoost
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import requests
from datetime import datetime

ALPHA_VANTAGE_API_KEY = '9F2ZXE5KLGDWTG7C'


def fetch_latest_market_data(symbol):
    api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        time_series = data.get("Time Series (1min)", {})
        if not time_series:
            return None

        latest_time = max(time_series.keys())
        latest_data = time_series[latest_time]
        parsed_data = {
            'open': float(latest_data['1. open']),
            'close': float(latest_data['4. close']),
            'high': float(latest_data['2. high']),
            'volume': float(latest_data['5. volume']),
            'date': datetime.strptime(latest_time, '%Y-%m-%d %H:%M:%S')
        }
        return parsed_data
    else:
        return None


def predict_view(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol')

        # Fetch historical data from database
        historical_data = HistoricalData.objects.filter(symbol=symbol).order_by('date')

        if not historical_data.exists():
            return JsonResponse({'error': f'No historical data found for symbol {symbol}'})

        # Fetch current trend data from database
        current_trend_data = CurrentTrendData.objects.latest('date')

        # Prepare data for prediction
        X_historical = pd.DataFrame(list(historical_data.values()))
        X_current = pd.DataFrame({
            'open': [current_trend_data.open],
            'close': [current_trend_data.close],
            'high': [current_trend_data.high],
            'volume': [current_trend_data.volume],
        })

        # Fetch the latest real-world market data
        latest_data = fetch_latest_market_data(symbol)
        if latest_data:
            X_latest = pd.DataFrame([latest_data])

            # Combine the historical and latest data
            X_combined = pd.concat([X_historical, X_latest], ignore_index=True)
        else:
            X_combined = X_historical

        # Separate features and target for ML models
        X_features = X_combined[['open', 'high', 'volume']]
        y_target = X_combined['close']

        # Instantiate and train models
        sma_model = SimpleMovingAverage(window=10)
        sma_model.train(X_combined['close'])

        rsi_model = RSI(window=14)
        rsi_model.train(X_combined['close'])

        macd_model = MACD(short_window=12, long_window=26, signal_window=9)
        macd_model.train(X_combined['close'])

        bb_model = BollingerBands(window=20, num_std=2)
        bb_model.train(X_combined['close'])

        # Use scikit-learn models for ensemble
        lr_model = LinearRegression()
        dt_model = DecisionTreeRegressor()
        rf_model = RandomForestRegressor()
        svm_model = SVR()

        lr_model.fit(X_features, y_target)
        dt_model.fit(X_features, y_target)
        rf_model.fit(X_features, y_target)
        svm_model.fit(X_features, y_target)

        arima_model = ARIMAModel(order=(5, 1, 0))
        arima_model.train(X_combined['close'])

        fixed_fraction_model = FixedFractionModel(fraction=0.1)
        kelly_model = KellyCriterionModel(win_prob=0.6, win_loss_ratio=2)  # Example values
        ev_model = ExpectedValueModel(win_prob=0.6, win_amount=100, loss_amount=50)  # Example values

        mean_reversion_model = MeanReversionModel()
        mean_reversion_model.train(X_combined['close'])

        volatility_model = VolatilityModel()
        volatility_model.train(X_combined['close'])

        carry_trade_model = CarryTradeModel()  # Implement training if necessary

        # Combine models into an ensemble
        models = [lr_model, dt_model, rf_model, svm_model]
        ensemble = ModelEnsemble(models)

        # Train the ensemble model
        ensemble.train(X_features, y_target)

        # Make predictions using ensemble model
        X_current_features = X_current[['open', 'high', 'volume']]
        predictions = ensemble.predict(X_current_features)

        # Handle NaN values
        predictions_clean = np.nan_to_num(predictions, nan=0.0)

        # Calculate average prediction if predictions_clean is an array
        average_prediction = predictions_clean if isinstance(predictions_clean, (float, int)) else np.mean(
            predictions_clean)

        # Save predictions to Prediction model
        try:
            existing_prediction = Prediction.objects.filter(symbol=symbol, predicted_value=average_prediction).first()
            if not existing_prediction:
                prediction_obj = Prediction(symbol=symbol, predicted_value=average_prediction)
                prediction_obj.save()

            return render(request, 'predictor/predict_result.html', {'predictions': average_prediction})

        except ValueError as e:
            return JsonResponse({'error': f'Error saving prediction: {str(e)}'})

    # Handle GET request (initial loading of the form)
    return render(request, 'predictor/predict_form.html')
