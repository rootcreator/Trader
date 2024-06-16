from django.shortcuts import render
from django.http import JsonResponse
from predictor.algorithm.ensemble import EnsembleModel
from predictor.algorithm.forex_models import MeanReversionModel
from predictor.algorithm.risk_management_models import ExpectedValueModel
from predictor.algorithm.technical_models import SimpleMovingAverage
from predictor.algorithm.machine_learning_models import LinearRegressionModel
from predictor.models import HistoricalData, CurrentTrendData, Prediction
import pandas as pd


def index(request):
    """Render the main page."""
    return render(request, 'index.html')


def get_historical_data(symbol):
    """Fetch historical data for a given symbol from the database."""
    data = HistoricalData.objects.filter(symbol=symbol).values('date', 'close')
    return pd.DataFrame(list(data))


def get_current_trend_data(symbol):
    """Fetch current trend data for a given symbol from the database."""
    data = CurrentTrendData.objects.filter(symbol=symbol).values('date', 'close')
    return pd.DataFrame(list(data))


def predict(request):
    """Handle prediction requests."""
    try:
        symbol = request.GET.get('symbol')  # Get symbol from request
        if not symbol:
            return JsonResponse({'error': 'Symbol parameter is required.'}, status=400)

        historical_data = get_historical_data(symbol)
        current_trends = get_current_trend_data(symbol)

        if historical_data.empty or current_trends.empty:
            return JsonResponse({'error': 'No data available for the provided symbol.'}, status=404)

        # Initialize models
        sma_model = SimpleMovingAverage(window=3)
        lr_model = LinearRegressionModel()
        ev_model = ExpectedValueModel()
        mr_model = MeanReversionModel()

        # Train models
        sma_model.train(historical_data['close'])
        lr_model.train(historical_data[['close']], historical_data['close'])
        ev_model.train(historical_data['close'])
        mr_model.train(historical_data['close'])

        # Create ensemble model
        ensemble_model = EnsembleModel(models=[sma_model, lr_model, ev_model, mr_model])

        # Predict using ensemble model
        prediction = ensemble_model.predict(current_trends[['close']])

        # Save the prediction to the database
        Prediction.objects.create(
            date=pd.Timestamp.now(),
            predicted_price=prediction[0],
            symbol=symbol
        )

        return JsonResponse({'prediction': prediction.tolist()})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)