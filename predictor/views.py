# predictor/views.py

from django.shortcuts import render
from django.http import JsonResponse
from predictor.models import HistoricalData, CurrentTrendData, Prediction
from predictor.algorithm.ensemble import ModelEnsemble
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
from predictor.algorithm.technical_models import (
    BollingerBands,
    MACD,
    RSI,
    SimpleMovingAverage
)
import pandas as pd
import numpy as np

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
            # Add more fields as needed
        })

        # Instantiate and train Simple Moving Average model
        sma_model = SimpleMovingAverage(window=10)
        X_historical_close = X_historical['close']  # Assuming 'close' is the column name for closing prices
        sma_model.train(X_historical_close)

        # Instantiate FixedFractionModel with required argument 'fraction'
        fixed_fraction_model = FixedFractionModel(fraction=0.1)  # Replace with your desired fraction value

        # Combine various models into an ensemble
        models = [
            CarryTradeModel(),
            VolatilityModel(),
            MeanReversionModel(),
            BollingerBands(window=20, num_std=2),
            MACD(short_window=12, long_window=26, signal_window=9),
            RSI(window=14),
            KellyCriterionModel(),
            fixed_fraction_model,
            ExpectedValueModel(),
            sma_model
        ]

        # Initialize ModelEnsemble with models
        ensemble = ModelEnsemble(models)

        # Make predictions using ensemble model
        predictions = ensemble.predict(X_historical)

        # Handle NaN values (replace with a default value or remove NaNs)
        predictions_clean = np.nan_to_num(predictions, nan=0.0)

        # Calculate average prediction if predictions_clean is an array
        if isinstance(predictions_clean, np.ndarray):
            average_prediction = np.mean(predictions_clean)
        else:
            average_prediction = predictions_clean

        # Save predictions to Prediction model
        try:
            # Check if a similar prediction already exists for the symbol and value
            existing_prediction = Prediction.objects.filter(symbol=symbol, predicted_value=average_prediction).first()
            if existing_prediction:
                # Handle case where prediction already exists (update or skip)
                pass
            else:
                prediction_obj = Prediction(symbol=symbol, predicted_value=average_prediction)
                prediction_obj.save()

            return render(request, 'predictor/predict_result.html', {'predictions': average_prediction})

        except ValueError as e:
            return JsonResponse({'error': f'Error saving prediction: {str(e)}'})

    # Handle GET request (initial loading of the form)
    return render(request, 'predictor/predict_form.html')
