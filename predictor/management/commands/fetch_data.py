import yfinance as yf
from django.core.management.base import BaseCommand

from predictor.algorithm.technical_models import (
    SimpleMovingAverage,
    RSI,
    MACD,
    BollingerBands
)
from predictor.models import HistoricalData


class Command(BaseCommand):
    help = 'Fetches historical stock data, calculates technical indicators, and stores in database.'

    def add_arguments(self, parser):
        parser.add_argument('symbols', nargs='+', type=str, help='Stock symbols to fetch data for')
        parser.add_argument('--period', type=str, default='1yr', help='Period of historical data (default: 1mo)')

    def handle(self, *args, **options):
        symbols = options['symbols']
        period = options['period']

        for symbol in symbols:
            try:
                # Fetch historical data using yfinance
                data = yf.download(symbol, period=period)

                # Instantiate models
                sma_model = SimpleMovingAverage(window=10)  # Adjust window size as needed
                rsi_model = RSI(window=14)  # Adjust window size as needed
                macd_model = MACD(short_window=12, long_window=26, signal_window=9)
                bb_model = BollingerBands(window=20, num_std=2)

                # Calculate technical indicators
                sma_predictions = sma_model.predict(data['Close'])  # Assuming 'Close' is used for calculations
                rsi_predictions = rsi_model.predict(data['Close'])
                macd_predictions, macd_signal = macd_model.predict(data['Close'])
                upper_band, lower_band = bb_model.predict(data['Close'])

                # Iterate through each row in the data and save to HistoricalData model
                for index, row in data.iterrows():
                    HistoricalData.objects.update_or_create(
                        symbol=symbol,
                        date=index,
                        defaults={
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'close': row['Close'],
                            'volume': row['Volume'],
                            'sma': sma_predictions[index],  # Assuming sma_predictions is a pandas Series with dates as index
                            'rsi': rsi_predictions[index],  # Assuming rsi_predictions is a pandas Series with dates as index
                            'macd': macd_predictions[index],  # Assuming macd_predictions is a pandas Series with dates as index
                            'signal_line': macd_signal[index],  # Assuming macd_signal is a pandas Series with dates as index
                            'upper_band': upper_band[index],  # Assuming upper_band is a pandas Series with dates as index
                            'lower_band': lower_band[index],  # Assuming lower_band is a pandas Series with dates as index
                        }
                    )

                self.stdout.write(self.style.SUCCESS(f'Successfully fetched and stored data for {symbol}.'))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f'Failed to fetch and calculate data for {symbol}: {str(e)}'))
