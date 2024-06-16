from django.core.management.base import BaseCommand
from predictor.models import HistoricalData
from predictor.utils import fetch_yahoo_finance_data


class Command(BaseCommand):
    help = 'Fetch predictor from Yahoo Finance'

    def add_arguments(self, parser):
        parser.add_argument('symbol', type=str, help='Stock symbol to fetch predictor for')

    def handle(self, *args, **options):
        symbol = options['symbol']

        data = fetch_yahoo_finance_data(symbol)

        if not data.empty:  # Ensure predictor is not empty
            for index, row in data.iterrows():
                date = row['date']
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                volume = row['volume']
                trend_score = row['trend_score']

                HistoricalData.objects.update_or_create(
                    date=date, symbol=symbol,
                    defaults={
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'trend_score': trend_score
                    }
                )
            self.stdout.write(self.style.SUCCESS(f'Successfully fetched and saved predictor for {symbol}'))
        else:
            self.stdout.write(self.style.ERROR(f'Failed to fetch predictor for {symbol}'))
