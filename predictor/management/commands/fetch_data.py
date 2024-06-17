from django.core.management.base import BaseCommand
from predictor.models import HistoricalData, CurrentTrendData
from predictor.utils import fetch_alphavantage_data, fetch_google_data, fetch_newsapi_data, fetch_yahoo_finance_data


class Command(BaseCommand):
    help = 'Fetch predictor from multiple sources'

    def add_arguments(self, parser):
        parser.add_argument('symbol', type=str, help='Stock symbol to fetch predictor for')

    def handle(self, *args, **options):
        symbol = options['symbol']
        self.fetch_alphavantage_data(symbol)
        self.fetch_google_news(symbol)
        self.fetch_newsapi_news(symbol)
        self.fetch_yahoo_finance_data(symbol)

    def fetch_alphavantage_data(self, symbol):
        FUNCTION = 'TIME_SERIES_DAILY'
        data = fetch_alphavantage_data(symbol, FUNCTION)
        if not data.empty:
            for index, row in data.iterrows():
                date = row['timestamp']
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
            self.stdout.write(self.style.SUCCESS(f'Successfully fetched and saved predictor for {symbol} from '
                                                 f'Alphavantage'))
        else:
            self.stdout.write(self.style.ERROR(f'Failed to fetch predictor for {symbol} from Alphavantage'))

    def fetch_google_news(self, query):
        articles = fetch_google_data(query)
        if articles:
            for article in articles:
                title = article['title']
                description = article['description']
                url = article['url']
                published_at = article['publishedAt']
                sentiment_score = article['sentiment_score']

                CurrentTrendData.objects.update_or_create(
                    title=title,
                    defaults={
                        'description': description,
                        'url': url,
                        'published_at': published_at,
                        'query': query,
                        'sentiment_score': sentiment_score
                    }
                )
            self.stdout.write(self.style.SUCCESS(f'Successfully fetched and saved news for {query} from Google'))
        else:
            self.stdout.write(self.style.ERROR(f'Failed to fetch news for {query} from Google'))

    def fetch_newsapi_news(self, query):
        articles = fetch_newsapi_data(query)
        if articles:
            for article in articles:
                title = article['title']
                description = article['description']
                url = article['url']
                published_at = article['publishedAt']
                sentiment_score = article['sentiment_score']

                CurrentTrendData.objects.update_or_create(
                    title=title,
                    defaults={
                        'description': description,
                        'url': url,
                        'published_at': published_at,
                        'query': query,
                        'sentiment_score': sentiment_score
                    }
                )
            self.stdout.write(self.style.SUCCESS(f'Successfully fetched and saved news for {query} from NewsAPI'))
        else:
            self.stdout.write(self.style.ERROR(f'Failed to fetch news for {query} from NewsAPI'))

    def fetch_yahoo_finance_data(self, symbol):
        data = fetch_yahoo_finance_data(symbol)
        if not data.empty:
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
            self.stdout.write(
                self.style.SUCCESS(f'Successfully fetched and saved predictor for {symbol} from Yahoo Finance'))
        else:
            self.stdout.write(self.style.ERROR(f'Failed to fetch predictor for {symbol} from Yahoo Finance'))
