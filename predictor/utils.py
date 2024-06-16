import requests
import pandas as pd
import os
from textblob import TextBlob
from pytrends.request import TrendReq

ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY')


def fetch_alphavantage_data(symbol, function="TIME_SERIES_DAILY"):
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}&datatype=csv"
    response = requests.get(url)

    if response.status_code == 200:
        data = pd.read_csv(response.content.decode('utf-8'))
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['trend_score'] = data['close'].pct_change().rolling(window=7).mean()  # Example trend score calculation
        return data
    else:
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


def fetch_yahoo_finance_data(symbol):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1=0&period2=9999999999&interval=1d&events=history"
    response = requests.get(url)

    if response.status_code == 200:
        data = pd.read_csv(response.content.decode('utf-8'))
        data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        data['date'] = pd.to_datetime(data['date'])
        data['trend_score'] = data['close'].pct_change().rolling(window=7).mean()  # Example trend score calculation
        return data
    else:
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


def fetch_google_data(keyword):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
    data = pytrends.interest_over_time()
    data['trend_score'] = data[keyword].rolling(window=7).mean()  # Example trend score calculation
    return data


def fetch_newsapi_data(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        for article in articles:
            sentiment = TextBlob(article['description']).sentiment.polarity
            article['sentiment_score'] = sentiment
        return articles
    else:
        return []  # Return an empty list in case of an error
