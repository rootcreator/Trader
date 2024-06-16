from django.db import models


class HistoricalData(models.Model):
    date = models.DateField()
    symbol = models.CharField(max_length=10)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()
    trend_score = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ('date', 'symbol')


class CurrentTrendData(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    url = models.URLField()
    published_at = models.DateTimeField()
    query = models.CharField(max_length=100)
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.IntegerField()
    symbol = models.CharField(max_length=10)
    sentiment_score = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ('title', 'published_at')


class Prediction(models.Model):
    date = models.DateField()
    predicted_price = models.DecimalField(max_digits=10, decimal_places=2)
    symbol = models.CharField(max_length=10)
    # Add more fields as needed
