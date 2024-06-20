from django.db import models


class HistoricalData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()
    sma = models.FloatField(null=True, blank=True)
    rsi = models.FloatField(null=True, blank=True)
    macd = models.FloatField(null=True, blank=True)
    signal_line = models.FloatField(null=True, blank=True)
    upper_band = models.FloatField(null=True, blank=True)
    lower_band = models.FloatField(null=True, blank=True)
    fib_level1 = models.FloatField(null=True, blank=True)
    fib_level2 = models.FloatField(null=True, blank=True)
    fib_level3 = models.FloatField(null=True, blank=True)
    trend_score = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ('symbol', 'date')

    def __str__(self):
        return f"{self.symbol} - {self.date}"


class CurrentTrendData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open = models.FloatField(null=True, blank=True)
    high = models.FloatField(null=True, blank=True)
    low = models.FloatField(null=True, blank=True)
    close = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    sma = models.FloatField(null=True, blank=True)
    rsi = models.FloatField(null=True, blank=True)
    macd = models.FloatField(null=True, blank=True)
    signal_line = models.FloatField(null=True, blank=True)
    upper_band = models.FloatField(null=True, blank=True)
    lower_band = models.FloatField(null=True, blank=True)
    fib_level1 = models.FloatField(null=True, blank=True)
    fib_level2 = models.FloatField(null=True, blank=True)
    fib_level3 = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = (('symbol', 'date'),)

    def __str__(self):
        return f"{self.symbol} - {self.date}"


class Prediction(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    predicted_value = models.DecimalField(max_digits=10, decimal_places=2)

    symbol = models.CharField(max_length=10)

    # Add more fields as needed

    class Meta:
        unique_together = ('symbol',  'predicted_value')

    def __str__(self):
        return f"{self.symbol} - {self.date}"
