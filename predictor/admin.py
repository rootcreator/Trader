from django.contrib import admin
from predictor.models import HistoricalData, CurrentTrendData

admin.site.register(HistoricalData)
admin.site.register(CurrentTrendData)
