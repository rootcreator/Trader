from django.contrib import admin
from django.core.management import call_command
from django.contrib import messages
from import_export.admin import ImportExportModelAdmin
from import_export.resources import ModelResource
from .models import HistoricalData, CurrentTrendData, Prediction


# Define resource classes for each model
class HistoricalDataResource(ModelResource):
    class Meta:
        model = HistoricalData
        fields = ['symbol', 'date', 'open', 'close', 'high', 'low']  # Specify the fields to export


class CurrentTrendDataResource(ModelResource):
    class Meta:
        model = CurrentTrendData
        fields = ['symbol', 'date', 'open', 'close', 'high', 'low']  # Specify the fields to export


class PredictionResource(ModelResource):
    class Meta:
        model = Prediction
        fields = ['symbol', 'date', 'open', 'close', 'high', 'low']  # Specify the fields to export


# Define a custom admin action mixin
class CustomAdminActions(admin.ModelAdmin):
    actions = ['fetch_predictor_data']

    def fetch_predictor_data(self, request, queryset):
        for obj in queryset:
            symbol = obj.symbol
            try:
                call_command('fetch_data', symbol=symbol)
                self.message_user(request, f'Successfully fetched predictor data for {symbol}', messages.SUCCESS)
            except Exception as e:
                self.message_user(request, f'Error fetching predictor data for {symbol}: {e}', messages.ERROR)

    fetch_predictor_data.short_description = 'Fetch predictor data for selected symbols'


# Combine custom actions and import-export functionality
class HistoricalDataAdmin(CustomAdminActions, ImportExportModelAdmin):
    resource_class = HistoricalDataResource


class CurrentTrendDataAdmin(CustomAdminActions, ImportExportModelAdmin):
    resource_class = CurrentTrendDataResource


class PredictionAdmin(ImportExportModelAdmin):
    resource_class = PredictionResource


# Register the models with custom admin actions and import-export functionality
admin.site.register(HistoricalData, HistoricalDataAdmin)
admin.site.register(CurrentTrendData, CurrentTrendDataAdmin)
admin.site.register(Prediction, PredictionAdmin)
