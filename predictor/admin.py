from django.contrib import admin
from django.core.management import call_command
from .models import HistoricalData, CurrentTrendData
from django.contrib import messages


class CustomAdminActions(admin.ModelAdmin):
    actions = ['fetch_predictor_data']

    def fetch_predictor_data(self, request, queryset):
        for obj in queryset:
            symbol = obj.symbol
            try:
                call_command('fetch_data.py', symbol=symbol)
                self.message_user(request, f'Successfully fetched predictor data for {symbol}', messages.SUCCESS)
            except Exception as e:
                self.message_user(request, f'Error fetching predictor data for {symbol}: {e}', messages.ERROR)

    fetch_predictor_data.short_description = 'Fetch predictor data for selected symbols'


# Register your models and custom admin actions
admin.site.register(HistoricalData, CustomAdminActions)
admin.site.register(CurrentTrendData, CustomAdminActions)
