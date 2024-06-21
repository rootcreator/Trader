from django.core.management.base import BaseCommand, CommandError
from django.core.management import call_command


class Command(BaseCommand):
    help = 'Fetches both historical and current stock data, calculates technical indicators, and stores in database.'

    def add_arguments(self, parser):
        parser.add_argument('symbols', nargs='+', type=str, help='Stock symbols to fetch data for')
        parser.add_argument('--period', type=str, default='1mo', help='Period of historical data (default: 1mo)')

    def handle(self, *args, **options):
        symbols = options['symbols']
        period = options['period']

        try:
            # Call the fetch_data command
            call_command('fetch_data', *symbols, period=period)
            self.stdout.write(self.style.SUCCESS('Successfully executed fetch_data'))

            # Call the fetch_current command
            call_command('fetch_current', *symbols, period=period)
            self.stdout.write(self.style.SUCCESS('Successfully executed fetch_current'))

        except CommandError as e:
            self.stderr.write(self.style.ERROR(f'Error: {e}'))
