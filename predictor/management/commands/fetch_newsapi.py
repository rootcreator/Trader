from django.core.management.base import BaseCommand
from predictor.models import CurrentTrendData
from predictor.utils import fetch_newsapi_data


class Command(BaseCommand):
    help = 'Fetch news predictor from NewsAPI'

    def add_arguments(self, parser):
        parser.add_argument('query', type=str, help='Query to fetch news for')

    def handle(self, *args, **options):
        query = options['query']

        articles = fetch_newsapi_data(query)

        if articles:  # Ensure there are articles
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
            self.stdout.write(self.style.SUCCESS(f'Successfully fetched and saved news for {query}'))
        else:
            self.stdout.write(self.style.ERROR(f'Failed to fetch news for {query}'))
