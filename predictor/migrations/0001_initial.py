# Generated by Django 5.0.3 on 2024-06-17 04:21

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('predicted_price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('symbol', models.CharField(max_length=10)),
            ],
        ),
        migrations.CreateModel(
            name='CurrentTrendData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('description', models.TextField()),
                ('url', models.URLField()),
                ('published_at', models.DateTimeField()),
                ('query', models.CharField(max_length=100)),
                ('date', models.DateField()),
                ('open', models.FloatField()),
                ('high', models.FloatField()),
                ('low', models.FloatField()),
                ('close', models.FloatField()),
                ('volume', models.IntegerField()),
                ('symbol', models.CharField(max_length=10)),
                ('sentiment_score', models.FloatField(blank=True, null=True)),
            ],
            options={
                'unique_together': {('title', 'published_at')},
            },
        ),
        migrations.CreateModel(
            name='HistoricalData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('symbol', models.CharField(max_length=10)),
                ('open', models.FloatField()),
                ('high', models.FloatField()),
                ('low', models.FloatField()),
                ('close', models.FloatField()),
                ('volume', models.BigIntegerField()),
                ('trend_score', models.FloatField(blank=True, null=True)),
            ],
            options={
                'unique_together': {('date', 'symbol')},
            },
        ),
    ]
