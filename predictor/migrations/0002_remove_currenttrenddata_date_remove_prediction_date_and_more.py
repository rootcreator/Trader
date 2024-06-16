# Generated by Django 5.0.3 on 2024-06-16 01:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='currenttrenddata',
            name='date',
        ),
        migrations.RemoveField(
            model_name='prediction',
            name='date',
        ),
        migrations.AddField(
            model_name='prediction',
            name='close',
            field=models.FloatField(default=1),
            preserve_default=False,
        ),
    ]
