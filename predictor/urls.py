from django.urls import path
from .views import predict_view

urlpatterns = [
    path('', predict_view, name='predict'),  # Root URL
    path('predict/', predict_view, name='predict'),  # URL for '/predict/'
]
