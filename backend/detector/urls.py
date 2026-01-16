from django.urls import path
from .views import predict_fraud, home

urlpatterns = [
    path("", home),
    path("predict/", predict_fraud),
]
