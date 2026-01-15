from django.urls import path
from .views import predict_fraud

urlpatterns = [
    path("predict/", predict_fraud),
]
