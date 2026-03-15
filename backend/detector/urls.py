from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),
    path("analyze/", views.home, name="analyze"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("history/", views.history, name="history"),
    path("analytics/", views.analytics, name="analytics"),
    path("api/predict/", views.predict_fraud),
]