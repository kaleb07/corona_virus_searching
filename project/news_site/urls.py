from django.contrib import admin
from django.urls import include, path
from news_site import views

urlpatterns = [
    path('', views.home),
    path('result/', views.result),
]