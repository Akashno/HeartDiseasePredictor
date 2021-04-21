from django.urls import path
from main.views import index,analyse

urlpatterns = [
    path('', index, name='index'),
    path('analyse/<int:pk>', analyse, name='analyse'),


]
