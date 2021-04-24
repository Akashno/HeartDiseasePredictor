from django.urls import path
from main.views import index,analyse,train

urlpatterns = [
    path('', index, name='index'),
    path('analyse/<int:pk>', analyse, name='analyse'),
    path('train/', train, name='train'),


]
