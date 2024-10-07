from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('sobre/', views.about, name='about'),
    path('combine_csv/', views.combine_csv_files, name='combine_csv_files'),
]