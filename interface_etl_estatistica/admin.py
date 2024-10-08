# admin.py
from django.contrib import admin
from .models import ZipFile, ProcessedFile

admin.site.register(ZipFile)
admin.site.register(ProcessedFile)