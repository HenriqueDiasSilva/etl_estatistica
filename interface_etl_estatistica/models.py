# models.py
from django.db import models

class ZipFile(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
class ProcessedFile(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='processed/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
    
class CsvData(models.Model):
    name = models.CharField(max_length=255)  # Nome do arquivo CSV
    file = models.FileField(upload_to='csv_files/')  # Armazena o arquivo CSV

    def __str__(self):
        return self.name