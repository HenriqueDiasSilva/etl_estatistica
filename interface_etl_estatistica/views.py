from django.shortcuts import render
import pandas as pd
from django.http import HttpResponse
from .models import CSVFile

def home(request):
    if request.method == 'POST':
        files = request.FILES.getlist('file')
        for file in files:
            csv_file = CSVFile(name=file.name, file=file)
            csv_file.save()

        return HttpResponse("Arquivos CSV enviados e armazenados com sucesso!")
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def combine_csv_files(request):
    csv_files = CSVFile.objects.order_by('-uploaded_at')[:2]
    
    if csv_files.count() < 2:
        return HttpResponse("É necessário ter pelo menos dois arquivos para combinar.")

    dataframes = []
    for csv_file in csv_files:
        file_path = csv_file.file.path
        df = pd.read_csv(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_file_path = 'combined_output.csv'
    combined_df.to_csv(combined_file_path, index=False)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename={combined_file_path}'
    combined_df.to_csv(response, index=False)
    
    return response