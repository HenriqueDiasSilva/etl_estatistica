import pandas as pd
import difflib
import zipfile
import re
from io import BytesIO
import unicodedata
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import CsvData

def normalize_text(text):
    # Normaliza o texto para remover caracteres especiais e acentos
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def match_names(df1_names, df2_names, threshold=0.8):
    # Encontra o nome mais próximo em df2_names para cada nome em df1_names
    name_mapping = {}
    for name in df1_names:
        match = difflib.get_close_matches(name, df2_names, n=1, cutoff=threshold)
        if match:
            name_mapping[name] = match[0]
    return name_mapping

def home(request):
    if request.method == 'POST':
        files = request.FILES.getlist('file')
        consolidated_data = []
        averaged_data = None

        year_pattern = re.compile(r'^\d{4}\.zip$')

        for file in files:
            if not year_pattern.match(file.name):
                if not zipfile.is_zipfile(file):
                    messages.warning(request, f"O arquivo {file.name} não é um arquivo ZIP válido.")
                    continue

                with zipfile.ZipFile(file) as z:
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]

                    if len(csv_files) != 1:
                        messages.warning(request, f"O arquivo {file.name} deve conter exatamente um arquivo CSV.")
                        continue

                    csv_filename = csv_files[0]
                    with z.open(csv_filename) as csv_file:
                        df = pd.read_csv(csv_file, delimiter=',', encoding='latin1')
                        df['municipio'] = df['municipio'].apply(normalize_text)
                        df['data'] = pd.to_datetime(df['data_pas']).dt.date
                        df = df[df['risco_fogo'] != -999]
                        averaged_data = df.groupby(['municipio', 'data'], as_index=False).agg({
                            'precipitacao': 'mean',
                            'risco_fogo': 'mean'
                        })

                        # Salvar o arquivo averaged_data no banco de dados
                        averaged_csv = BytesIO()
                        averaged_data.to_csv(averaged_csv, index=False, encoding='utf-8')
                        averaged_csv.seek(0)
                        new_filename = f'averaged_{csv_filename}'
                        csv_data = CsvData(name=new_filename)
                        csv_data.file.save(new_filename, ContentFile(averaged_csv.read()))
                        
                    messages.success(request, f"O arquivo com médias {new_filename} foi salvo no banco de dados.")
                continue
            
            if not zipfile.is_zipfile(file):
                messages.warning(request, f"O arquivo {file.name} não é um arquivo ZIP válido.")
                continue

            with zipfile.ZipFile(file) as z:
                for csv_filename in z.namelist():
                    if csv_filename.endswith('.CSV'):
                        with z.open(csv_filename) as csv_file:
                            header_lines = []
                            for _ in range(8):
                                line = csv_file.readline().decode('latin1').strip()
                                header_lines.append(line)
                        
                            estacao = normalize_text(header_lines[2].split(';')[1])
                            df = pd.read_csv(csv_file, delimiter=';', encoding='latin1', skiprows=8)

                            df.columns = ['Data', 'Hora UTC', 'Precipitacao', 'Pressao', 'Pressao Max', 'Pressao Min', 'Radiacao', 
                                          'Temp Bulbo Seco', 'Temp Orvalho', 'Temp Max', 'Temp Min', 'Orvalho Max', 'Orvalho Min', 
                                          'Umidade Rel Max', 'Umidade Rel Min', 'Umidade Rel', 'Vento Dir', 'Vento Rajada Max', 
                                          'Vento Velocidade', 'Extra']
                            
                            df['ESTACAO'] = estacao
                            df_filtered = df[['ESTACAO', 'Data', 'Precipitacao']]
                            df_filtered['Precipitacao'] = pd.to_numeric(df_filtered['Precipitacao'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
                            df_grouped = df_filtered.groupby(['ESTACAO', 'Data'], as_index=False).agg({'Precipitacao': 'sum'})
                            consolidated_data.append(df_grouped)

        # Concatenar e salvar consolidated_data no banco
        if consolidated_data:
            consolidated_df = pd.concat(consolidated_data, ignore_index=True)
            consolidated_csv = BytesIO()
            consolidated_df.to_csv(consolidated_csv, index=False, encoding='utf-8')
            consolidated_csv.seek(0)
            consolidated_filename = 'consolidated_data.csv'
            csv_data = CsvData(name=consolidated_filename)
            csv_data.file.save(consolidated_filename, ContentFile(consolidated_csv.read()))
            messages.success(request, "Arquivo CSV consolidado foi armazenado com sucesso.")

            # Verifica se averaged_data não está vazio antes de tentar a junção
            if not averaged_data.empty:
                name_mapping = match_names(averaged_data['municipio'].unique(), consolidated_df['ESTACAO'].unique())
                averaged_data['municipio_mapped'] = averaged_data['municipio'].map(name_mapping)
                
                # Realizar a junção final baseada no nome mapeado e na data
                merged_df = pd.merge(
                    averaged_data, 
                    consolidated_df, 
                    left_on=['municipio_mapped', 'data'], 
                    right_on=['ESTACAO', 'Data'], 
                    suffixes=('_average', '_consolidated')
                )

                # Gerar e salvar o arquivo combinado
                merged_csv = BytesIO()
                merged_df.to_csv(merged_csv, index=False, encoding='utf-8')
                merged_csv.seek(0)
                merged_filename = 'merged_data.csv'
                csv_data = CsvData(name=merged_filename)
                csv_data.file.save(merged_filename, ContentFile(merged_csv.read()))
                messages.success(request, "Arquivo CSV combinado foi armazenado com sucesso.")
            else:
                messages.warning(request, "Dados de médias estavam vazios, a junção não foi realizada.")
        else:
            messages.warning(request, "Não foram gerados dados consolidados.")

    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def interface(request):
    return render(request, 'interface.html')