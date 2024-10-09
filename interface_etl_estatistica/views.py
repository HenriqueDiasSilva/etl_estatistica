import pandas as pd
import zipfile
import re
from io import BytesIO
import unicodedata
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import CsvData
import numpy as np
import matplotlib.pyplot as plt
import base64

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def home(request):
    correlation, p_value, slope, intercept, regression_plot, equation, total_municipios = None, None, None, None, None, None, None

    if request.method == 'POST':
        files = request.FILES.getlist('file')
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

                        # Calcular a média diária por município
                        daily_avg = df.groupby(['municipio', 'data'], as_index=False).agg({
                            'precipitacao': 'mean',
                            'risco_fogo': 'mean'
                        })

                        # Calcular a média geral de precipitação e risco de fogo por município
                        averaged_data = daily_avg.groupby('municipio', as_index=False).agg({
                            'precipitacao': 'mean',
                            'risco_fogo': 'mean'
                        })

                        # Calcular o total de municípios
                        total_municipios = averaged_data['municipio'].nunique()

                        # Remover NaNs e infs
                        averaged_data = averaged_data.replace([np.inf, -np.inf], np.nan).dropna()

                        # Calcular a correlação de Pearson
                        correlation, p_value = pearsonr(averaged_data['precipitacao'], averaged_data['risco_fogo'])
                        
                        # Regressão Linear
                        X = averaged_data['precipitacao'].values.reshape(-1, 1)
                        y = averaged_data['risco_fogo'].values
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        slope = model.coef_[0]
                        intercept = model.intercept_
                        
                        # Formatar a equação da reta
                        equation = f"y = {slope:.4f}x + {intercept:.4f}"

                        # Gerar o gráfico de regressão linear
                        plt.figure()
                        plt.scatter(averaged_data['precipitacao'], averaged_data['risco_fogo'], color='blue', label='Dados')
                        plt.plot(averaged_data['precipitacao'], model.predict(X), color='red', label='Linha de Regressão')
                        plt.xlabel('Precipitação')
                        plt.ylabel('Risco de Fogo')
                        plt.title('Regressão Linear entre Precipitação e Risco de Fogo')
                        plt.legend()

                        # Salvar o gráfico em um objeto BytesIO e convertê-lo para base64
                        buffer = BytesIO()
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        regression_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        buffer.close()

                        # Salvar os dados no banco de dados
                        averaged_csv = BytesIO()
                        averaged_data.to_csv(averaged_csv, index=False, encoding='utf-8')
                        averaged_csv.seek(0)
                        new_filename = f'averaged_{csv_filename}'
                        csv_data = CsvData(name=new_filename)
                        csv_data.file.save(new_filename, ContentFile(averaged_csv.read()))

                        messages.success(request, f"O arquivo enviado foi processado com sucesso e armazenado no banco de dados após a aplicação do ETL ({new_filename})")
                        messages.success(request, f"A correlação de Pearson e gráfico de regressão foram calculados.")

    context = {
        'correlation': correlation,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'equation': equation,
        'regression_plot': regression_plot,
        'total_municipios': total_municipios,
    }
    
    return render(request, 'home.html', context)


def about(request):
    return render(request, 'about.html')

    if request.method == 'POST':
        
        df = pd.read_csv('../csv_files/averaged_focos_br_todos-sats_2022.csv')

        # Calcular a Correlação de Pearson
        correlation, p_value = pearsonr(df['precipitacao'], df['risco_fogo'])
        print(f"Correlação de Pearson: {correlation:.4f}")
        print(f"Valor-p: {p_value:.4f}")

        # Preparar os dados para a Regressão Linear
        X = df['precipitacao'].values.reshape(-1, 1)  # Variável independente
        y = df['risco_fogo'].values  # Variável dependente

        # Instanciar e ajustar o modelo de Regressão Linear
        model = LinearRegression()
        model.fit(X, y)

        # Coeficientes da Regressão Linear
        slope = model.coef_[0]
        intercept = model.intercept_
        print(f"Coeficiente angular (slope): {slope:.4f}")
        print(f"Intercepto: {intercept:.4f}")

        # Previsão para os valores de X
        y_pred = model.predict(X)

        # Plotar o Gráfico de Dispersão e a Linha de Regressão
        plt.scatter(df['precipitacao'], df['risco_fogo'], color='blue', label='Dados')
        plt.plot(df['precipitacao'], y_pred, color='red', label='Linha de Regressão')
        plt.xlabel('Precipitação')
        plt.ylabel('Risco de Fogo')
        plt.title('Regressão Linear entre Precipitação e Risco de Fogo')
        plt.legend()
        plt.show()
    
    
    return render(request, 'interface.html')