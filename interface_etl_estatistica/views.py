import pandas as pd
import zipfile
import re
from io import BytesIO
import unicodedata
from scipy.stats import pearsonr, spearmanr
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

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std_dev = df[column].std()
    outliers = df[(df[column] < mean - threshold * std_dev) | (df[column] > mean + threshold * std_dev)]
    filtered_df = df[(df[column] >= mean - threshold * std_dev) & (df[column] <= mean + threshold * std_dev)]
    return filtered_df, outliers

def plot_regression(df, x_column, y_column, title):
    X = df[x_column].values.reshape(-1, 1)
    y = df[y_column].values

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {slope:.4f}x + {intercept:.4f}"

    plt.figure()
    plt.scatter(df[x_column], df[y_column], color='blue', label='Dados')
    plt.plot(df[x_column], model.predict(X), color='red', label=f'Linha de Tendência')
    plt.xlabel('Precipitação')
    plt.ylabel('Risco de Queimadas')
    plt.title(title)
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return plot_base64, equation

def home(request):
    # Inicializar variáveis no início da função
    correlation, p_value, slope, intercept, regression_plot, equation, total_municipios = None, None, None, None, None, None, None
    spearman_corr, spearman_p_value = None, None
    correlation_no_outliers, p_value_no_outliers = None, None
    spearman_corr_no_outliers, spearman_p_value_no_outliers = None, None
    spearman_regression_plot_all, spearman_equation_all = None, None
    spearman_regression_plot_no_outliers, spearman_equation_no_outliers = None, None
    slope_no_outliers, intercept_no_outliers, regression_plot_no_outliers, equation_no_outliers = None, None, None, None
    outliers, total_municipios_outliers, total_municipios_sem_outliers = None, None, None

    if request.method == 'POST':
        files = request.FILES.getlist('file')
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

                        daily_avg = df.groupby(['municipio', 'data'], as_index=False).agg({
                            'precipitacao': 'mean',
                            'risco_fogo': 'mean'
                        })

                        averaged_data = daily_avg.groupby('municipio', as_index=False).agg({
                            'precipitacao': 'mean',
                            'risco_fogo': 'mean'
                        })

                        total_municipios = averaged_data['municipio'].nunique()
                        averaged_data = averaged_data.replace([np.inf, -np.inf], np.nan).dropna()

                        # Calcular a correlação de Pearson e Spearman para o conjunto total
                        correlation, p_value = pearsonr(averaged_data['precipitacao'], 
                                                        averaged_data['risco_fogo'])
                        spearman_corr, spearman_p_value = spearmanr(averaged_data['precipitacao'], 
                                                                    averaged_data['risco_fogo'])
                        spearman_regression_plot_all, spearman_equation_all = plot_regression(averaged_data, 'precipitacao', 'risco_fogo', 
                                                                                      'Regressão Linear entre Precipitação e Risco de Fogo (Todos os Dados)')

                        # Formatar os valores de p_value e spearman_p_value em notação científica
                        p_value = f"{p_value:.2e}"
                        spearman_p_value = f"{spearman_p_value:.2e}"

                        # Gráfico com todos os dados
                        x = averaged_data['precipitacao'].values.reshape(-1, 1)
                        y = averaged_data['risco_fogo'].values
                        model = LinearRegression()
                        model.fit(x, y)
                        slope = model.coef_[0]
                        intercept = model.intercept_
                        equation = f"y = {slope:.4f}x + {intercept:.4f}"

                        plt.figure()
                        plt.scatter(averaged_data['precipitacao'], averaged_data['risco_fogo'],
                                    color='green', label='Dados')
                        plt.plot(averaged_data['precipitacao'], model.predict(x), 
                                 color='orange', label='Linha de Regressão')
                        plt.xlabel('Precipitação')
                        plt.ylabel('Risco de Queimadas')
                        plt.title('Regressão Linear entre Precipitação e Risco de Queimadas (Todos os Dados)')
                        plt.legend()

                        buffer = BytesIO()
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        regression_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        buffer.close()

                        # Remover outliers
                        averaged_data_no_outliers, outliers_precipitacao = remove_outliers(averaged_data, 'precipitacao')
                        averaged_data_no_outliers, outliers_risco_fogo = remove_outliers(averaged_data_no_outliers, 'risco_fogo')
                        outliers = pd.concat([outliers_precipitacao, outliers_risco_fogo]).drop_duplicates()

                        # Calcular o total de municípios entre os outliers e sem outliers
                        total_municipios_outliers = outliers['municipio'].nunique()
                        total_municipios_sem_outliers = averaged_data_no_outliers['municipio'].nunique()

                        # Verificar se há dados suficientes sem outliers para calcular a regressão
                        if not averaged_data_no_outliers.empty and len(averaged_data_no_outliers['precipitacao'].unique()) > 1:
                            # Calcular a correlação de Pearson e Spearman para o conjunto sem outliers
                            correlation_no_outliers, p_value_no_outliers = pearsonr(
                                averaged_data_no_outliers['precipitacao'], averaged_data_no_outliers['risco_fogo']
                            )
                            spearman_corr_no_outliers, spearman_p_value_no_outliers = spearmanr(
                                averaged_data_no_outliers['precipitacao'], averaged_data_no_outliers['risco_fogo']
                            )
                            spearman_regression_plot_no_outliers, spearman_equation_no_outliers = plot_regression(
                                averaged_data_no_outliers, 'precipitacao', 'risco_fogo', 
                                'Regressão Linear entre Precipitação e Risco de Queimadas (Sem Outliers)')
                            
                            # Formatar os valores de p_value e spearman_p_value em notação científica
                            p_value_no_outliers = f"{p_value_no_outliers:.2e}"
                            spearman_p_value_no_outliers = f"{spearman_p_value_no_outliers:.2e}"

                            X_no_outliers = averaged_data_no_outliers['precipitacao'].values.reshape(-1, 1)
                            y_no_outliers = averaged_data_no_outliers['risco_fogo'].values
                            model_no_outliers = LinearRegression()
                            model_no_outliers.fit(X_no_outliers, y_no_outliers)
                            slope_no_outliers = model_no_outliers.coef_[0]
                            intercept_no_outliers = model_no_outliers.intercept_
                            equation_no_outliers = f"y = {slope_no_outliers:.4f}x + {intercept_no_outliers:.4f}"

                            plt.figure()
                            plt.scatter(averaged_data_no_outliers['precipitacao'], averaged_data_no_outliers['risco_fogo'], color='green', label='Dados')
                            plt.plot(averaged_data_no_outliers['precipitacao'], model_no_outliers.predict(X_no_outliers), color='orange', label='Linha de Regressão')
                            plt.xlabel('Precipitação')
                            plt.ylabel('Risco de Queimadas')
                            plt.title('Regressão Linear entre Precipitação e Risco de Queimadas (Sem Outliers)')
                            plt.legend()

                            buffer_no_outliers = BytesIO()
                            plt.savefig(buffer_no_outliers, format='png')
                            buffer_no_outliers.seek(0)
                            regression_plot_no_outliers = base64.b64encode(buffer_no_outliers.getvalue()).decode('utf-8')
                            buffer_no_outliers.close()

                        # Salvar os dados no banco de dados
                        averaged_csv = BytesIO()
                        averaged_data.to_csv(averaged_csv, index=False, encoding='utf-8')
                        averaged_csv.seek(0)
                        new_filename = f'averaged_{csv_filename}'
                        csv_data = CsvData(name=new_filename)
                        csv_data.file.save(new_filename, ContentFile(averaged_csv.read()))

    context = {
        'correlation': correlation,
        'p_value': p_value,
        'spearman_corr': spearman_corr,
        'spearman_p_value': spearman_p_value,
        'spearman_regression_plot_all': spearman_regression_plot_all,
        'spearman_equation_all': spearman_equation_all,
        'spearman_regression_plot_no_outliers': spearman_regression_plot_no_outliers,
        'spearman_equation_no_outliers': spearman_equation_no_outliers,
        'slope': slope,
        'intercept': intercept,
        'equation': equation,
        'regression_plot': regression_plot,
        'total_municipios': total_municipios,
        'correlation_no_outliers': correlation_no_outliers,
        'p_value_no_outliers': p_value_no_outliers,
        'spearman_corr_no_outliers': spearman_corr_no_outliers,
        'spearman_p_value_no_outliers': spearman_p_value_no_outliers,
        'slope_no_outliers': slope_no_outliers,
        'intercept_no_outliers': intercept_no_outliers,
        'equation_no_outliers': equation_no_outliers,
        'regression_plot_no_outliers': regression_plot_no_outliers,
        'outliers': outliers.to_dict(orient='records') if outliers is not None else None,
        'total_municipios_outliers': total_municipios_outliers,
        'total_municipios_sem_outliers': total_municipios_sem_outliers
    }
    
    return render(request, 'home.html', context)

def about(request):
    return render(request, 'about.html')