import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast # Para convertir strings a diccionarios de forma segura


def preprocess_huggingface_data_final_v9(input_path='HFCO2.csv', output_path='HFCO2_preprocessed.csv'):
    """
    Script de preprocesamiento, versión 9.

    Cambio:
    - Se ha añadido 'geographical_location' a la lista de columnas para eliminar.
    """
    print(f"🔄 Iniciando el preprocesamiento v9 (excluyendo location) del archivo: {input_path}")

    # --- 1. Carga y Limpieza Inicial ---
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ Error: El archivo '{input_path}' no fue encontrado.")
        return

    df.columns = df.columns.str.strip()
    print("✅ Nombres de columnas limpiados.")

    # --- 2. Selección y Eliminación de Columnas ---
    # **CAMBIO AQUÍ: 'geographical_location' añadido a la lista**
    columns_to_drop = [
        'modelId', 'datasets', 'co2_reported', 'created_at', 'library_name',
        'environment', 'source', 'domain', 'geographical_location'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print("✅ Columnas irrelevantes (incluyendo 'geographical_location') eliminadas.")

    # --- 3. Extracción de Métricas de Rendimiento ---
    if 'performance_metrics' in df.columns:
        print("⚙️  Procesando la columna 'performance_metrics'...")

        def parse_single_dict_metrics_robust(metric_string):
            try:
                processed_string = str(metric_string).replace('nan', 'None')
                return ast.literal_eval(processed_string)
            except (ValueError, SyntaxError, TypeError):
                return {}

        temp_metrics = df['performance_metrics'].apply(parse_single_dict_metrics_robust)

        df['metric_accuracy'] = temp_metrics.apply(lambda x: x.get('accuracy'))
        df['metric_f1'] = temp_metrics.apply(lambda x: x.get('f1'))
        df['metric_rouge1'] = temp_metrics.apply(lambda x: x.get('rouge1'))
        df['metric_rougeL'] = temp_metrics.apply(lambda x: x.get('rougeL'))

        df = df.drop(columns=['performance_metrics'])
        print("✅ Métricas de rendimiento extraídas correctamente.")

    # --- 4. Conversión de Tipos y Relleno de Nulos Categóricos ---
    bool_cols = df.select_dtypes(include='bool').columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    if len(bool_cols) > 0:
        print(f"✅ Columnas booleanas ({', '.join(bool_cols)}) convertidas a 0/1.")

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna('unknown', inplace=True)
    print("✅ Valores nulos en columnas categóricas rellenados con 'unknown'.")

    # --- 5. Codificación One-Hot ---
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False, drop_first=True)
    print(f"✅ Variables categóricas convertidas con One-Hot Encoding.")

    # --- 6. Preservación de Nulos Numéricos ---
    print("⚠️  Se omite el relleno de valores numéricos nulos para preservar la integridad de los datos.")

    # --- 7. Guardar el archivo preprocesado ---
    df.to_csv(output_path, index=False)
    print(f"🎉 ¡Preprocesamiento completo! El archivo final ha sido guardado en: {output_path}")


import pandas as pd

def covariances():
    try:
        df = pd.read_csv('HFCO2_preprocessed.csv')
        print("✅ Archivo preprocesado cargado correctamente.")
        print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

        # --- 1. Calcular y guardar la Matriz de Varianza-Covarianza ---
        cov_matrix = df.cov()
        cov_matrix.to_csv('covariance_matrix.csv')
        print("✅ Matriz de Varianza-Covarianza guardada en 'covariance_matrix.csv'")

        # --- 2. Calcular y guardar la Matriz de Correlación ---
        corr_matrix = df.corr()
        corr_matrix.to_csv('correlation_matrix.csv')
        print("✅ Matriz de Correlación guardada en 'correlation_matrix.csv'")

    except FileNotFoundError:
        print("❌ Error: Asegúrate de que 'HFCO2_preprocessed.csv' está en la misma carpeta.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == '__main__':
    preprocess_huggingface_data_final_v9()

