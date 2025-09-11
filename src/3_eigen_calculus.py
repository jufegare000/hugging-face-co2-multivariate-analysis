import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer  # Importamos la nueva librería
from sklearn.decomposition import PCA, FactorAnalysis
import numpy as np

preprocessed_datafile = 'HFCO2_preprocessed.csv'

def eigen_calculus():
    try:
        df = pd.read_csv(preprocessed_datafile)
        print(f"✅ Archivo preprocesado cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")

        # --- SOLUCIÓN: Reemplazar valores infinitos con NaN ---
        # Esta línea busca infinitos (positivos y negativos) y los convierte en nulos.
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("✅ Valores infinitos reemplazados por NaN.")

        # --- Imputación de valores NaN usando K-Vecinos Más Cercanos ---
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = imputer.fit_transform(df)
        df = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
        print("✅ Valores nulos (NaN) rellenados usando el método KNN.")

        # 1. Estandarizar los datos
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        print("✅ Los datos han sido estandarizados.")

        # --- 2. Análisis de Componentes Principales (ACP) ---
        pca = PCA(n_components=df.shape[1])
        pca.fit(df_scaled)
        pca_eigenvalues = pca.explained_variance_
        pca_eigenvectors_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i + 1}' for i in range(len(pca_eigenvalues))],
            index=df.columns
        )

        print("\n\n--- ANÁLISIS DE COMPONENTES PRINCIPALES (ACP) ---")
        print("\nValores Propios (Eigenvalues):")
        for i, val in enumerate(pca_eigenvalues):
            print(f"  PC{i + 1}: {val:.4f}")

        print("\nVectores Propios (Cargas de las variables en cada componente):")
        print(pca_eigenvectors_df.head(10))

        # --- 3. Análisis Factorial (AF) ---
        n_factors = 5
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(df_scaled)
        fa_loadings_df = pd.DataFrame(
            fa.components_.T,
            columns=[f'Factor{i + 1}' for i in range(n_factors)],
            index=df.columns
        )

        print("\n\n--- ANÁLISIS FACTORIAL (AF) ---")
        print("\nMatriz de Cargas Factoriales (Vectores Propios):")
        print(fa_loadings_df.head(10))

    except FileNotFoundError:
        print("❌ Error: Asegúrate de que 'HFCO2_preprocessed.csv' está en la misma carpeta.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")


def scale_data(data_frame):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data_frame)
    print("\n✅ Data scaled successfully.")
    return df_scaled

if __name__ == '__main__':
    eigen_calculus()