import pandas as pd
import ast
import numpy as np
from sklearn.impute import KNNImputer  # Importamos la nueva librería

class DataSetClean():
    def preprocess_data(self, input_path='../assets/HFCO2.csv'):
        print(f"🔄 Processing dataset: {input_path}")

        data_frame = self.read_file(input_path)
        self.save_cvs(data_frame)

    def read_file(self, input_path):
        try:
            data_frame = pd.read_csv(input_path)
            return self.clean_columns(data_frame)
        except FileNotFoundError:
            print(f"❌ Error: file '{input_path}' not found")
            return

    def get_data_frame(self, preprocessed_datafile):
        data_frame = pd.read_csv(preprocessed_datafile)
        data_frame.columns.str.strip()
        columns_to_drop = [
            'modelId', 'datasets', 'co2_reported', 'created_at', 'library_name',
            'performance_metrics', 'datasets_size',
            'datasets_size_efficency', 'performance_score', 'environment',
            'training_type', 'source', 'geographical_location', 'domain'
        ]
        data_frame = data_frame.drop(columns=columns_to_drop, errors='ignore')
        data_frame = self.clean_columns(data_frame)
        return data_frame

    def clean_columns(self, data_frame):
        data_frame.columns.str.strip()
        columns_to_drop = [
            'modelId', 'datasets', 'co2_reported', 'created_at', 'library_name',
             'performance_metrics', 'datasets_size',
            'datasets_size_efficency', 'performance_score'
        ]

        data_frame = data_frame.drop(columns=columns_to_drop, errors='ignore')
        mask = ~np.isinf(data_frame['size_efficency'])

        # Aplicar la máscara al DataFrame
        df_cleaned = data_frame[mask]

        df_cleaned.dropna(subset=['size_efficency'], inplace=True)
        df_imputed = self.impute_data(df_cleaned)
        return df_imputed

    def impute_data(self, df):
        columns_to_impute = ['size', 'size_efficency']

        df_imputed = df.copy()

        imputer = KNNImputer(n_neighbors=5)

        df_imputed[columns_to_impute] = imputer.fit_transform(df_imputed[columns_to_impute])

        return df_imputed

    def parse_single_dict_metrics_robust(self, metric_string):
        try:
            processed_string = str(metric_string).replace('nan', 'None')
            return ast.literal_eval(processed_string)
        except (ValueError, SyntaxError, TypeError):
            return {}

    def save_cvs(self, data_frame, output_path='../assets/HFCO2_preprocessed.csv'):
        data_frame.to_csv(output_path, index=False)
        print(f"Preprocessing completed, file saved on: {output_path}")


