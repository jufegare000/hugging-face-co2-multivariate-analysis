import pandas as pd
import ast
import numpy as np
from sklearn.linear_model import LinearRegression

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
            'training_type', 'source', 'geographical_location', 'domain', 'auto'
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
        df_imputed = self.impute_regression_numerical(df_cleaned)
        return df_imputed

    def impute_regression_numerical(self, data_frame):

        df = data_frame.copy()

        variables_to_impute = ['size', 'size_efficency']

        for target_column in variables_to_impute:
            if df[target_column].isnull().any():
                print(f"Imputando valores faltantes en la columna: {target_column}")

                df_train = df.dropna(subset=[target_column])
                df_to_impute = df[df[target_column].isnull()]

                predictor_columns = [col for col in df.columns if col not in variables_to_impute]

                X_train = df_train[predictor_columns]
                y_train = df_train[target_column]

                X_to_impute = df_to_impute[predictor_columns]

                model = LinearRegression()
                model.fit(X_train, y_train)

                predicted_values = model.predict(X_to_impute)
                df.loc[df[target_column].isnull(), target_column] = predicted_values

        return df

    def parse_single_dict_metrics_robust(self, metric_string):
        try:
            processed_string = str(metric_string).replace('nan', 'None')
            return ast.literal_eval(processed_string)
        except (ValueError, SyntaxError, TypeError):
            return {}

    def save_cvs(self, data_frame, output_path='../assets/HFCO2_preprocessed.csv'):
        data_frame.to_csv(output_path, index=False)
        print(f"Preprocessing completed, file saved on: {output_path}")


