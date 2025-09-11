import pandas as pd
import ast

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

    def clean_columns(self, data_frame):
        data_frame.columns.str.strip()
        columns_to_drop = [
            'modelId', 'datasets', 'co2_reported', 'created_at', 'library_name',
             'performance_metrics', 'datasets_size',
            'datasets_size_efficency', 'performance_score'
        ]
        data_frame = data_frame.drop(columns=columns_to_drop, errors='ignore')
        return data_frame

    def parse_single_dict_metrics_robust(self, metric_string):
        try:
            processed_string = str(metric_string).replace('nan', 'None')
            return ast.literal_eval(processed_string)
        except (ValueError, SyntaxError, TypeError):
            return {}

    def save_cvs(self, data_frame, output_path='../assets/HFCO2_preprocessed.csv'):
        data_frame.to_csv(output_path, index=False)
        print(f"Preprocessing completed, file saved on: {output_path}")

