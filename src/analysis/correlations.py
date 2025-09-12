import pandas as pd

preprocessed_datafile = '../assets/HFCO2.csv'

class CovarianceUtils():
    def covariances_and_correlations(self):
        try:
            data_frame = pd.read_csv(preprocessed_datafile)
            self.calculate_covariance_matrix(data_frame)
            self.calculate_correlation_matrix(data_frame)

        except FileNotFoundError:
            print("❌ Error: Ensure 'HFCO2_preprocessed.csv' exists")
        except Exception as e:
            print(f"Error:  {e}")

    def calculate_covariance_matrix(self, data_frame):
        return data_frame.cov()


    def calculate_correlation_matrix(self, data_frame):

        return data_frame.corr()

    def convert_categorical(self, data_frame):
        categorical_columns = ['training_type', 'source', 'geographical_location', 'domain']

        data_frame = pd.get_dummies(data_frame, columns=categorical_columns, drop_first=True)
        return data_frame

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
        return data_frame
