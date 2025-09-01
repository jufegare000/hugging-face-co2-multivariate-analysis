import pandas as pd

preprocessed_datafile = 'HFCO2_preprocessed.csv'
covariance_matrix_output_path = 'covariance_matrix.csv'

def covariances_and_correlations():
    try:
        data_frame = pd.read_csv(preprocessed_datafile)
        calculate_covariance_matrix(data_frame)
        calculate_correlation_matrix(data_frame)

    except FileNotFoundError:
        print("❌ Error: Ensure 'HFCO2_preprocessed.csv' exists")
    except Exception as e:
        print(f"Error:  {e}")

def calculate_covariance_matrix(data_frame):
    cov_matrix = data_frame.cov()
    cov_matrix.to_csv(covariance_matrix_output_path)
    print("✅Covariance matriz saved on: '", covariance_matrix_output_path)

def calculate_correlation_matrix(data_frame):
    corr_matrix = data_frame.corr()
    corr_matrix.to_csv('correlation_matrix.csv')
    print("✅ Correlation matrix saved on: 'correlation_matrix.csv'")

if __name__ == '__main__':
    covariances_and_correlations()