import pandas as pd
import ast

def preprocess_huggingface_data_final_v9(input_path='HFCO2.csv'):
    print(f"🔄 Processing dataset: {input_path}")

    data_frame = read_file(input_path)
    extract_performance_metrics(data_frame)
    convert_types(data_frame)
    save_cvs(data_frame)

def read_file(input_path):
    try:

        data_frame = pd.read_csv(input_path)
        return clean_columns(data_frame)
    except FileNotFoundError:
        print(f"❌ Error: file '{input_path}' not found")
        return

def clean_columns(data_frame):
    data_frame.columns.str.strip()
    columns_to_drop = [
        'modelId', 'datasets', 'co2_reported', 'created_at', 'library_name',
        'environment', 'source', 'domain', 'geographical_location'
    ]
    data_frame = data_frame.drop(columns=columns_to_drop, errors='ignore')
    print("✅ Columns now are cleaned, Irrelevant columns erased")
    return data_frame

def extract_performance_metrics(data_frame):
    if 'performance_metrics' in data_frame.columns:
        print("⚙️  Processing column 'performance_metrics'...")

        temp_metrics = data_frame['performance_metrics'].apply(parse_single_dict_metrics_robust)

        data_frame['metric_accuracy'] = temp_metrics.apply(lambda x: x.get('accuracy'))
        data_frame['metric_f1'] = temp_metrics.apply(lambda x: x.get('f1'))
        data_frame['metric_rouge1'] = temp_metrics.apply(lambda x: x.get('rouge1'))
        data_frame['metric_rougeL'] = temp_metrics.apply(lambda x: x.get('rougeL'))

        print("✅Performance metrics successfully extracted")
        return data_frame.drop(columns=['performance_metrics'])

def parse_single_dict_metrics_robust(metric_string):
    try:
        processed_string = str(metric_string).replace('nan', 'None')
        return ast.literal_eval(processed_string)
    except (ValueError, SyntaxError, TypeError):
        return {}

def convert_types(data_frame):
    bool_cols = data_frame.select_dtypes(include='bool').columns
    for col in bool_cols:
        data_frame[col] = data_frame[col].astype(int)
    if len(bool_cols) > 0:
        print(f"✅ Boolean columns ({', '.join(bool_cols)}) converted toa 0/1.")

    categorical_cols = data_frame.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data_frame[col].fillna('unknown', inplace=True)
    print("✅ Null values in categorical columns filled with 'unknown'.")
    data_frame = pd.get_dummies(data_frame, columns=categorical_cols, dummy_na=False, drop_first=True)
    print(f"✅ Categorical variables converted into a one-hot encoding")
    return data_frame

def save_cvs(data_frame, output_path='HFCO2_preprocessed.csv'):
    data_frame.to_csv(output_path, index=False)
    print(f"Preprocessing completed, file saved on: {output_path}")


if __name__ == '__main__':
    preprocess_huggingface_data_final_v9()
