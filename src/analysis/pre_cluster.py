import pandas as pd
import numpy as np
import json
import ast


class PreClusterUtils():

    def parse_metrics(self, x):
        if isinstance(x, dict):
            return x
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return np.nan
        s_norm = s.replace("NaN", "null").replace("nan", "null").replace("None", "null").replace("'", '"')
        try:
            d = json.loads(s_norm)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return {k: (np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else v) for k, v in
                        d.items()}
        except Exception:
            return np.nan
        return np.nan

    def all_nan_metrics(self, dct):
        if not isinstance(dct, dict):
            return True
        if len(dct) == 0:
            return True
        return not any(pd.notna(v) for v in dct.values())

    def save_dataset(self):
        datafile = '../../assets/HFCO2.csv'
        df = pd.read_csv(datafile)

        df['performance_metrics_parsed'] = df['performance_metrics'].apply(self.parse_metrics)

        mask_to_drop = df['performance_metrics_parsed'].apply(self.all_nan_metrics)

        df_clean = df[~mask_to_drop].copy()

        metrics_df = pd.json_normalize(df_clean['performance_metrics_parsed']).add_prefix('pm_')

        df_clean.reset_index(drop=True, inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)

        df_final = pd.concat([df_clean, metrics_df], axis=1)

        df_final = df_final.drop(columns=['performance_metrics', 'performance_metrics_parsed'])

        df_final.to_csv("../../assets/huggingface_co2_clean.csv", index=False)

    def categorize_model_type_detailed(self, input_filepath, output_filepath):
        try:
            df = pd.read_csv(input_filepath)
            print(f"📄 Archivo '{input_filepath}' cargado con {len(df)} filas.")

            has_accuracy = df['pm_accuracy'].notna()
            has_f1 = df['pm_f1'].notna()
            has_rouge1 = df['pm_rouge1'].notna()
            has_rougeL = df['pm_rougeL'].notna()

            condicion_tipo1 = has_accuracy & has_f1
            condicion_tipo2 = has_rouge1 & has_rougeL

            condicion_tipo3 = has_accuracy & ~has_f1 & ~has_rouge1 & ~has_rougeL

            condicion_tipo4 = has_f1 & ~has_accuracy & ~has_rouge1 & ~has_rougeL

            condicion_tipo5 = has_rouge1 & ~has_accuracy & ~has_f1 & ~has_rougeL

            condicion_tipo6 = has_rougeL & ~has_accuracy & ~has_f1 & ~has_rouge1

            conditions = [
                condicion_tipo1,
                condicion_tipo2,
                condicion_tipo3,
                condicion_tipo4,
                condicion_tipo5,
                condicion_tipo6,
            ]

            choices = [
                'type1 (accuracy & f1)',
                'type2 (rouge)',
                'type3 (accuracy)',
                'type4 (f1)',
                'type5 (rouge1)',
                'type6 (rougeL)',
            ]

            # 5. Aplicar la clasificación.
            df['model_type'] = np.select(conditions, choices, default='undefined')

            # 6. Mostrar el resumen.
            print("\n📊 Conteo de modelos por tipo (clasificación detallada):")
            print(df['model_type'].value_counts())

            # 7. Guardar el resultado.
            df.to_csv(output_filepath, index=False)
            print(f"\n✅ Archivo con categorías detalladas guardado en '{output_filepath}'.")

        except FileNotFoundError:
            print(f"❌ Error: No se pudo encontrar el archivo de entrada en '{input_filepath}'.")
        except KeyError as e:
            print(f"❌ Error: Falta una columna necesaria en el archivo: {e}")
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")

import os

print("Current Working Directory:", os.getcwd())

utils = PreClusterUtils()

utils.categorize_model_type_detailed("../../assets/huggingface_co2_clean.csv", "../../assets/huggingface_co2_clean_pm.csv")
