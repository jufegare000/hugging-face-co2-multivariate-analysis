

import pandas as pd
import json

# 1. Carga el archivo CSV en un DataFrame de pandas
try:
    df = pd.read_csv('HFClean.csv')
except Exception as e:
    print(f"Error al leer el CSV: {e}")
    df = pd.read_csv('HFClean.csv', encoding='latin1')

# --- LÍNEAS NUEVAS PARA DIAGNÓSTICO ---
# Imprime los nombres de todas las columnas para que puedas encontrar el correcto.
print("🕵️‍♂️ Los nombres de las columnas en tu archivo son:")
print(list(df.columns))

# Imprime las primeras 5 filas para que veas el contenido y puedas identificar la columna JSON.
print("\n primeiras 5 filas do teu arquivo:")
print(df.head())
# --- FIN DE LAS LÍNEAS DE DIAGNÓSTICO ---


# 2. Identifica la columna JSON (¡AQUÍ ESTÁ EL CAMBIO IMPORTANTE!)
# Reemplaza 'tags' con el nombre real que viste en la lista de arriba.
# Por ejemplo, si la columna se llama 'model_info' o 'metadata', cámbialo aquí.
json_column_name = 'performance_metrics' # <--- ¡CAMBIA ESTE VALOR!
# El resto del script sigue igual...
# 3. Convierte la cadena de texto de cada fila a un objeto JSON
def parse_json(x):
    # Verifica si el valor es una cadena de texto antes de intentar procesarlo
    if isinstance(x, str):
        try:
            return json.loads(x.replace("'", "\""))
        except (json.JSONDecodeError, AttributeError):
            return {}
    return {} # Devuelve un diccionario vacío si no es una cadena de texto

parsed_json = df[json_column_name].apply(parse_json)

# 4. Expande el JSON en nuevas columnas
json_expanded = pd.json_normalize(parsed_json)

# 5. Une las nuevas columnas al DataFrame original
df = df.drop(columns=[json_column_name])
df_final = pd.concat([df.reset_index(drop=True), json_expanded], axis=1)

print("\n✅ DataFrame con el JSON expandido:")
print(df_final.head())

df_final.to_csv('HFCO2_limpio2.csv', index=False)

print("\n🎉 ¡Proceso completado! Busca el archivo 'HFCO2_limpio.csv' en tu carpeta.")