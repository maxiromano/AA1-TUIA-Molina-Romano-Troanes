import joblib
import pandas as pd
import warnings
import logging
import os
import numpy as np
from sys import stdout

warnings.simplefilter('ignore')

# Configuración del Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


MODEL_FILENAME = 'pipeline.pkl'
META_FILENAME = 'imputacion_metadata.pkl'
INPUT_FILENAME = 'input.csv'
OUTPUT_FILENAME = 'output.csv'

base_path_docker = '/files'
base_path_local = 'docker/files'

# Función auxiliar para encontrar archivos
def get_path(filename, check_exists=True):
    path = os.path.join(base_path_docker, filename)
    if os.path.exists(path):
        return path
    path = os.path.join(base_path_local, filename)
    if os.path.exists(path):
        return path
    if os.path.exists(filename):
        return filename
    if not check_exists:
        return os.path.join(base_path_docker, filename)
    return None

#cargar pipeline
model_path = get_path(MODEL_FILENAME)
if not model_path:
    raise FileNotFoundError(f"No se encuentra {MODEL_FILENAME}")
pipeline = joblib.load(model_path)
logger.info(f'Pipeline cargado desde: {model_path}')

#cargar datos imputacion
meta_path = get_path(META_FILENAME)
if not meta_path:
    raise FileNotFoundError(f"No se encuentra {META_FILENAME}. Recuerda generar este archivo en el entrenamiento.")
metadata = joblib.load(meta_path)
logger.info(f'Metadata de imputación cargada desde: {meta_path}')

medias_por_prov = metadata['medias_por_prov']
modas_por_prov = metadata['modas_por_prov']
global_means = metadata['global_means']
global_modes = metadata['global_modes']
mapeo_ciudades = metadata['mapeo_ciudades']
normalizar_ubicaciones = metadata['normalizar_ubicaciones']

input_path = get_path(INPUT_FILENAME)
if not input_path:
    input_path = os.path.join(base_path_docker, INPUT_FILENAME)
    raise FileNotFoundError(f"No se encuentra el archivo de entrada: {input_path}")

df_input = pd.read_csv(input_path)
logger.info(f'Input cargado desde {input_path}')


if 'RainTomorrow' in df_input.columns:
    df_input = df_input.drop(columns=['RainTomorrow'])
if 'Date' in df_input.columns:
    df_input = df_input.drop(columns=['Date'])

#imputacion

#crear columna provincia
df_input['Location_normalized'] = df_input['Location'].replace(normalizar_ubicaciones)
df_input['provincia'] = df_input['Location_normalized'].map(mapeo_ciudades)


correcciones_manuales = {
    'NorfolkIsland': 'Norfolk Island Territory',
    'Nhil': 'Victoria',
    'Dartmoor': 'Victoria',
    'Woomera': 'South Australia',
    'Witchcliffe': 'Western Australia',
    'SalmonGums': 'Western Australia',
    'Walpole': 'Western Australia',
    'AliceSprings': 'Northern Territory',
    'Uluru': 'Northern Territory'
}
for loc, prov in correcciones_manuales.items():
    mask = df_input['Location'] == loc
    if mask.any():
        df_input.loc[mask, 'provincia'] = prov

#imputacion numerica 
cols_numericas = medias_por_prov.columns 
for col in cols_numericas:
    if col in df_input.columns:
        fill_values = df_input['provincia'].map(medias_por_prov[col])
        df_input[col] = df_input[col].fillna(fill_values)

#imputacion categoricas
cols_categoricas = modas_por_prov.columns
for col in cols_categoricas:
    if col in df_input.columns:
        fill_values = df_input['provincia'].map(modas_por_prov[col])
        df_input[col] = df_input[col].fillna(fill_values)


if 'Location_normalized' in df_input.columns:
    df_input = df_input.drop(columns=['Location_normalized'])

logger.info('Imputación finalizada.')

#prediccion
try:
    predictions = pipeline.predict(df_input)
    probs = pipeline.predict_proba(df_input)[:, 1] # Probabilidad de clase 1 (Si)
    logger.info('Predicciones generadas correctamente')
except Exception as e:
    logger.error(f"Error durante la predicción: {e}")
    raise e

df_output = pd.DataFrame({
    'RainTomorrow_predicted': predictions, 
    'Probability': probs                  
})

df_output['RainTomorrow_Label'] = df_output['RainTomorrow_predicted'].map({1: 'Yes', 0: 'No'})

output_path = get_path(OUTPUT_FILENAME, check_exists=False)

if output_path is None: 
    if os.path.exists('docker/files'):
        output_path = 'docker/files/output.csv'
    else:
        output_path = 'output.csv'

df_output.to_csv(output_path, index=False)

logger.info(f'Resultados guardados en: {output_path}')
print("Primeras filas de salida:")
print(df_output.head())