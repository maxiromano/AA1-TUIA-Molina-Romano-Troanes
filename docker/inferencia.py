import joblib
import pandas as pd
import warnings
import logging
import os
import numpy as np
from sys import stdout

warnings.simplefilter('ignore')

#configuracion logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
 
model_path = 'pipeline.pkl'
meta_path  = 'files/imputacion_metadata.pkl' 
input_path = 'files/input.csv'
output_path = 'files/output.csv'

pipeline = joblib.load(model_path)

metadata = joblib.load(meta_path)

medias_por_prov = metadata['medias_por_prov']
modas_por_prov = metadata['modas_por_prov']
global_means = metadata['global_means']
global_modes = metadata['global_modes']
mapeo_ciudades = metadata['mapeo_ciudades']
normalizar_ubicaciones = metadata['normalizar_ubicaciones']

df_input = pd.read_csv(input_path)

if 'RainTomorrow' in df_input.columns:
    df_input = df_input.drop(columns=['RainTomorrow'])
if 'Date' in df_input.columns:
    df_input = df_input.drop(columns=['Date'])

#imputacion

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

#Imputación numérica 
cols_numericas = medias_por_prov.columns 
for col in cols_numericas:
    if col in df_input.columns:
        fill_values = df_input['provincia'].map(medias_por_prov[col])
        df_input[col] = df_input[col].fillna(fill_values)

#Imputación categórica
cols_categoricas = modas_por_prov.columns
for col in cols_categoricas:
    if col in df_input.columns:
        fill_values = df_input['provincia'].map(modas_por_prov[col])
        df_input[col] = df_input[col].fillna(fill_values)

if 'Location_normalized' in df_input.columns:
    df_input = df_input.drop(columns=['Location_normalized'])

#prediccion
predictions = pipeline.predict(df_input)
probs = pipeline.predict_proba(df_input)[:, 1]


#salida
df_output = pd.DataFrame({
    'RainTomorrow_predicted': predictions, 
    'Probability': probs                  
})

df_output['RainTomorrow_Label'] = df_output['RainTomorrow_predicted'].map({1: 'Llueve', 0: 'No llueve'})

# Guardar
df_output.to_csv(output_path, index=False)
print("Primeras filas de salida:")
print(df_output.head())