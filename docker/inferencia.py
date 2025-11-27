import joblib
import pandas as pd
import warnings
import sklearn
import logging
import os
from sys import stdout

warnings.simplefilter('ignore')

# Configuración del Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


model_path = 'pipeline.pkl'

if not os.path.exists(model_path):
    logger.error(f"No se encontró el archivo: {model_path}")
    if os.path.exists(f'docker/{model_path}'):
        model_path = f'docker/{model_path}'
        logger.info(f"Modelo encontrado en: {model_path}")
    else:
        raise FileNotFoundError("No se encuentra pipeline.pkl")

pipeline = joblib.load(model_path)
logger.info('loaded pipeline')

input_path = '/files/input.csv' 

df_input = pd.read_csv(input_path)
logger.info(f'loaded input from {input_path}')

if 'RainTomorrow' in df_input.columns:
    df_input = df_input.drop(columns=['RainTomorrow'])

if 'Date' in df_input.columns:
    df_input = df_input.drop(columns=['Date'])

predictions = pipeline.predict(df_input)
probs = pipeline.predict_proba(df_input)[:, 1] # probabilidad si

logger.info('made predictions')

df_output = pd.DataFrame({
    'RainTomorrow_predicted': predictions, 
    'Probability': probs                  
})

df_output['RainTomorrow_Label'] = df_output['RainTomorrow_predicted'].map({1: 'Yes', 0: 'No'})

output_path = '/files/output.csv'
df_output.to_csv(output_path, index=False)

logger.info(f'saved output to {output_path}')
print("Primeras filas de salida:")
print(df_output.head())