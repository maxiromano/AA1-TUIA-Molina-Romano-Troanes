import pandas as pd
import joblib
import sys
import numpy as np

# Cargar artefactos
# IMPORTANTE: Los archivos .pkl deben estar en la misma carpeta que este script
print("Cargando modelo y artefactos...")
try:
    MAPPING = joblib.load('artefactos_mapping.pkl')
    ENCODERS = joblib.load('label_encoders.pkl')
    SCALER = joblib.load('scaler.pkl')
    MODELO = joblib.load('modelo_final.pkl')
except FileNotFoundError as e:
    print(f"Error fatal: No se encontró el archivo {e.filename}")
    print("Asegurate de haber copiado los .pkl a la carpeta docker/")
    sys.exit(1)

def preprocesar_datos(df):
    """
    Replica la limpieza del notebook original para el modelo de Regresión Logística.
    """
    df = df.copy()
    
    # 1. Normalizar Ubicaciones
    norm_dict = MAPPING['normalizar_ubicaciones']
    df['Location_normalized'] = df['Location'].replace(norm_dict)
    
    # 2. Ingeniería de Features (Mapeos)
    df['provincia'] = df['Location_normalized'].map(MAPPING['mapeo_ciudades'])
    df['lat'] = df['Location_normalized'].map(MAPPING['mapeo_lat'])
    df['lon'] = df['Location_normalized'].map(MAPPING['mapeo_lon'])
    
    # Eliminar columna auxiliar
    if 'Location_normalized' in df.columns:
        df = df.drop(columns=['Location_normalized'])
    
    # 3. Manejo de Nulos (Rellenamos con 0 para inferencia segura)
    df = df.fillna(0) 

    # 4. Label Encoding
    columnas_label = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'provincia', 'Location']
    
    for col in columnas_label:
        if col in df.columns:
            le = ENCODERS[col]
            # Truco para manejar categorías nuevas no vistas en train
            df[col] = df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
            
    # 5. Scaling
    cols_esperadas = SCALER.feature_names_in_
    # Reordenamos columnas para coincidir exactamente con el entrenamiento
    df = df.reindex(columns=cols_esperadas, fill_value=0)
    
    datos_escalados = SCALER.transform(df)
    return datos_escalados

def predecir(datos_nuevos):
    X_procesado = preprocesar_datos(datos_nuevos)
    prediccion = MODELO.predict(X_procesado)
    return prediccion

if __name__ == "__main__":
    # Datos de prueba de ejemplo
    datos_ejemplo = pd.DataFrame([{
        'MinTemp': 13.4, 'MaxTemp': 22.9, 'Rainfall': 0.6, 'Evaporation': 2.0,
        'Sunshine': 5.0, 'WindGustDir': 'W', 'WindGustSpeed': 44.0,
        'WindDir9am': 'W', 'WindDir3pm': 'WNW', 'WindSpeed9am': 20.0,
        'WindSpeed3pm': 24.0, 'Humidity9am': 71.0, 'Humidity3pm': 22.0,
        'Pressure9am': 1007.7, 'Pressure3pm': 1007.1, 'Cloud9am': 8.0,
        'Cloud3pm': 5.0, 'Temp9am': 16.9, 'Temp3pm': 21.8,
        'RainToday': 'No', 'Location': 'Albury'
    }])

    print("\nDatos de entrada procesados. Realizando inferencia...")
    
    try:
        resultado = predecir(datos_ejemplo)
        clase = "LLOVERÁ" if resultado[0] == 1 else "NO LLOVERÁ"
        print(f"\nResultado del Modelo: {resultado[0]} -> {clase}")
    except Exception as e:
        print(f"\nERROR EN INFERENCIA: {e}")