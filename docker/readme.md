# Inferencia de Predicción de Lluvia (TP2)

Contenedor Docker para ejecutar inferencia sobre el modelo de Regresión Logística entrenado.

## Contenido de la carpeta
* `inferencia.py`: Script principal de predicción.
* `Dockerfile`: Configuración de la imagen.
* `requirements.txt`: Dependencias necesarias.
* `*.pkl`: Artefactos del modelo (Scaler, Encoders, Modelo).

## Instrucciones de Ejecución

### 1. Construir la imagen
Desde la terminal, ubicado en esta carpeta `docker`:
```bash
docker build -t tp-lluvia .