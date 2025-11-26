# Imagen base ligera de Python
FROM python:3.10-slim

# Evitar que Python genere archivos de caché
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos primero requirements para aprovechar caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de los archivos (script y modelos)
COPY . .

# Comando por defecto
CMD ["python", "inferencia.py"]