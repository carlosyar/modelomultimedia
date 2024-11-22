# Usar una imagen base oficial de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos necesarios
COPY requirements.txt .
COPY app.py .
COPY model3.keras ./model3.keras
COPY templates/ ./templates
COPY static/ ./static

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el contenedor
EXPOSE 8080

# Ejecutar la aplicación
CMD ["python", "app.py"]
