# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:11:21 2024

@author: CARLOS MARTINEZ
"""
from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Desactivar el uso de la GPU si no está disponible
tf.config.set_visible_devices([], 'GPU')
print("GPU desactivada para uso en la aplicación.")

# Variable global para almacenar el modelo
model = None

# Función para cargar el modelo solo cuando sea necesario
def load_trained_model():
    global model
    if model is None:
        model = load_model('model3.keras')
        print("Modelo cargado.")
    return model

def crear_app():
    app = Flask(__name__)

    # Nombres de las clases
    class_names = [
        'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 
        'Broccoli', 'Cabagge', 'Capsicum', 'Carrot', 
        'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 
        'Pumpkin', 'Radish', 'Tomato'
    ]
    
    # Ruta principal
    @app.route('/')
    def home():
        return render_template('index.html')
    
    # Ruta para predecir la imagen
    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Leer y procesar la imagen
        img = Image.open(file.stream)
        img = img.resize((150, 150))  # Redimensionar a 150x150
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar

        # Cargar el modelo solo cuando se haga la predicción (Lazy Loading)
        model = load_trained_model()

        # Realizar la predicción
        preds = model.predict(img_array)
        predicted_class_index = np.argmax(preds[0])  # Obtener el índice de la clase con mayor probabilidad
        predicted_class = class_names[predicted_class_index]  # Obtener el nombre de la clase
    
        # Construir la ruta para la imagen de la clase predicha
        predicted_image_path = f'static/images/{predicted_class}.jpg'
    
        return render_template('result.html', predicted_class=predicted_class, predicted_image=predicted_image_path)
    
    return app

if __name__ == '__main__':
    app = crear_app()
    app.run(debug=True)
