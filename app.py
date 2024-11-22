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

# Global variable to store the model and prevent reloading
model = None

# Function to load the model only when needed
def load_trained_model():
    global model
    if model is None:
        model = load_model('model3.keras')
        print("Modelo cargado.")
    return model

def crear_app():
    app = Flask(__name__)

    # Class names (you can keep this global or move it to the function)
    class_names = [
        'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 
        'Broccoli', 'Cabagge', 'Capsicum', 'Carrot', 
        'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 
        'Pumpkin', 'Radish', 'Tomato'
    ]
    
    # Define the main route
    @app.route('/')
    def home():
        return render_template('index.html')
    
    # Route to classify the image
    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Read the image file and convert it
        img = Image.open(file.stream)
        img = img.resize((150, 150))  # Resize to 150x150
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Load the model only when making the prediction (Lazy Loading)
        model = load_trained_model()

        # Perform the prediction
        preds = model.predict(img_array)
        predicted_class_index = np.argmax(preds[0])  # Get the index of the class with the highest probability
        predicted_class = class_names[predicted_class_index]  # Get the class name
    
        # Construct the path to the predicted class image
        predicted_image_path = f'static/images/{predicted_class}.jpg'
    
        return render_template('result.html', predicted_class=predicted_class, predicted_image=predicted_image_path)
    
    return app

if __name__ == '__main__':
    app = crear_app()
    app.run(debug=True)
