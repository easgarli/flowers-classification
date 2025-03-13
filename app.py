from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import io

app = Flask(__name__)

# Define class names
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Model loading functions
def load_custom_cnn():
    return keras.Sequential([
        keras.layers.TFSMLayer("custom_cnn_model_savedmodel", call_endpoint="serving_default")
    ])

def load_vgg16():
    return keras.Sequential([
        keras.layers.TFSMLayer("VGG16_savedmodel", call_endpoint="serving_default")
    ])

def load_resnet152():
    return keras.Sequential([
        keras.layers.TFSMLayer("ResNet152_savedmodel", call_endpoint="serving_default")
    ])

# Load all models at startup
models = {
    'custom_cnn': load_custom_cnn(),
    'vgg16': load_vgg16(),
    'resnet152': load_resnet152()
}

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    model_choice = request.form.get('model', 'custom_cnn')
    
    try:
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        
        # Get prediction
        model = models[model_choice]
        prediction = model.predict(processed_image)
        confidence_values = list(prediction.values())[0]
        
        predicted_class = class_names[np.argmax(confidence_values)]
        confidence = float(np.max(confidence_values))
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)


