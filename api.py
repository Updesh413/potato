import tensorflow as tf
import numpy as np
import logging
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PotatoDiseaseClassifier:
    def __init__(self, model_path, class_names):
        self.image_size = 256
        self.class_names = class_names
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def preprocess_image(self, img_path):
        try:
            img = tf.keras.preprocessing.image.load_img(
                img_path, 
                target_size=(self.image_size, self.image_size)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict(self, img_path):
        try:
            img_array = self.preprocess_image(img_path)
            predictions = self.model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = round(100 * predictions[0][class_idx], 2)
            predicted_class = self.class_names[class_idx]
            
            logger.info(f"Prediction made: {predicted_class} ({confidence}%)")
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize classifier
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
classifier = PotatoDiseaseClassifier("potato.keras", CLASS_NAMES)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Service is running'})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            predicted_class, confidence = classifier.predict(filepath)
            
            # Clean up - remove uploaded file
            os.remove(filepath)
            
            # Return prediction
            return jsonify({
                'status': 'success',
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)