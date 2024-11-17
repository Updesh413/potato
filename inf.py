import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging

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

    def visualize_prediction(self, img_path, save_path=None):
        try:
            img = tf.keras.preprocessing.image.load_img(img_path)
            predicted_class, confidence = self.predict(img_path)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence}%")
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing prediction: {str(e)}")
            raise

# Initialize classifier with class names
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
classifier = PotatoDiseaseClassifier("potato.keras", CLASS_NAMES)

# Specify your image path here
img_path = r"Dataset\PotatoPlants\Potato___Early_blight\fdc691b0-2b15-4cb6-8f5d-c4e5654389e0___RS_Early.B 7935.JPG"  # Replace with your image path

# Make prediction and visualize
predicted_class, confidence = classifier.predict(img_path)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence}%")

# Show the visualization
classifier.visualize_prediction(img_path)