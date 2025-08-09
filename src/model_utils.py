# Model loading and prediction functions

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from typing import Dict, Tuple, Optional
import os


class CatDogClassifier:
    """Handles model loading and predictions"""
    
    def __init__(self, model_path: str = 'cat_dog_cnn_model.h5'):
        """Load the trained model"""
        self.model_path = model_path
        self.model = None
        self.img_height = 150
        self.img_width = 150
        self.class_names = ['Cat', 'Dog']
        self.load_model()
    
    def load_model(self) -> None:
        """Load TensorFlow model from file"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = load_model(self.model_path)
        print(f"✅ Model loaded successfully from {self.model_path}")
    
    def preprocess_image(self, img_path: str) -> np.ndarray:
        """Resize image to 150x150 and normalize pixels"""
        img = image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    
    def predict(self, img_path: str) -> Dict:
        """Get cat/dog prediction with confidence score"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare image for model
        processed_img = self.preprocess_image(img_path)
        
        # Get model output
        prediction = self.model.predict(processed_img, verbose=0)
        probability = float(prediction[0][0])
        
        # Convert probability to class
        if probability > 0.5:
            predicted_class = 'Dog'
            confidence = probability
        else:
            predicted_class = 'Cat'
            confidence = 1 - probability
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probability_dog': probability,
            'probability_cat': 1 - probability
        }
    
    def predict_batch(self, image_paths: list) -> list:
        """Process multiple images and return results"""
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                result['image_path'] = img_path
                result['status'] = 'success'
            except Exception as e:
                result = {
                    'image_path': img_path,
                    'status': 'error',
                    'error': str(e)
                }
            results.append(result)
        return results
    
    def get_model_summary(self) -> Dict:
        """Return basic model info"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        total_params = self.model.count_params()
        
        return {
            'total_parameters': total_params,
            'input_shape': (self.img_height, self.img_width, 3),
            'num_layers': len(self.model.layers),
            'architecture': 'CNN with 4 Conv blocks + Dense layers'
        }


def get_prediction_emoji(predicted_class: str) -> str:
    """Cat or dog emoji for UI"""
    return "🐱" if predicted_class == "Cat" else "🐕"


def format_confidence(confidence: float) -> str:
    """Convert decimal to percentage string"""
    return f"{confidence:.1%}"